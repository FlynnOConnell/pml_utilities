"""Measure how fast ``mbo serve`` streams the Studio to a browser, and plot it.

``run`` is one browser tab on the live session of a running gateway: it sizes
the canvas, presses play, and counts the frames that arrive while the movie
plays. It appends one record per run to a json file. ``plot`` draws the frame
rate of every record in that file.

    python scripts/benchmark_serve.py run http://pml-w1:8800 --token T --label "tailnet"
    python scripts/benchmark_serve.py plot serve_benchmark.json docs/_images/serve --title ...

``run`` needs ``websockets`` (the ``server`` extra); ``plot`` needs matplotlib.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import time
import urllib.request
from datetime import datetime
from http.cookiejar import CookieJar
from pathlib import Path

# NDWidget's play/pause button sits this far above the canvas's bottom edge,
# 15 px from the left, when the movie has one slider (T)
PLAY_X = 15
PLAY_FROM_BOTTOM = 46
POINTER = {"buttons": [], "modifiers": [], "ntouches": 0, "touches": {}}


def login(base: str, token: str | None):
    """An opener holding the gateway's cookie, and that cookie as a header value."""
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    query = f"?token={token}" if token else ""
    with opener.open(f"{base}/{query}", timeout=30):
        pass
    return opener, "; ".join(f"{c.name}={c.value}" for c in jar)


def ready_session(opener, base: str, path: str | None, timeout: float = 300) -> dict:
    """The gateway's live session, opening ``path`` first when given."""
    if path:
        request = urllib.request.Request(
            f"{base}/api/sessions",
            data=json.dumps({"path": path}).encode(),
            headers={"content-type": "application/json"},
        )
        with opener.open(request, timeout=30):
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with opener.open(f"{base}/api/sessions", timeout=30) as r:
            live = [s for s in json.load(r)["sessions"] if s["state"] != "exited"]
        if live and live[0]["state"] == "ready":
            return live[0]
        time.sleep(1)
    raise TimeoutError(f"no ready session at {base}")


async def send_event(ws, **event) -> None:
    event.setdefault("timestamp", 0)
    await ws.send(json.dumps(event))


async def next_frame(ws, timeout: float) -> bytes:
    """The next frame's first buffer, acknowledged as renderview-client.js does."""
    while True:
        message = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(message, bytes):
            continue
        msg = json.loads(message)
        if msg.get("type") != "framebufferdata":
            continue
        buffers = [
            await asyncio.wait_for(ws.recv(), timeout)
            for _ in range(msg.get("nbuffers", 1))
        ]
        await send_event(
            ws,
            type="_framefeedback",
            index=msg["index"],
            localtime=time.time(),
            timestamp=msg["timestamp"],
        )
        return buffers[0]


async def frames_for(ws, seconds: float) -> list[bytes]:
    frames = []
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        try:
            frames.append(await next_frame(ws, left))
        except TimeoutError:
            break
    return frames


async def still(ws, quiet: float = 3.0, limit: float = 120.0) -> bool:
    """Read frames until none arrives for ``quiet`` seconds; False if they never stop."""
    end = time.monotonic() + limit
    while time.monotonic() < end:
        try:
            await next_frame(ws, quiet)
        except TimeoutError:
            return True
    return False


async def press(ws, x: int, y: int) -> None:
    """A click where the pointer already is, so no hover redraw counts as a frame."""
    await send_event(
        ws, type="pointer_down", x=x, y=y, button=1, **{**POINTER, "buttons": [1]}
    )
    await send_event(ws, type="pointer_up", x=x, y=y, button=1, **POINTER)


async def trial(ws, play, seconds: float, first_timeout: float) -> dict:
    """Play for ``seconds`` from the first frame after pressing play, then pause."""
    # a pause that did not land leaves the movie playing: press again
    if not await still(ws):
        await press(ws, *play)
        if not await still(ws):
            raise RuntimeError("the view never went still")
    await press(ws, *play)
    pressed = time.monotonic()
    frames = []
    try:
        frames.append(await next_frame(ws, first_timeout))
    except TimeoutError:
        pass
    first = time.monotonic() - pressed
    if frames:
        frames += await frames_for(ws, seconds)
    await press(ws, *play)
    return {
        "frames": len(frames),
        "distinct": len({hashlib.md5(f).digest() for f in frames}),
        "bytes": sum(len(f) for f in frames),
        "seconds": seconds,
        "first_frame_s": round(first, 3) if frames else None,
    }


async def measure(url, cookie, size, seconds, trials, first_timeout) -> list[dict]:
    from websockets.asyncio.client import connect

    width, height = size
    play = (PLAY_X, height - PLAY_FROM_BOTTOM)
    results = []
    async with connect(
        url, additional_headers={"cookie": cookie}, max_size=None, compression=None
    ) as ws:
        await send_event(
            ws,
            type="resize",
            width=width,
            height=height,
            pwidth=width,
            pheight=height,
            ratio=1,
        )
        await send_event(
            ws, type="pointer_move", x=play[0], y=play[1], button=0, **POINTER
        )
        for i in range(trials):
            result = await trial(ws, play, seconds, first_timeout)
            results.append(result)
            nbytes, n = result["bytes"], result["frames"]
            print(
                f"trial {i + 1}: {n / seconds:5.1f} frames/s, "
                f"{nbytes / max(1, n) / 1e3:4.0f} KB/frame, "
                f"{nbytes / seconds / 1e6:5.2f} MB/s, first frame after "
                f"{result['first_frame_s']} s"
            )
        await still(ws)
    return results


def alone(opener, base: str, sid: str, timeout: float = 180) -> float:
    """Wait until no other browser is on the session; the seconds it took.

    rendercanvas takes events only from the longest-connected browser, so a
    tab still closing would swallow every click of this one.
    """
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        with opener.open(f"{base}/api/sessions", timeout=30) as r:
            session = next(s for s in json.load(r)["sessions"] if s["id"] == sid)
        if session["clients"] == 0:
            return time.monotonic() - started
        time.sleep(1)
    raise TimeoutError(f"another browser stayed on session {sid}")


def run(args) -> None:
    base = args.url.rstrip("/")
    opener, cookie = login(base, args.token)
    session = ready_session(opener, base, args.open)
    waited = alone(opener, base, session["id"])
    if waited > 1:
        print(f"waited {waited:.0f} s for another browser to leave")
    ws_url = (
        base.replace("http://", "ws://").replace("https://", "wss://") + session["url"]
    )
    size = tuple(int(v) for v in args.size.split("x"))
    trials = asyncio.run(
        measure(ws_url, cookie, size, args.seconds, args.trials, args.first_timeout)
    )
    record = {
        "label": args.label,
        "link": args.link,
        "client": socket.gethostname(),
        "gateway": base.split("//")[-1].split(":")[0],
        "file": session["name"],
        "size": list(size),
        "date": datetime.now().isoformat(timespec="seconds"),
        "trials": trials,
    }
    out = Path(args.out)
    records = json.loads(out.read_text()) if out.exists() else []
    records.append(record)
    out.write_text(json.dumps(records, indent=1))


def plot(args) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams

    runs = json.loads(Path(args.results).read_text())
    # runs sharing a label pool their trials into one bar, in first-run order
    records = {}
    for run in runs:
        rec = records.setdefault(run["label"], {**run, "trials": []})
        rec["trials"] += run["trials"]
    records = list(records.values())
    rcParams.update(
        {
            "font.family": ["Arial", "Nimbus Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 11,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"local": "#2a78d6", "network": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7.5, 1.0 + 0.8 * len(records)))
    labels = []
    for i, rec in enumerate(records):
        fps = np.array([t["frames"] / t["seconds"] for t in rec["trials"]])
        frames = sum(t["frames"] for t in rec["trials"])
        kb = sum(t["bytes"] for t in rec["trials"]) / max(1, frames) / 1e3
        color = colors["local"] if rec["link"] == "none" else colors["network"]
        y = len(records) - 1 - i
        ax.barh(y, fps.mean(), height=0.62, color=color, linewidth=0)
        ax.scatter(
            fps,
            np.full(fps.size, y),
            s=14,
            color="white",
            edgecolor="#333",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(
            max(fps.max(), fps.mean()) + 0.6,
            y,
            f"{fps.mean():.1f} fps",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        labels.append(
            ax.annotate(
                rec["label"],
                (0, y),
                xytext=(-8, 6),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=11,
                annotation_clip=False,
            )
        )
        ax.annotate(
            f"{kb:.0f} KB per frame",
            (0, y),
            xytext=(-8, -8),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color="#666",
            annotation_clip=False,
        )
    # the Studio plays a movie at 30 frames per second; nothing can arrive faster
    ax.axvline(30, color="#999", linewidth=0.75, zorder=0)
    ax.text(
        30,
        len(records) - 0.45,
        "playback rate",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666",
    )
    ax.set_yticks([])
    ax.set_ylim(-0.5, len(records) - 0.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 32)
    ax.set_xlabel("frames per second reaching the browser while a movie plays")
    # title and caption start where the row labels do, not at the axes
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    left = min(to_fig.transform(t.get_window_extent())[0, 0] for t in labels)
    top = to_fig.transform(ax.get_window_extent())[1, 1]
    point = 1 / (72 * fig.get_figheight())
    fig.text(
        left, top + 18 * point, args.title, fontsize=13, fontweight="bold", va="bottom"
    )
    if args.note:
        bottom = to_fig.transform(ax.xaxis.label.get_window_extent())[0, 1]
        fig.text(
            left, bottom - 8 * point, args.note, fontsize=9, color="#666", va="top"
        )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(
            out / f"serve_playback_fps.{ext}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=False,
        )
    # nimbus sans stands in for arial on linux (same metrics); illustrator wants arial
    svg = out / "serve_playback_fps.svg"
    svg.write_text(svg.read_text().replace("Nimbus Sans", "Arial"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="measure one gateway session from this machine")
    p.add_argument("url", help="the gateway, e.g. http://pml-w1:8800")
    p.add_argument("--token", default=None)
    p.add_argument("--label", required=True, help="the bar's name in the plot")
    p.add_argument(
        "--link",
        default="none",
        help="what the frames cross: none (client on the server), tailnet relay, ...",
    )
    p.add_argument("--out", default="serve_benchmark.json")
    p.add_argument("--open", default=None, help="open this path in a new session first")
    p.add_argument("--size", default="1400x900", help="browser canvas, css px")
    p.add_argument("--seconds", type=float, default=10.0, help="per trial")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument(
        "--first-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for the first frame after play",
    )
    p.set_defaults(func=run)
    p = sub.add_parser("plot", help="bar chart of every run in a results file")
    p.add_argument("results")
    p.add_argument("out_dir")
    p.add_argument("--title", default="Studio playback in a browser")
    p.add_argument("--note", default="", help="a caption under the chart")
    p.set_defaults(func=plot)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
