"""The Studio served to browsers: the http canvas, one session's server, the gateway.

``StudioServer`` puts the viewer on a ``BrowserCanvas``; the tests drive its
ASGI app as uvicorn would, with the fake browser in ``tests.asgi_client``.
The gateway is driven the same way. Its end-to-end test starts a real
Studio process behind it and is marked slow.
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
import tifffile

from tests.asgi_client import Browser, Lifespan, frame_of_size, get, request, ws_reply

pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

KEY = [(b"x-mbo-session-key", b"sesame")]


@pytest.fixture
def movie(tmp_path):
    t = np.arange(40)[:, None, None]
    y = np.arange(96)[None, :, None]
    x = np.arange(128)[None, None, :]
    data = 1000 + 500 * np.sin((x + 3 * t) / 9.0) * np.cos((y - 2 * t) / 11.0)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, data.astype(np.int16))
    return path


def test_the_page_escapes_its_title_and_links_home_only_when_asked():
    from mbo_utilities.gui.canvas_server import client_js, page_html

    page = page_html("a <b> & c", "Opening ...", home="/")
    assert "<title>a &lt;b&gt; &amp; c</title>" in page
    assert "id='home'" in page and "href='/'" in page
    assert "id='home'" not in page_html("x", "y")
    # behind https (tailscale serve) the client must open wss://
    assert "location.protocol === 'https:' ? 'wss://'" in client_js()


async def _stream(server):
    status, _ = await get(server.asgi, "/")
    assert status == 403
    status, body = await get(server.asgi, "/", KEY)
    assert status == 200 and "movie.tif" in body.decode()
    status, body = await get(server.asgi, "/renderview-client.js", KEY)
    assert status == 200 and b"wss://" in body
    assert await ws_reply(server.asgi, "/") == {"type": "websocket.close", "code": 1008}

    lifespan = Lifespan(server.asgi)
    await lifespan.start()
    browser = Browser(server.asgi, headers=KEY)
    await browser.connect()
    # before the browser reports a size the canvas keeps the one it was made at
    await frame_of_size(browser, (900, 640), tries=1)

    # a HiDPI browser gets frames at one pixel per css pixel
    await browser.event(
        type="resize", width=800, height=600, pwidth=1600, pheight=1200, ratio=2
    )
    await frame_of_size(browser, (800, 600))
    assert tuple(server.canvas.get_physical_size()) == (800, 600)

    # once the view stops changing, one lossless frame follows, then nothing
    deadline = time.monotonic() + 15
    while True:
        msg, _ = await browser.next_frame(timeout=15)
        if msg["mimetype"] == "image/png":
            break
        assert time.monotonic() < deadline, "the still view never went lossless"
    with pytest.raises(TimeoutError):
        await browser.next_frame(timeout=1.0)

    # a second tab that connects now still gets the frame on screen
    other = Browser(server.asgi, headers=KEY)
    await other.connect()
    msg, _ = await other.next_frame(timeout=10)
    assert msg["mimetype"] == "image/png"
    await other.disconnect()

    await browser.disconnect()
    server.close()
    # the rendercanvas loop notices the closed canvas and stops
    await asyncio.sleep(0.3)
    await lifespan.stop()


def test_a_session_streams_at_its_size_behind_its_key_and_only_when_it_changes(movie):
    from mbo_utilities.gui.studio_server import StudioServer

    server = StudioServer(movie, size=(900, 640), key="sesame")
    try:
        asyncio.run(_stream(server))
    finally:
        server.close()


def test_browse_lists_folders_first_and_hides_dot_entries(tmp_path):
    from mbo_utilities.gui.gateway import browse

    (tmp_path / "b_dir").mkdir()
    (tmp_path / "A_dir").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "a.tif").write_bytes(b"x" * 10)
    listing = browse(tmp_path)
    assert [e["name"] for e in listing["entries"]] == ["A_dir", "b_dir", "a.tif"]
    assert listing["entries"][2]["size"] == 10
    assert listing["crumbs"][-1]["path"] == str(tmp_path.resolve())
    # a file lists the folder holding it
    assert browse(tmp_path / "a.tif")["path"] == str(tmp_path.resolve())


async def _token_and_refusals(gateway, movie):
    status, _ = await get(gateway, "/")
    assert status == 403
    status, _ = await get(gateway, "/?token=wrong")
    assert status == 403
    status, headers, _ = await request(gateway, "GET", "/s/abc/?token=t0ken")
    assert status == 303 and headers[b"location"] == b"/s/abc/"
    assert headers[b"set-cookie"].startswith(b"mbo_gateway_8899=t0ken;")
    cookie = [(b"cookie", b"mbo_gateway_8899=t0ken")]

    status, body = await get(gateway, "/", cookie)
    assert status == 200 and b"Miller Brain Studio" in body
    status, body = await get(gateway, "/api/sessions", cookie)
    assert json.loads(body) == {
        "sessions": [],
        "max_sessions": 0,
        "idle_timeout": 600.0,
    }
    status, body = await get(gateway, f"/api/browse?path={movie.parent}", cookie)
    assert [e["name"] for e in json.loads(body)["entries"]] == ["movie.tif"]

    for payload, code in (
        ({"path": str(movie.parent / "missing.tif")}, 400),
        ({"path": str(movie), "widget": "nope"}, 400),
        ({"path": str(movie)}, 409),  # max_sessions is 0
    ):
        status, _, body = await request(
            gateway, "POST", "/api/sessions", json.dumps(payload).encode(), cookie
        )
        assert status == code, body
    assert gateway.sessions == {}

    status, _ = await get(gateway, "/s/nope/", cookie)
    assert status == 404
    refused = {"type": "websocket.close", "code": 1008}
    assert await ws_reply(gateway, "/s/nope/", cookie) == refused


def test_the_gateway_wants_its_token_and_refuses_what_it_cannot_open(movie):
    from mbo_utilities.gui.gateway import Gateway

    gateway = Gateway(port=8899, max_sessions=0, token="t0ken")
    asyncio.run(_token_and_refusals(gateway, movie))


async def _session_through_gateway(gateway, movie):
    lifespan = Lifespan(gateway)
    await lifespan.start()
    session = gateway.open(movie)
    deadline = time.monotonic() + 180
    while session.state == "starting":
        assert time.monotonic() < deadline, session.log_tail()
        await asyncio.sleep(0.5)
    assert session.state == "ready", session.log_tail()

    status, body = await get(gateway, session.url)
    assert status == 200 and b"renderview-client.js" in body
    browser = Browser(gateway, path=session.url)
    await browser.connect()
    assert session.clients == 1
    await browser.resize(800, 600)
    await frame_of_size(browser, (800, 600))
    await browser.disconnect()
    assert session.clients == 0

    # no browser for longer than the idle timeout stops the process
    gateway.idle_timeout = 1.0
    deadline = time.monotonic() + 30
    while session.state != "exited":
        assert time.monotonic() < deadline
        await asyncio.sleep(0.5)
    assert session.proc.poll() is not None
    assert "no browser" in session.error
    await gateway.stop(session.id)
    assert session.id not in gateway.sessions
    await lifespan.stop()


@pytest.mark.slow
def test_a_session_runs_in_its_own_process_and_stops_when_left_idle(
    movie, tmp_path, monkeypatch
):
    from mbo_utilities.gui.gateway import Gateway

    # the session's log and the recent-files entry stay out of ~/.mbo
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    gateway = Gateway(port=8898, idle_timeout=0, token=None)
    try:
        asyncio.run(_session_through_gateway(gateway, movie))
    finally:
        for session in gateway.sessions.values():
            if session.proc.poll() is None:
                session.proc.kill()


def test_cli_serve_is_wired():
    from click.testing import CliRunner
    from mbo_utilities.cli import main

    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    for flag in (
        "--host",
        "--max-sessions",
        "--idle-timeout",
        "--quality",
        "--no-token",
    ):
        assert flag in result.output
