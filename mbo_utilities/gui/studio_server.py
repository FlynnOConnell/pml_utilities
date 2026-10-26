"""Serve Miller Brain Studio on one dataset to browsers over HTTP.

The viewer ``mbo PATH`` opens, on rendercanvas's ``http`` canvas instead of a
desktop window, so a browser on another machine drives it. One process holds
one Studio; ``gateway`` runs one per browser session.

Usage:
    mbo serve PATH                     # through the gateway (the usual way)
    python -m mbo_utilities.gui.studio_server PATH [--host ..] [--port ..]
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import psutil

from mbo_utilities.gpu import apply_persisted_compute_gpu
from mbo_utilities.gui.canvas_server import (
    SESSION_KEY_ENV,
    BrowserCanvas,
    CanvasServer,
    select_http_backend,
)

__all__ = ["StudioServer", "main", "serve_studio"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 60650
DEFAULT_SIZE = (1600, 1000)
WIDGETS = ("preview", "manualroi", "none")


class StudioServer(CanvasServer):
    """The Studio on ``path`` on an ``http`` rendercanvas, ready to serve.

    Opening the data happens here, before ``serve()`` listens, so a port
    that accepts connections is a Studio that is ready. File > Open in the
    viewer takes a typed path on this machine; native dialogs and windows
    of their own have nowhere to appear.

    Parameters
    ----------
    path
        Anything ``mbo PATH`` opens.
    widget : str
        ``"preview"``, ``"manualroi"`` or ``"none"``, as ``DataVis``.
    unit : int or str, optional
        Which measurement unit of a ``.mesc`` to open; the first otherwise.
    host, port : str, int
        Where ``serve`` listens.
    size : (int, int)
        Canvas size until the first browser reports its window size.
    key : str, optional
        Accept only requests carrying it (the gateway's session key).
    """

    def __init__(
        self,
        path,
        *,
        widget: str = "preview",
        unit: int | str | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        size=DEFAULT_SIZE,
        key: str | None = None,
    ):
        from masknmf.visualization.imgui.options import OPTIONS

        from mbo_utilities import __version__
        from mbo_utilities.gui.data_vis import DataVis

        # a second figure would open on this machine, where no one sees it
        OPTIONS.separate_image_window = False
        size = tuple(size)
        vis = DataVis(
            path,
            widget=widget,
            unit=unit,
            size=size,
            canvas=BrowserCanvas(size=size, max_fps=60.0),
        )
        name = Path(str(path[0] if isinstance(path, (list, tuple)) else path)).name
        super().__init__(
            vis,
            title=f"{name} - Miller Brain Studio v{__version__}",
            waiting=f"Opening {name} ...",
            host=host,
            port=port,
            key=key,
        )


def serve_studio(
    path,
    *,
    widget: str = "preview",
    unit: int | str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    size=DEFAULT_SIZE,
    key: str | None = None,
    parent_pid: int | None = None,
) -> None:
    """Serve the Studio on ``path`` until interrupted or its canvas closes.

    With ``parent_pid`` it also stops once that process is gone, so a
    gateway that dies leaves no Studio holding the GPU.
    """
    select_http_backend()
    # before any worker spawns, as run_gui does
    apply_persisted_compute_gpu()
    server = StudioServer(
        path, widget=widget, unit=unit, host=host, port=port, size=size, key=key
    )
    if parent_pid is not None:
        threading.Thread(
            target=_stop_with, args=(parent_pid, server), daemon=True
        ).start()
    server.serve()


def _stop_with(pid: int, server: StudioServer) -> None:
    while psutil.pid_exists(pid):
        time.sleep(5.0)
    server.stop()


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", type=Path, help="a file or folder mbo PATH opens")
    ap.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"interface to listen on (default {DEFAULT_HOST})",
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--widget", choices=WIDGETS, default="preview")
    ap.add_argument("--unit", default=None, help="the .mesc unit to open")
    ap.add_argument(
        "--size",
        type=_size,
        default=DEFAULT_SIZE,
        metavar="WxH",
        help="canvas size until a browser reports its window (default 1600x1000)",
    )
    ap.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="stop once this process is gone (the gateway that started it)",
    )
    args = ap.parse_args(argv)
    print(f"studio: http://{args.host}:{args.port}/", flush=True)
    serve_studio(
        args.path,
        widget=args.widget,
        unit=args.unit,
        host=args.host,
        port=args.port,
        size=args.size,
        key=os.environ.get(SESSION_KEY_ENV) or None,
        parent_pid=args.parent_pid,
    )


if __name__ == "__main__":
    main()
