"""
Serve the vnoiser event curation dashboard to browsers over HTTP.

Usage:
    mbo curate <path> --serve [--host 127.0.0.1] [--port 60649]
    python -m mbo_utilities.gui.curation_server <path> [--host ..] [--port ..]

    server = CurationServer(path)      # in code / an asyncio app
    await server.serve_async()         # or app.mount("/curation", server.asgi)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mbo_utilities import log
from mbo_utilities.gui.canvas_server import (
    BrowserCanvas,
    CanvasServer,
    select_http_backend,
)

__all__ = ["CurationServer", "main", "serve_curation"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 60649
DEFAULT_SIZE = (1500, 900)
WAITING = "Connecting to the curation dashboard ..."


class CurationServer(CanvasServer):
    """The curation dashboard on an ``http`` rendercanvas, ready to serve.

    Building it renders nothing yet: frames are drawn once a browser
    connects. ``serve()`` runs uvicorn in this thread until interrupted;
    ``serve_async()`` does the same inside a running asyncio loop; ``asgi``
    mounts into a larger ASGI app (FastAPI, Starlette).

    Parameters
    ----------
    path : optional
        What to curate; see :mod:`mbo_utilities.gui.curation_viewer`. None
        reopens the last data path.
    channel : int
        Channel averaged into a raw line scan's traces.
    host, port : str, int
        Where ``serve`` listens.
    size : (int, int)
        Canvas size until the first browser reports its window size.
    title : str, optional
        The page title; the dashboard's own by default.
    """

    def __init__(
        self,
        path=None,
        *,
        channel: int = 0,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        size=DEFAULT_SIZE,
        title: str | None = None,
        logger=None,
    ):
        from mbo_utilities.gui.curation_viewer import CurationVis

        size = tuple(size)
        vis = CurationVis(
            path,
            channel=channel,
            canvas=BrowserCanvas(size=size, max_fps=60.0),
            size=size,
            logger=logger or log.get("gui.curation_server"),
        )
        super().__init__(
            vis, title=title or vis.title, waiting=WAITING, host=host, port=port
        )

    @property
    def widget(self):
        """The :class:`EventCurationWidget` being served."""
        return self.vis.widget


def serve_curation(
    path=None,
    *,
    channel: int = 0,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    size=DEFAULT_SIZE,
) -> None:
    """Serve the dashboard on ``path`` until interrupted."""
    select_http_backend()
    CurationServer(path, channel=channel, host=host, port=port, size=size).serve()


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="a PF folder (or the experiment folder holding it), or a line-scan .mesc "
        "(default: the last data path)",
    )
    ap.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"interface to listen on (default {DEFAULT_HOST}; 0.0.0.0 for the network)",
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument(
        "--channel", type=int, default=0, help="channel averaged for raw line scans"
    )
    ap.add_argument(
        "--size",
        type=_size,
        default=DEFAULT_SIZE,
        metavar="WxH",
        help="canvas size until a browser reports its window (default 1500x900)",
    )
    args = ap.parse_args(argv)
    print(f"curation dashboard: http://{args.host}:{args.port}/", flush=True)
    serve_curation(
        args.path, channel=args.channel, host=args.host, port=args.port, size=args.size
    )


if __name__ == "__main__":
    main()
