"""Serve the vnoiser event curation dashboard to browsers over HTTP.

The same dashboard :mod:`mbo_utilities.gui.curation_viewer` draws in a
desktop window or a notebook cell, on rendercanvas's ``http`` backend: the
process renders with wgpu wherever it runs (a lab server, a cluster node)
and streams JPEG frames over a websocket to any browser that opens the URL,
which sends its pointer and key events back. Nothing is installed on the
user's machine; the page is a full-window canvas.

- One canvas per process, several viewers per canvas: everyone connected
  sees the same frames; the longest-connected client is the one whose input
  counts (the page's badge says ``active`` or ``passive``). One process per
  user is how a hub would scale this, like a kernel per notebook.
- No authentication: bind to ``127.0.0.1`` and tunnel (``ssh -L``), or put
  it behind a proxy that authenticates. ``--host 0.0.0.0`` exposes it to
  the network as is.

Usage:
    mbo curate <path> --serve [--host 127.0.0.1] [--port 60649]
    python -m mbo_utilities.gui.curation_server <path> [--host ..] [--port ..]

    server = CurationServer(path)      # in code / an asyncio app
    await server.serve_async()         # or app.mount("/curation", server.asgi)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

__all__ = ["CurationServer", "PAGE_HTML", "main", "serve_curation"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 60649
DEFAULT_SIZE = (1500, 900)

# The client JS wants a '#canvas' wrapper and an optional '#status' badge;
# it loads renderview.js / renderview-client.js / renderview.css from the
# same path. The wrapper's size must be an INLINE style: renderview's resize
# observer pins a wrapper without one to its first pixel size (and caps it
# at 90vmin), which is what a stylesheet rule would get. 100vw x 100vh
# fills the browser window and follows it; the dashboard's edge window
# follows the canvas. 'is-resizable' shows renderview's drag corner, which
# writes a pixel size into the same inline style; 'fit window' puts the
# viewport size back.
PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>$title</title>
    <script type='module' src='renderview.js'></script>
    <script type='module' src='renderview-client.js'></script>
    <link rel='stylesheet' href='renderview.css'>
    <style>
        html, body { margin: 0; height: 100%; overflow: hidden; background: #1e1f22; color: #ddd;
                     font-family: system-ui, sans-serif; }
        #canvas p { width: 100%; height: 100%; margin: 0; display: flex; justify-content: center;
                    align-items: center; color: #888; }
        #canvas .renderview-resizer { width: 18px; height: 18px;
                    background: linear-gradient(135deg, transparent 55%, rgba(200, 200, 200, 0.7) 55%); }
        .badge { position: fixed; bottom: 6px; padding: 2px 8px; font: 12px monospace; color: #bbb;
                 background: rgba(0, 0, 0, 0.45); border-radius: 4px; }
        #status { right: 8px; }
        #fit { right: 8px; bottom: 30px; cursor: pointer; }
        .badge button { font: inherit; margin-left: 6px; }
    </style>
</head>
<body>
    <div id='canvas' class='is-resizable' style='display:block; width:100vw; height:100vh; --line-thickness:0'>
        <p>Connecting to the curation dashboard ...</p>
    </div>
    <div id='fit' class='badge' title='drag the bottom-right corner to resize; this fits the window again'
         onclick="const c = document.getElementById('canvas'); c.style.width = '100vw'; c.style.height = '100vh';">
        fit window</div>
    <div id='status' class='badge'></div>
</body>
</html>
"""


class CurationServer:
    """The curation dashboard on an ``http`` rendercanvas, ready to serve.

    Building it renders nothing yet: frames are drawn once a browser
    connects and reports its size. ``serve()`` runs uvicorn in this thread
    until interrupted; ``serve_async()`` does the same inside a running
    asyncio loop; ``asgi`` mounts into a larger ASGI app (FastAPI, Starlette).

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

    def __init__(self, path=None, *, channel: int = 0, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT, size=DEFAULT_SIZE, title: str | None = None,
                 logger: logging.Logger | None = None):
        # the auto backend is never used here, but fastplotlib resolves it at
        # import and the desktop branch of the figure helper would ask Qt for
        # a screen: a server has none
        os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")
        from rendercanvas.http import asgi, resources

        from mbo_utilities.gui.curation_viewer import CurationVis

        self.logger = logger or logging.getLogger("curation_server")
        self.host = str(host)
        self.port = int(port)
        self.vis = CurationVis(path, channel=channel, canvas="http", size=tuple(size), logger=self.logger)
        self.title = title or self.vis.title
        resources["index.html"] = "text/html", PAGE_HTML.replace("$title", self.title)
        self.asgi = asgi
        self._closed = False
        self.vis.show()

    @property
    def widget(self):
        """The :class:`EventCurationWidget` being served."""
        return self.vis.widget

    @property
    def canvas(self):
        return self.vis.figure.canvas

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def clients(self) -> int:
        """How many browsers are connected."""
        return int(self.asgi.get_count())

    def _uvicorn_config(self, **kwargs):
        import uvicorn

        return uvicorn.Config(
            self.asgi, host=self.host, port=self.port, log_level="warning", **kwargs,
        )

    def serve(self) -> None:
        """Run the server in this thread until it is interrupted."""
        import uvicorn

        self.logger.warning("curation dashboard at %s (ctrl+c stops)", self.url)
        print(f"curation dashboard: {self.url}", flush=True)
        try:
            uvicorn.Server(self._uvicorn_config()).run()
        finally:
            self.close()

    async def serve_async(self) -> None:
        """Run the server inside the current asyncio loop until it is stopped."""
        import uvicorn

        self.logger.warning("curation dashboard at %s", self.url)
        try:
            await uvicorn.Server(self._uvicorn_config()).serve()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.vis.close()


def serve_curation(path=None, *, channel: int = 0, host: str = DEFAULT_HOST,
                   port: int = DEFAULT_PORT, size=DEFAULT_SIZE) -> None:
    """Serve the dashboard on ``path`` until interrupted."""
    CurationServer(path, channel=channel, host=host, port=port, size=size).serve()


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", type=Path,
                    help="Data / animal / experiment / PF folder, a .mat, or a line-scan .mesc "
                         "(default: the last data path)")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"interface to listen on (default {DEFAULT_HOST}; 0.0.0.0 for the network)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--channel", type=int, default=0, help="channel averaged for raw line scans")
    ap.add_argument("--size", type=_size, default=DEFAULT_SIZE, metavar="WxH",
                    help="canvas size until a browser reports its window (default 1500x900)")
    args = ap.parse_args(argv)
    serve_curation(args.path, channel=args.channel, host=args.host, port=args.port, size=args.size)


if __name__ == "__main__":
    main()
