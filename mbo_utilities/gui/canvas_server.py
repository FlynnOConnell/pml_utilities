"""Serve one fastplotlib figure to browsers over HTTP, one figure per process.

rendercanvas's ``http`` backend streams the canvas as JPEG frames over a
websocket and takes the browser's pointer and key events back. A
:class:`CanvasServer` puts uvicorn in front of it. The curation dashboard
(``curation_server``) and the Studio (``studio_server``) are served this way;
``gateway`` runs one Studio process per browser session behind one port.

A server process must call :func:`select_http_backend` before fastplotlib is
imported, so fastplotlib's loop is the one the canvas runs on.
"""

from __future__ import annotations

import hmac
import html
import os
import sys
from string import Template

import numpy as np
from rendercanvas.http import HttpRenderCanvas, resources
from rendercanvas.http import asgi as canvas_app

from mbo_utilities import log

__all__ = [
    "SESSION_KEY_ENV",
    "SESSION_KEY_HEADER",
    "BrowserCanvas",
    "CanvasServer",
    "client_js",
    "page_html",
    "require_uvicorn",
    "select_http_backend",
]

logger = log.get("gui.canvas_server")

# a gateway's session process accepts only requests carrying its key, so no
# other local user can reach it on its localhost port
SESSION_KEY_ENV = "MBO_SESSION_KEY"
SESSION_KEY_HEADER = "x-mbo-session-key"

_PAGE = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>$title</title>
    <script type='module' src='renderview.js'></script>
    <script type='module' src='renderview-client.js'></script>
    <link rel='stylesheet' href='renderview.css'>
    <style>
        html, body { margin: 0; height: 100%; overflow: auto; background: #1e1f22; color: #ddd;
                     font-family: system-ui, sans-serif; }
        #canvas p { width: 100%; height: 100%; margin: 0; display: flex; justify-content: center;
                    align-items: center; color: #888; }
        #canvas .renderview-resizer { width: 18px; height: 18px;
                    background: linear-gradient(135deg, transparent 55%, rgba(200, 200, 200, 0.7) 55%); }
        .badge { position: fixed; bottom: 6px; padding: 2px 8px; font: 12px monospace; color: #bbb;
                 background: rgba(0, 0, 0, 0.45); border-radius: 4px; text-decoration: none; }
        #status { right: 8px; }
        #fit { right: 8px; bottom: 30px; cursor: pointer; }
        #home { left: 8px; }
        .badge button { font: inherit; margin-left: 6px; }
    </style>
</head>
<body>
    <div id='canvas' class='is-resizable' style='display:block; width:100vw; height:100vh; --line-thickness:0'>
        <p>$waiting</p>
    </div>
    <div id='fit' class='badge' title='drag the bottom-right corner to resize; this fits the window again'
         onclick='fitWindow()'>fit window</div>
    $home
    <div id='status' class='badge'></div>
    <script>
        function fitWindow() {
            const c = document.getElementById('canvas');
            c.style.width = '100vw';
            c.style.height = '100vh';
        }
        // the app can set the canvas to a fixed size (its minimum layout);
        // a browser window resized afterwards takes the canvas with it again
        window.addEventListener('resize', fitWindow);
    </script>
</body>
</html>
""")


def page_html(title: str, waiting: str, home: str | None = None) -> str:
    """The page holding the canvas: full window, a fit badge, the connection status.

    ``waiting`` shows until the first frame arrives; ``home`` adds a badge
    linking back there (the gateway's session list).
    """
    link = (
        f"<a id='home' class='badge' href='{home}' title='all sessions'>sessions</a>"
        if home
        else ""
    )
    return _PAGE.substitute(
        title=html.escape(title), waiting=html.escape(waiting), home=link
    )


def client_js() -> str:
    """The rendercanvas browser client, opening ``wss://`` on a page served over https.

    The shipped client always opens ``ws://``, which a page behind an https
    proxy (``tailscale serve``) is not allowed to.
    """
    return resources["renderview-client.js"][1].replace(
        "new WebSocket('ws://'",
        "new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://')",
    )


def select_http_backend() -> None:
    """Make ``rendercanvas.auto``, and so fastplotlib's loop, the http backend.

    NDWidget fetches frames on fastplotlib's loop. Under the offscreen backend
    that loop never runs and scrubbing stalls; over ssh the auto choice is the
    terminal backend. Has no effect once ``rendercanvas.auto`` is imported.
    """
    if "rendercanvas.auto" in sys.modules:
        logger.warning(
            "rendercanvas.auto is already imported; fastplotlib's loop stays on its backend"
        )
        return
    os.environ.pop("RENDERCANVAS_FORCE_OFFSCREEN", None)
    os.environ.pop("WGPU_FORCE_OFFSCREEN", None)
    os.environ["RENDERCANVAS_BACKEND"] = "http"


class BrowserCanvas(HttpRenderCanvas):
    """The http canvas, sending only frames that changed.

    Two departures from ``HttpRenderCanvas``. It keeps the size it was asked
    for until a browser reports its own; otherwise the first draw after a
    browser connects is on a 1 x 1 canvas, where every subplot's viewport is
    negative. And it drops a frame identical to the last one sent: an imgui
    figure draws every frame whether anything moved or not, which streamed
    at 60 fps of jpeg even while idle, and each draw request cancelled the
    lossless frame meant to follow the last change, so a still view never
    sharpened.

    Frames are rendered at no more than ``max_pixel_ratio`` physical pixels
    per css pixel (a HiDPI browser gets a frame it upscales, a quarter of the
    bytes), as jpeg of ``quality`` while they change.
    """

    def __init__(
        self, *args, quality: int = 80, max_pixel_ratio: float = 1.0, **kwargs
    ):
        self._last_array = None
        self._lossless_sent = False
        self.max_pixel_ratio = float(max_pixel_ratio)
        super().__init__(*args, **kwargs)

    def _on_event(self, event, id):
        ratio = event.get("ratio", 1.0) if event.get("type") == "resize" else 1.0
        if ratio > self.max_pixel_ratio:
            scale = self.max_pixel_ratio / ratio
            event = {
                **event,
                "pwidth": max(1, round(event["pwidth"] * scale)),
                "pheight": max(1, round(event["pheight"] * scale)),
                "ratio": self.max_pixel_ratio,
            }
        super()._on_event(event, id)

    def _rc_set_logical_size(self, width, height):
        super()._rc_set_logical_size(width, height)
        if canvas_app.get_count() == 0:
            self._size_info.set_physical_size(
                max(1, round(width)), max(1, round(height)), 1.0
            )

    def _rc_request_draw(self):
        # the base also cancels the pending lossless frame here; a changed
        # frame reschedules it in _encode_frame, an identical one must not
        if not self._draw_requested:
            self._draw_requested = True
            self._schedule_maybe_draw()

    def _encode_frame(self, array, is_lossless_redraw=False):
        if is_lossless_redraw:
            self._lossless_sent = True
        elif self._last_array is not None and np.array_equal(array, self._last_array):
            if self._lossless_draw_info is None and not self._lossless_sent:
                self._schedule_lossless_draw(self._last_array)
            # a browser that connected since still needs the frame
            self._send_last_frame_to_ready_clients()
            return
        else:
            # the bitmap may be a buffer the next frame reuses
            array = self._last_array = array.copy()
            self._lossless_sent = False
        super()._encode_frame(array, is_lossless_redraw)


def require_uvicorn():
    """uvicorn, or an ImportError naming the extra that installs it."""
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            'serving to browsers needs uvicorn: uv pip install "pml_utilities[server] @ '
            'git+https://github.com/FlynnOConnell/pml_utilities.git"'
        ) from e
    return uvicorn


class CanvasServer:
    """A shown figure on a :class:`BrowserCanvas`, ready for uvicorn to serve.

    Frames are drawn once a browser connects. ``serve()`` runs uvicorn in
    this thread until interrupted or the canvas closes; ``serve_async()``
    does the same inside a running asyncio loop; ``asgi`` mounts into a
    larger ASGI app.

    Parameters
    ----------
    vis
        What is served: an object with ``figure``, ``show()`` and ``close()``
        whose figure is on a :class:`BrowserCanvas`.
    title : str
        The page title.
    waiting : str
        What the page says until the first frame.
    host, port : str, int
        Where ``serve`` listens.
    key : str, optional
        Accept only requests whose ``x-mbo-session-key`` header carries it.
    """

    def __init__(
        self,
        vis,
        *,
        title: str,
        waiting: str,
        host: str,
        port: int,
        key: str | None = None,
    ):
        self.vis = vis
        self.title = title
        self.host = str(host)
        self.port = int(port)
        self.key = key
        resources["index.html"] = "text/html", page_html(title, waiting)
        resources["renderview-client.js"] = "text/javascript", client_js()
        self._server = None
        self._closed = False
        self.vis.show()
        self.canvas.add_event_handler(self.stop, "close")

    @property
    def figure(self):
        return self.vis.figure

    @property
    def canvas(self):
        return self.vis.figure.canvas

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def clients(self) -> int:
        """How many browsers are connected."""
        return int(canvas_app.get_count())

    async def asgi(self, scope, receive, send):
        """The ASGI app: rendercanvas's, behind the session key when there is one."""
        if self.key is not None and scope["type"] in ("http", "websocket"):
            got = dict(scope.get("headers") or []).get(SESSION_KEY_HEADER.encode(), b"")
            if not hmac.compare_digest(got, self.key.encode()):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": b"Forbidden"})
                return
        await canvas_app(scope, receive, send)

    def _config(self, uvicorn, **kwargs):
        # a bound method reads as asgi2 to uvicorn's sniffing; frames are
        # jpeg, deflating them costs cpu for nothing
        return uvicorn.Config(
            self.asgi,
            host=self.host,
            port=self.port,
            log_level="warning",
            interface="asgi3",
            ws_per_message_deflate=False,
            **kwargs,
        )

    def stop(self, event=None) -> None:
        """Ask a running ``serve`` to return; safe from any thread."""
        if self._server is not None:
            self._server.should_exit = True

    def serve(self) -> None:
        """Run the server in this thread until it is interrupted or the canvas closes."""
        uvicorn = require_uvicorn()
        self._server = uvicorn.Server(self._config(uvicorn))
        logger.warning("%s at %s", self.title, self.url)
        try:
            self._server.run()
        finally:
            self.close()

    async def serve_async(self) -> None:
        """Run the server inside the current asyncio loop until it is stopped."""
        uvicorn = require_uvicorn()
        self._server = uvicorn.Server(self._config(uvicorn))
        logger.warning("%s at %s", self.title, self.url)
        try:
            await self._server.serve()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.vis.close()
