"""The curation dashboard served to browsers.

``CurationServer`` puts the dashboard on rendercanvas's ``http`` canvas,
whose ASGI app streams JPEG frames over a websocket and takes the browser's
events back. The tests drive that ASGI app directly, as uvicorn would: a
GET for the page, then a fake browser that connects, reports its size,
acknowledges frames and sends pointer and key events that must reach the
imgui dashboard. Skipped when vnoiser is not installed.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import simplejpeg

from tests.test_event_curation import _write_spatial_recording

vnoiser = pytest.importorskip("vnoiser")
pytest.importorskip("uvicorn")


@pytest.fixture(autouse=True)
def _keep_preferences(monkeypatch):
    from mbo_utilities.gui import event_curation

    monkeypatch.setattr(event_curation, "set_last_dir", lambda *a, **k: None)
    monkeypatch.setattr(event_curation, "get_last_dir", lambda *a, **k: None)


@pytest.fixture
def data_root(tmp_path):
    return _write_spatial_recording(tmp_path)


async def _get(app, path: str) -> tuple[int, bytes]:
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app({"type": "http", "method": "GET", "path": path, "headers": []}, receive, send)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return sent[0]["status"], body


class _Browser:
    """A fake browser tab on the websocket: what renderview-client.js does."""

    def __init__(self, app):
        self.app = app
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.frames: list[tuple[dict, list[bytes]]] = []
        self.task = None

    async def connect(self):
        async def receive():
            return await self.inbox.get()

        async def send(message):
            await self.outbox.put(message)

        self.task = asyncio.create_task(self.app({"type": "websocket", "path": "/"}, receive, send))
        first = await asyncio.wait_for(self.outbox.get(), 5)
        assert first["type"] == "websocket.accept"

    async def event(self, **event):
        event.setdefault("timestamp", 0)
        await self.inbox.put({"type": "websocket.receive", "text": json.dumps(event)})

    async def resize(self, width: int, height: int):
        await self.event(type="resize", width=width, height=height, pwidth=width, pheight=height, ratio=1)

    async def next_frame(self, timeout: float = 30.0) -> tuple[dict, list[bytes]]:
        """Wait for the next frame, acknowledge it as the client does."""
        while True:
            message = await asyncio.wait_for(self.outbox.get(), timeout)
            if "text" not in message:
                continue
            msg = json.loads(message["text"])
            if msg.get("type") != "framebufferdata":
                continue
            buffers = []
            for _ in range(msg["nbuffers"]):
                message = await asyncio.wait_for(self.outbox.get(), timeout)
                buffers.append(message["bytes"])
            self.frames.append((msg, buffers))
            await self.event(type="_framefeedback", index=msg["index"], localtime=0, timestamp=msg["timestamp"])
            return msg, buffers

    async def disconnect(self):
        await self.inbox.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(self.task, 5)


async def _frame_of_size(browser: _Browser, size: tuple[int, int], tries: int = 12):
    """The next frame whose JPEG is ``size`` (width, height) pixels."""
    for _ in range(tries):
        msg, buffers = await browser.next_frame()
        height, width, _cs, _ss = simplejpeg.decode_jpeg_header(buffers[0])
        if (width, height) == tuple(size):
            return msg, buffers
    raise AssertionError(f"no {size} frame; last was {(width, height)}")


class _Lifespan:
    """uvicorn's lifespan: startup kicks the rendercanvas loop on this asyncio loop."""

    def __init__(self, app):
        self.app = app
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.task = None

    async def start(self):
        async def receive():
            return await self.inbox.get()

        async def send(message):
            pass

        await self.inbox.put({"type": "lifespan.startup"})
        self.task = asyncio.create_task(self.app({"type": "lifespan"}, receive, send))

    async def stop(self):
        await self.inbox.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(self.task, 5)


def test_page_and_client_resources_are_served(data_root):
    from mbo_utilities.gui.curation_server import CurationServer

    server = CurationServer(data_root, size=(800, 500))
    try:
        assert server.url == "http://127.0.0.1:60649/"
        assert server.clients() == 0

        async def run():
            status, body = await _get(server.asgi, "/")
            assert status == 200
            page = body.decode()
            assert "renderview-client.js" in page and "id='canvas'" in page
            assert f"<title>{server.title}</title>" in page
            assert data_root.name in server.title
            for name in ("renderview.js", "renderview-client.js", "renderview.css"):
                status, body = await _get(server.asgi, f"/{name}")
                assert status == 200 and len(body) > 100
            status, _ = await _get(server.asgi, "/nope.txt")
            assert status == 404

        asyncio.run(run())
    finally:
        server.close()


def test_a_browser_gets_frames_and_its_input_reaches_the_dashboard(data_root):
    from mbo_utilities.gui.curation_server import CurationServer
    from mbo_utilities.gui.curation_viewer import FIGURE_MARGIN

    server = CurationServer(data_root, size=(900, 600))
    widget = server.widget
    widget.wait(60)
    assert widget.session is not None and widget.session.loaded, widget.status

    async def run():
        lifespan = _Lifespan(server.asgi)
        await lifespan.start()
        browser = _Browser(server.asgi)
        await browser.connect()
        assert server.clients() == 1
        # a real browser reports its size only after it connects, so the
        # first draw happens on the 1 x 1 canvas: it must not end the stream
        await asyncio.sleep(0.3)
        await browser.resize(900, 600)

        msg, buffers = await _frame_of_size(browser, (900, 600))
        assert msg["mimetype"] == "image/jpeg"
        assert tuple(server.canvas.get_logical_size()) == (900, 600)
        # the dashboard window follows the canvas
        assert server.vis.window.size == 600 - FIGURE_MARGIN

        # hover the dashboard, press k: the keybinds popup toggles
        assert not widget.show_keybinds
        await browser.event(type="pointer_move", x=300, y=200, button=0, buttons=[], modifiers=[], ntouches=0, touches={})
        await browser.next_frame()
        await browser.next_frame()
        await browser.event(type="key_down", key="k", modifiers=[])
        await browser.event(type="key_up", key="k", modifiers=[])
        for _ in range(6):
            await browser.next_frame()
            if widget.show_keybinds:
                break
        assert widget.show_keybinds

        # shrunk under the minimum the dashboard draws nothing but the
        # stream goes on; grown again (a browser window resize, the drag
        # corner) it renders at the new size
        await browser.resize(200, 100)
        for _ in range(3):
            await browser.next_frame()
        await browser.resize(1200, 700)
        await _frame_of_size(browser, (1200, 700))
        assert server.vis.window.size == 700 - FIGURE_MARGIN

        # a second tab sees the same frames but does not drive
        other = _Browser(server.asgi)
        await other.connect()
        assert server.clients() == 2
        await other.next_frame()
        await browser.next_frame()
        await other.disconnect()
        assert server.clients() == 1

        await browser.disconnect()
        assert server.clients() == 0
        server.close()
        await asyncio.sleep(0.3)  # the rendercanvas loop notices the closed canvas and stops
        await lifespan.stop()

    try:
        asyncio.run(run())
    finally:
        server.close()


def test_cli_and_module_entry_points_exist():
    from click.testing import CliRunner

    from mbo_utilities.cli import main
    from mbo_utilities.gui import curation_server

    result = CliRunner().invoke(main, ["curate", "--help"])
    assert result.exit_code == 0, result.output
    assert "--serve" in result.output and "--host" in result.output
    assert callable(curation_server.main)
