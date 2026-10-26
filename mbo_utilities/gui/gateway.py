"""Serve Miller Brain Studio to browsers: one process per session, behind one port.

``mbo serve`` runs this. Each session is a ``studio_server`` process on a
localhost port of its own that answers only to its key. The gateway starts
it, serves its page under ``/s/<id>/``, carries its websocket, and stops it
once no browser has been connected for ``idle_timeout`` seconds. The
launcher at ``/`` stands in for the desktop file dialog: it browses this
machine's folders and opens a path in a new session.

The link printed at startup carries a token, traded for a cookie on first
use; nothing is served without one. Sessions run as the user who started the
gateway and read whatever that user can.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import html
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from http.cookies import SimpleCookie
from pathlib import Path
from string import Template
from urllib.parse import parse_qs

from rendercanvas.http import resources

from mbo_utilities import __version__, log
from mbo_utilities.gui.canvas_server import (
    SESSION_KEY_ENV,
    SESSION_KEY_HEADER,
    client_js,
    page_html,
    require_uvicorn,
)
from mbo_utilities.preferences import (
    add_recent_file,
    get_default_open_dir,
    get_mbo_dirs,
    get_recent_files,
)

__all__ = ["Gateway", "Session", "SessionLimitError"]

logger = log.get("gui.gateway")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8800
WIDGETS = ("preview", "manualroi", "none")
# a folder of thousands of tiffs lists this many; the path field takes the rest
MAX_ENTRIES = 2000
KEEP_EXITED = 20


class SessionLimitError(RuntimeError):
    """Opening another session would pass ``max_sessions``."""


class Session:
    """One Studio process: what it opened, where it listens, how many tabs it has."""

    def __init__(self, sid, path, widget, port, key, log_file, proc):
        self.id = sid
        self.path = path
        self.widget = widget
        self.port = port
        self.key = key
        self.log_file = log_file
        self.proc = proc
        self.started = time.time()
        self.state = "starting"
        self.clients = 0
        self.idle_since = time.monotonic()
        self.error = ""

    @property
    def name(self) -> str:
        return Path(self.path).name or str(self.path)

    @property
    def url(self) -> str:
        return f"/s/{self.id}/"

    def info(self) -> dict:
        return {
            "id": self.id,
            "path": str(self.path),
            "name": self.name,
            "widget": self.widget,
            "state": self.state,
            "clients": self.clients,
            "started": self.started,
            "pid": self.proc.pid,
            "error": self.error,
            "url": self.url,
        }

    def log_tail(self, nbytes: int = 8192) -> str:
        try:
            with self.log_file.open("rb") as f:
                f.seek(max(0, self.log_file.stat().st_size - nbytes))
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _listening(port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), 1.0
        )
    except (OSError, TimeoutError):
        return False
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return True


def _signal(proc: subprocess.Popen, sig) -> None:
    """Send ``sig`` to the session's process group (its own children with it)."""
    with contextlib.suppress(ProcessLookupError):
        if os.name == "posix":
            os.killpg(proc.pid, sig)
        elif sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()


def browse(path: str | os.PathLike | None) -> dict:
    """One folder's listing for the launcher: folders first, hidden entries left out.

    ``path`` may be a file (its folder is listed) or empty (the folder the
    desktop file dialog would open in).
    """
    p = Path(path).expanduser() if path else get_default_open_dir()
    if p.is_file():
        p = p.parent
    p = p.resolve()
    entries = []
    with os.scandir(p) as it:
        for e in it:
            if e.name.startswith("."):
                continue
            try:
                is_dir = e.is_dir()
                size = None if is_dir else e.stat().st_size
            except OSError:
                continue
            entries.append(
                {"name": e.name, "path": e.path, "dir": is_dir, "size": size}
            )
    entries.sort(key=lambda e: (not e["dir"], e["name"].casefold()))
    crumbs = [
        {"name": part.name or part.anchor, "path": str(part)}
        for part in reversed([p, *p.parents])
    ]
    return {
        "path": str(p),
        "parent": str(p.parent) if p.parent != p else None,
        "crumbs": crumbs,
        "entries": entries[:MAX_ENTRIES],
        "truncated": len(entries) > MAX_ENTRIES,
    }


class Gateway:
    """The ASGI app ``mbo serve`` runs: launcher, session pages, websocket proxy.

    Parameters
    ----------
    host, port : str, int
        Where ``serve`` listens.
    max_sessions : int
        How many Studio processes may run at once.
    idle_timeout : float
        Seconds a session lives with no browser connected; 0 never stops one.
    token : str or None
        What the printed link carries; None serves anyone who can connect.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_sessions: int = 1,
        idle_timeout: float = 600.0,
        token: str | None = None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.max_sessions = int(max_sessions)
        self.idle_timeout = float(idle_timeout)
        self.token = token
        self.sessions: dict[str, Session] = {}
        self._cookie = f"mbo_gateway_{self.port}"
        self._watcher = None

    def link(self, path: str = "/") -> str:
        """The address to open, token included."""
        host = socket.gethostname() if self.host in ("0.0.0.0", "::") else self.host
        query = f"?token={self.token}" if self.token else ""
        return f"http://{host}:{self.port}{path}{query}"

    def live(self) -> list[Session]:
        return [s for s in self.sessions.values() if s.state != "exited"]

    def open(self, path, widget: str = "preview") -> Session:
        """Start a Studio on ``path``; it is ready once the watcher sees it listen."""
        p = Path(str(path)).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"not found on {socket.gethostname()}: {p}")
        if widget not in WIDGETS:
            raise ValueError(f"widget must be one of {', '.join(WIDGETS)}")
        if len(self.live()) >= self.max_sessions:
            raise SessionLimitError(
                f"{len(self.live())} session(s) running, the limit is "
                f"{self.max_sessions}: stop one first"
            )
        sid = secrets.token_hex(4)
        port = _free_port()
        key = secrets.token_urlsafe(24)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = get_mbo_dirs()["logs"] / f"{stamp}_studio_{sid}.log"
        cmd = [
            sys.executable,
            "-m",
            "mbo_utilities.gui.studio_server",
            str(p),
            "--port",
            str(port),
            "--widget",
            widget,
            "--parent-pid",
            str(os.getpid()),
        ]
        env = {**os.environ, SESSION_KEY_ENV: key, "PYTHONUTF8": "1"}
        if os.name == "posix":
            group = {"start_new_session": True}
        else:
            group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        with log_file.open("ab") as out:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                **group,
            )
        session = Session(sid, p, widget, port, key, log_file, proc)
        self.sessions[sid] = session
        add_recent_file(str(p), file_type="folder" if p.is_dir() else "file")
        logger.info("session %s: %s (pid %d, log %s)", sid, p, proc.pid, log_file)
        return session

    async def stop(self, sid: str, reason: str = "stopped") -> None:
        """Stop a session's process, or forget one that has already exited."""
        session = self.sessions.get(sid)
        if session is None:
            return
        if session.state == "exited":
            del self.sessions[sid]
            return
        _signal(session.proc, signal.SIGTERM)
        for _ in range(100):
            if session.proc.poll() is not None:
                break
            await asyncio.sleep(0.1)
        else:
            _signal(session.proc, getattr(signal, "SIGKILL", signal.SIGTERM))
            session.proc.wait()
        session.state = "exited"
        session.error = reason
        logger.info("session %s %s", sid, reason)

    async def _watch(self) -> None:
        """Mark sessions ready, exited or idle, once a second."""
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            for session in list(self.sessions.values()):
                code = session.proc.poll()
                if session.state != "exited" and code is not None:
                    session.state = "exited"
                    session.error = f"the Studio exited with code {code}; see its log"
                    logger.warning("session %s exited with code %s", session.id, code)
                elif session.state == "starting" and await _listening(session.port):
                    session.state = "ready"
                    session.idle_since = now
                elif (
                    session.state == "ready"
                    and session.clients == 0
                    and self.idle_timeout > 0
                    and now - session.idle_since > self.idle_timeout
                ):
                    minutes = self.idle_timeout / 60
                    await self.stop(
                        session.id, f"stopped after {minutes:g} min with no browser"
                    )
            exited = [s.id for s in self.sessions.values() if s.state == "exited"]
            for sid in exited[:-KEEP_EXITED]:
                del self.sessions[sid]

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
        elif scope["type"] == "http":
            await self._http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._websocket(scope, receive, send)

    async def _lifespan(self, receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                self._watcher = asyncio.create_task(self._watch())
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._watcher is not None:
                    self._watcher.cancel()
                await asyncio.gather(
                    *(self.stop(s.id, "gateway stopped") for s in self.live())
                )
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _authorized(self, scope) -> bool:
        if self.token is None:
            return True
        cookie = SimpleCookie()
        for name, value in scope.get("headers") or []:
            if name == b"cookie":
                with contextlib.suppress(Exception):
                    cookie.load(value.decode("latin-1"))
        morsel = cookie.get(self._cookie)
        return morsel is not None and hmac.compare_digest(morsel.value, self.token)

    def _session_of(self, path: str) -> tuple[Session | None, str]:
        """The session a ``/s/<id>/<rest>`` path names, and ``rest``."""
        _, _, sid_rest = path.partition("/s/")
        sid, _, rest = sid_rest.partition("/")
        return self.sessions.get(sid), rest

    async def _http(self, scope, receive, send) -> None:
        path = scope["path"]
        method = scope["method"]
        query = parse_qs(scope.get("query_string", b"").decode())
        if self.token is not None and "token" in query:
            if hmac.compare_digest(query["token"][0], self.token):
                cookie = (
                    f"{self._cookie}={self.token}; Path=/; HttpOnly; SameSite=Strict"
                )
                await _respond(
                    send,
                    303,
                    b"",
                    headers=[
                        (b"location", path.encode()),
                        (b"set-cookie", cookie.encode()),
                    ],
                )
                return
        if not self._authorized(scope):
            await _respond(send, 403, _NO_TOKEN, "text/html; charset=utf-8")
            return

        if path == "/":
            await _respond(send, 200, self._launcher(), "text/html; charset=utf-8")
        elif path == "/api/sessions" and method == "GET":
            await _json(
                send,
                200,
                {
                    "sessions": [s.info() for s in self.sessions.values()],
                    "max_sessions": self.max_sessions,
                    "idle_timeout": self.idle_timeout,
                },
            )
        elif path == "/api/sessions" and method == "POST":
            await self._open_from(receive, send)
        elif (
            path.startswith("/api/sessions/")
            and path.endswith("/stop")
            and method == "POST"
        ):
            await self.stop(path.split("/")[3])
            await _json(send, 200, {"ok": True})
        elif path.startswith("/api/sessions/") and path.endswith("/log"):
            session = self.sessions.get(path.split("/")[3])
            if session is None:
                await _respond(send, 404, b"no such session")
            else:
                await _respond(send, 200, session.log_tail(200_000))
        elif path == "/api/browse":
            try:
                listing = browse(query.get("path", [""])[0])
            except OSError as e:
                await _json(send, 400, {"error": str(e)})
            else:
                await _json(send, 200, listing)
        elif path == "/api/recent":
            await _json(send, 200, {"recent": get_recent_files()[:12]})
        elif path.startswith("/s/"):
            await self._session_http(path, send)
        else:
            await _respond(send, 404, b"Not Found")

    async def _open_from(self, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        try:
            request = json.loads(body or b"{}")
            session = self.open(
                request.get("path", "").strip(), request.get("widget", "preview")
            )
        except SessionLimitError as e:
            await _json(send, 409, {"error": str(e)})
        except (ValueError, OSError) as e:
            await _json(send, 400, {"error": str(e)})
        else:
            await _json(send, 200, session.info())

    async def _session_http(self, path: str, send) -> None:
        session, rest = self._session_of(path)
        if session is None:
            await _respond(
                send,
                404,
                _status_page(
                    "No such session", "It may have been stopped.", "", refresh=False
                ),
                "text/html; charset=utf-8",
            )
        elif not path.startswith(f"/s/{session.id}/"):
            await _respond(
                send, 307, b"", headers=[(b"location", session.url.encode())]
            )
        elif rest in ("", "index.html"):
            if session.state == "ready":
                body = page_html(
                    f"{session.name} - Miller Brain Studio",
                    f"Connecting to {session.name} ...",
                    home="/",
                )
                await _respond(send, 200, body, "text/html; charset=utf-8")
            elif session.state == "starting":
                body = _status_page(
                    f"Opening {session.name}",
                    f"Starting the Studio on {socket.gethostname()}; this page reloads when it is ready.",
                    session.log_tail(),
                    refresh=True,
                )
                await _respond(send, 200, body, "text/html; charset=utf-8")
            else:
                body = _status_page(
                    f"{session.name}: {session.error or 'exited'}",
                    "Open it again from the sessions page.",
                    session.log_tail(),
                    refresh=False,
                )
                await _respond(send, 200, body, "text/html; charset=utf-8")
        elif rest == "renderview-client.js":
            await _respond(send, 200, client_js(), "text/javascript")
        elif rest in resources:
            content_type, body = resources[rest]
            await _respond(send, 200, body, content_type)
        else:
            await _respond(send, 404, b"Not Found")

    async def _websocket(self, scope, receive, send) -> None:
        session, _ = self._session_of(scope["path"])
        if not self._authorized(scope) or session is None or session.state != "ready":
            await send({"type": "websocket.close", "code": 1008})
            return
        try:
            from websockets.asyncio.client import connect
        except ImportError as e:
            raise ImportError(
                'mbo serve needs websockets: uv pip install "pml_utilities[server] @ '
                'git+https://github.com/FlynnOConnell/pml_utilities.git"'
            ) from e
        try:
            # frames are jpeg (a lossless png after a pause): no size cap, no deflate
            upstream = await connect(
                f"ws://127.0.0.1:{session.port}/",
                additional_headers={SESSION_KEY_HEADER: session.key},
                max_size=None,
                compression=None,
                ping_interval=None,
                open_timeout=10,
            )
        except (OSError, TimeoutError) as e:
            logger.warning("session %s: no websocket (%s)", session.id, e)
            await send({"type": "websocket.close", "code": 1011})
            return
        await send({"type": "websocket.accept"})
        session.clients += 1
        down = asyncio.create_task(_to_browser(upstream, send))
        up = asyncio.create_task(_to_studio(receive, upstream))
        try:
            _, pending = await asyncio.wait(
                {down, up}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            session.clients -= 1
            session.idle_since = time.monotonic()
            await upstream.close()

    def _launcher(self) -> str:
        return _LAUNCHER.substitute(
            host=html.escape(socket.gethostname()), version=html.escape(__version__)
        )

    def serve(self) -> None:
        """Run the gateway in this thread until interrupted; stops every session on the way out."""
        uvicorn = require_uvicorn()
        config = uvicorn.Config(
            self,
            host=self.host,
            port=self.port,
            log_level="warning",
            ws_per_message_deflate=False,
            lifespan="on",
        )
        uvicorn.Server(config).run()


async def _to_browser(upstream, send) -> None:
    async for message in upstream:
        if isinstance(message, bytes):
            await send({"type": "websocket.send", "bytes": message})
        else:
            await send({"type": "websocket.send", "text": message})
    with contextlib.suppress(Exception):
        await send({"type": "websocket.close", "code": 1001})


async def _to_studio(receive, upstream) -> None:
    while True:
        message = await receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("text") is not None:
            await upstream.send(message["text"])
        elif message.get("bytes") is not None:
            await upstream.send(message["bytes"])


async def _respond(
    send, status: int, body, content_type="text/plain; charset=utf-8", headers=()
) -> None:
    if isinstance(body, str):
        body = body.encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type.encode()),
                (b"cache-control", b"no-store"),
                *headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _json(send, status: int, payload: dict) -> None:
    await _respond(send, status, json.dumps(payload), "application/json")


_STYLE = """
    :root { --bg: #1e1f22; --panel: #2b2d31; --line: #3a3c42; --text: #ddd; --muted: #8a8d93;
            --accent: #4c8bf5; --bad: #e06c6c; --ok: #6cc070; --warn: #e0b44c; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px system-ui, sans-serif; }
    main { max-width: 980px; margin: 0 auto; padding: 24px 16px; }
    h1 { font-size: 18px; margin: 0 0 4px; }
    a { color: var(--accent); }
    pre { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 10px;
          white-space: pre-wrap; word-break: break-all; color: #aaa; font-size: 12px; }
"""


def _status_page(heading: str, message: str, log_text: str, refresh: bool) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(heading)}</title>"
        + ("<meta http-equiv='refresh' content='1'>" if refresh else "")
        + f"<style>{_STYLE}</style></head><body><main>"
        f"<h1>{html.escape(heading)}</h1><p>{html.escape(message)} <a href='/'>sessions</a></p>"
        + (f"<pre>{html.escape(log_text)}</pre>" if log_text else "")
        + "</main></body></html>"
    )


_NO_TOKEN = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Miller Brain Studio</title>"
    f"<style>{_STYLE}</style></head><body><main><h1>Miller Brain Studio</h1>"
    "<p>Open the link <code>mbo serve</code> printed; it carries the token this page needs.</p>"
    "</main></body></html>"
)

_LAUNCHER = Template(
    """<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Miller Brain Studio on $host</title>
<style>
"""
    + _STYLE.replace("$", "$$")
    + """
    .sub { color: var(--muted); margin-bottom: 20px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
              padding: 12px 14px; margin-bottom: 16px; }
    h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
         margin: 0 0 10px; font-weight: 600; }
    table { width: 100%; border-collapse: collapse; }
    td { padding: 5px 6px; border-top: 1px solid var(--line); vertical-align: middle; }
    tr:first-child td { border-top: 0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }
    button, select, input { font: inherit; color: var(--text); background: var(--bg);
                            border: 1px solid var(--line); border-radius: 4px; padding: 4px 10px; }
    button { cursor: pointer; }
    button:hover { border-color: var(--muted); }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row input { flex: 1; min-width: 0; }
    .muted { color: var(--muted); }
    .err { color: var(--bad); white-space: pre-wrap; margin-top: 8px; }
    .ready { color: var(--ok); } .starting { color: var(--warn); } .exited { color: var(--muted); }
    .actions { text-align: right; white-space: nowrap; }
    #crumbs { margin-top: 12px; }
    #crumbs a { color: var(--text); text-decoration: none; cursor: pointer; }
    #crumbs a:hover { text-decoration: underline; }
    #list { max-height: 48vh; overflow: auto; margin-top: 6px; border-top: 1px solid var(--line); }
    .entry { padding: 3px 6px; cursor: pointer; display: flex; justify-content: space-between; gap: 12px; }
    .entry:hover { background: #35373c; }
    .entry.sel { background: #2f4a77; }
    .entry .size { color: var(--muted); white-space: nowrap; }
    #recent div { padding: 3px 0; cursor: pointer; }
    #recent div:hover { color: #fff; }
</style>
</head>
<body>
<main>
    <h1>Miller Brain Studio</h1>
    <div class='sub'>on <b>$host</b> &middot; v$version &middot; <span id='limit'></span></div>
    <section>
        <h2>Sessions</h2>
        <table id='sessions'></table>
        <div id='nosessions' class='muted'>No sessions. Open a file or folder below.</div>
    </section>
    <section>
        <h2>Open</h2>
        <div class='row'>
            <input id='path' class='mono' spellcheck='false' placeholder='a file or folder on $host'>
            <select id='widget' title='side widget'>
                <option value='preview'>preview</option>
                <option value='manualroi'>manual ROIs</option>
                <option value='none'>no widget</option>
            </select>
            <button class='primary' id='open'>open</button>
        </div>
        <div id='error' class='err'></div>
        <div id='crumbs' class='mono'></div>
        <div id='list' class='mono'></div>
        <div id='truncated' class='muted'></div>
    </section>
    <section>
        <h2>Recent</h2>
        <div id='recent' class='mono'></div>
    </section>
</main>
<script>
const el = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

async function api(path, options) {
    const r = await fetch(path, options);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
}

function age(started) {
    const s = Math.max(0, Date.now() / 1000 - started);
    if (s < 60) return Math.round(s) + ' s';
    if (s < 3600) return Math.round(s / 60) + ' min';
    return (s / 3600).toFixed(1) + ' h';
}

function sizeText(n) {
    if (n === null) return '';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return (i ? n.toFixed(1) : n) + ' ' + units[i];
}

async function refresh() {
    let state;
    try { state = await api('api/sessions'); } catch (e) { return; }
    const live = state.sessions.filter((s) => s.state !== 'exited').length;
    el('limit').textContent = live + ' of ' + state.max_sessions + ' session' + (state.max_sessions === 1 ? '' : 's');
    el('nosessions').style.display = state.sessions.length ? 'none' : '';
    el('sessions').innerHTML = state.sessions.map((s) => {
        const tabs = s.state === 'ready' ? ' &middot; ' + s.clients + ' tab' + (s.clients === 1 ? '' : 's') : '';
        const open = s.state === 'exited' ? '' : "<button class='primary' onclick=\\"location.href='" + s.url + "'\\">open</button> ";
        const stop = "<button onclick=\\"stopSession('" + s.id + "')\\">" + (s.state === 'exited' ? 'dismiss' : 'stop') + '</button> ';
        const err = s.error ? "<tr><td colspan='4' class='muted'>" + esc(s.error) + '</td></tr>' : '';
        return "<tr><td class='mono' title='" + esc(s.path) + "'>" + esc(s.name) + '</td>'
            + "<td class='" + s.state + "'>" + s.state + tabs + '</td>'
            + "<td class='muted'>" + age(s.started) + '</td>'
            + "<td class='actions'>" + open + stop + "<a href='api/sessions/" + s.id + "/log' target='_blank'>log</a></td></tr>" + err;
    }).join('');
}

async function stopSession(id) {
    await api('api/sessions/' + id + '/stop', {method: 'POST'});
    refresh();
}

async function openPath(path) {
    el('error').textContent = '';
    if (!path) { el('error').textContent = 'type or pick a path first'; return; }
    try {
        const s = await api('api/sessions', {
            method: 'POST',
            headers: {'content-type': 'application/json'},
            body: JSON.stringify({path: path, widget: el('widget').value}),
        });
        location.href = s.url;
    } catch (e) {
        el('error').textContent = e.message;
        refresh();
    }
}

async function browse(path) {
    let d;
    try { d = await api('api/browse?path=' + encodeURIComponent(path || '')); }
    catch (e) { el('error').textContent = e.message; return; }
    el('error').textContent = '';
    el('path').value = d.path;
    el('crumbs').innerHTML = d.crumbs.map((c) => "<a data-path='" + esc(c.path) + "'>" + esc(c.name) + '</a>').join(' / ');
    const rows = [];
    if (d.parent) rows.push({name: '..', path: d.parent, dir: true, size: null});
    rows.push(...d.entries);
    el('list').innerHTML = rows.map((e) =>
        "<div class='entry' data-path='" + esc(e.path) + "' data-dir='" + (e.dir ? 1 : 0) + "'>"
        + '<span>' + esc(e.name) + (e.dir ? '/' : '') + "</span><span class='size'>" + sizeText(e.size) + '</span></div>'
    ).join('');
    el('truncated').textContent = d.truncated ? 'more entries than shown; type the path' : '';
}

el('crumbs').addEventListener('click', (ev) => {
    const a = ev.target.closest('a');
    if (a) browse(a.dataset.path);
});
el('list').addEventListener('click', (ev) => {
    const row = ev.target.closest('.entry');
    if (!row) return;
    if (row.dataset.dir === '1') { browse(row.dataset.path); return; }
    document.querySelectorAll('.entry.sel').forEach((r) => r.classList.remove('sel'));
    row.classList.add('sel');
    el('path').value = row.dataset.path;
});
el('list').addEventListener('dblclick', (ev) => {
    const row = ev.target.closest('.entry');
    if (row && row.dataset.dir === '0') openPath(row.dataset.path);
});
el('open').addEventListener('click', () => openPath(el('path').value.trim()));
el('path').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); openPath(el('path').value.trim()); }
});

api('api/recent').then((d) => {
    el('recent').innerHTML = d.recent.length
        ? d.recent.map((r) => "<div data-path='" + esc(r.path) + "' title='open'>" + esc(r.path) + '</div>').join('')
        : "<span class='muted'>nothing yet</span>";
    el('recent').addEventListener('click', (ev) => {
        const row = ev.target.closest('div[data-path]');
        if (row) browse(row.dataset.path).then(() => { el('path').value = row.dataset.path; });
    });
});

refresh();
setInterval(refresh, 2000);
browse('');
</script>
</body>
</html>
"""
)
