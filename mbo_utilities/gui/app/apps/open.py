"""Opening a file or a folder into the host."""

from __future__ import annotations

from pathlib import Path

from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import imread, log
from mbo_utilities.gui._files import PathPrompt, draw_path_prompt
from mbo_utilities.gui.app._app import App
from mbo_utilities.preferences import add_recent_file, get_last_dir, set_last_dir

logger = log.get("gui.app")

HINTS = {
    "file": "one image file, read by this process",
    "folder": "a folder of image files, read by this process",
}


class OpenApp(App):
    """A typed path, with the native dialog as a browse shortcut, that opens lazily.

    ``kind`` is ``"file"`` or ``"folder"``, one app each. The path is opened
    with ``imread`` and handed to ``AppHost.set_data``; a path that does not
    exist or does not open keeps the prompt up with the reason under it.
    ``open`` is the prompt's own flag, so the menu, the shortcut and the
    prompt's close button are one switch.
    """

    window = True
    owns_window = True
    menu = "File"
    order = 5

    def __init__(self, kind: str = "file"):
        self.kind = kind
        self.id = f"open_{kind}"
        self.title = f"Open {kind}"
        self.shortcut = "o" if kind == "file" else "Shift+O"
        self.order = 5 if kind == "file" else 6
        self.prompt = PathPrompt(
            self.title,
            path=str(get_last_dir(f"open_{kind}") or ""),
            action="open",
            hint=HINTS[kind],
        )
        # a native dialog while one is up; polled every frame
        self.dialog = None
        super().__init__()

    @property
    def open(self) -> bool:
        return self.prompt.open

    @open.setter
    def open(self, value: bool) -> None:
        if value and not self.prompt.open:
            self.prompt.start()
        self.prompt.open = bool(value)

    def draw_window(self, host) -> None:
        if self.dialog is not None and self.dialog.ready():
            result = self.dialog.result()
            self.dialog = None
            if self.kind == "file" and result:
                self.load(host, result[0])
            elif self.kind == "folder" and result:
                self.load(host, result)
        path, browse = draw_path_prompt(self.prompt)
        if browse and self.dialog is None:
            start = Path(self.prompt.path).expanduser()
            start = str(start if start.is_dir() else start.parent)
            if self.kind == "folder":
                self.dialog = pfd.select_folder("Select data folder", start)
            else:
                self.dialog = pfd.open_file(
                    "Select data file", start, ["All Files", "*"]
                )
        if path:
            self.load(host, path)

    def load(self, host, path: str) -> None:
        """Open ``path`` into the host, or say under the prompt why not."""
        self.prompt.path = path
        target = Path(path).expanduser()
        if not target.exists():
            self.prompt.status = f"not found on this machine: {target}"
            self.prompt.open = True
            return
        if self.kind == "folder" and not target.is_dir():
            self.prompt.status = f"not a folder: {target}"
            self.prompt.open = True
            return
        try:
            array = imread(target)
        except Exception as error:
            logger.exception(f"could not open {target}")
            self.prompt.status = f"{type(error).__name__}: {error}"
            self.prompt.open = True
            return
        add_recent_file(str(target), file_type=self.kind)
        set_last_dir(f"open_{self.kind}", str(target))
        self.prompt.open = False
        host.set_data(array)
