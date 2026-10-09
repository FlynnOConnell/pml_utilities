"""File > Open Notebook: the notebooks for the open recording, served by ``imgui_notebooks``.

Two sources follow the open file: the ``scripts`` folder beside it, where ``mbo init`` and this menu
put notebooks, and the shipped templates that apply to it, written there with the path and unit
filled in the first time one is picked. A ``.mesc`` gets the ASAP7 spine pipeline. Folders added
from the panel and the recent list are kept in ``~/.mbo/imgui/notebooks.json``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from imgui_notebooks import (
    CallableSource,
    FolderSource,
    Notebook,
    NotebookMenu,
    NotebookMenuConfig,
    Store,
)
from imgui_notebooks.model import backend_of_name

from mbo_utilities import log
from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.notebook_templates import (
    TEMPLATES,
    notebook_path,
    scripts_dir,
    templates_for,
    write_notebook,
)
from mbo_utilities.preferences import get_mbo_dirs

__all__ = [
    "TEMPLATE_GROUP",
    "StudioNotebookMenu",
    "data_path",
    "draw_notebook_menu",
    "draw_notebooks_window",
    "get_notebook_menu",
    "scripts_notebooks",
    "template_notebooks",
]

logger = log.get("gui.notebooks")

TEMPLATE_GROUP = "new from template"
SCRIPTS_GROUP = "beside the data"

_menu: StudioNotebookMenu | None = None


def data_path(parent: Any) -> Path | None:
    """The open recording: the file, or the first file of a multi-file one."""
    fpath = parent.fpath
    if isinstance(fpath, (list, tuple)):
        fpath = fpath[0] if fpath else None
    return Path(str(fpath)) if fpath else None


def scripts_notebooks(parent: Any) -> list[Notebook]:
    """Every notebook in the ``scripts`` folder beside the open recording, when there is one."""
    data = data_path(parent)
    if data is None or not scripts_dir(data).is_dir():
        return []
    return FolderSource(scripts_dir(data), label=SCRIPTS_GROUP).notebooks()


def template_notebooks(parent: Any) -> list[Notebook]:
    """One entry per shipped template that applies to the open recording, at the path its copy gets."""
    data = data_path(parent)
    if data is None:
        return []
    dest = scripts_dir(data)
    return [
        Notebook(
            str(notebook_path(t, dest)),
            # named from the template: the copy is not written until it is first picked
            backend=backend_of_name(t.filename),
            label=t.title,
            group=TEMPLATE_GROUP,
            root=str(dest),
        )
        for t in templates_for(data)
    ]


class StudioNotebookMenu(NotebookMenu):
    """The menu over the Studio's open recording; a template is written beside it when first picked."""

    def __init__(self, parent: Any, config: NotebookMenuConfig):
        super().__init__(config)
        self.parent = parent

    def open(self, notebook: Notebook):
        if notebook.group == TEMPLATE_GROUP and not Path(notebook.path).exists():
            template = next(t for t in TEMPLATES if notebook.path.endswith(t.filename))
            array = self.parent.image_widget.data[0]
            # peel the squeeze wrapper so isinstance sees the real class
            underlying = getattr(array, "_arr", array)
            unit = underlying.unit_key if isinstance(underlying, MescArray) else ""
            write_notebook(template, Path(notebook.root), data_path(self.parent), unit)
            logger.info(f"wrote {notebook.path} from {template.filename}")
        return super().open(notebook)


def get_notebook_menu(parent: Any) -> StudioNotebookMenu:
    """The one notebook menu for this process, over ``parent``'s open recording."""
    global _menu
    if _menu is None:
        _menu = StudioNotebookMenu(
            parent,
            NotebookMenuConfig(
                title="Notebooks",
                window_id="mbo_notebooks",
                visible=False,
                window_size=(780, 600),
                sources=[
                    CallableSource(
                        partial(scripts_notebooks, parent), label=SCRIPTS_GROUP
                    ),
                    CallableSource(
                        partial(template_notebooks, parent), label=TEMPLATE_GROUP
                    ),
                ],
                store=Store(get_mbo_dirs()["imgui"] / "notebooks.json"),
            ),
        )
    return _menu


def draw_notebook_menu(parent: Any) -> None:
    """Draw the Open Notebook submenu, inside the File menu."""
    get_notebook_menu(parent).draw_menu()


def draw_notebooks_window(parent: Any) -> None:
    """Draw the Notebooks panel and its forms; every frame, open or not."""
    get_notebook_menu(parent).render_window()
