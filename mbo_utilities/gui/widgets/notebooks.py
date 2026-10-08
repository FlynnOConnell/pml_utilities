"""File > Open Notebook: the notebooks for the open recording, served by ``imgui_notebooks``.

Two sources follow the open file: the ``scripts`` folder beside it, where ``mbo init`` and this menu
put notebooks, and the shipped templates that apply to it, written there with the path and unit
filled in the first time one is picked. A ``.mesc`` gets the ASAP7 spine pipeline. Folders added
from the panel and the recent list are kept in ``~/.mbo/imgui/notebooks.json``.
"""

from __future__ import annotations

from dataclasses import replace
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

from mbo_utilities import log
from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.lazy_array import base_array
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
    "view_unit",
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


def view_unit(parent: Any) -> str:
    """The ``.mesc`` unit on screen (``MSession_0/MUnit_3``), or ``""`` for any other recording or none."""
    if parent.image_widget is None:
        return ""
    underlying = base_array(parent.image_widget.data[0])
    return underlying.unit_key if isinstance(underlying, MescArray) else ""


def template_notebooks(parent: Any) -> list[Notebook]:
    """One entry per shipped template that applies to the open recording, at the path its copy gets."""
    data = data_path(parent)
    if data is None:
        return []
    dest = scripts_dir(data)
    unit = view_unit(parent)
    return [
        Notebook(
            str(notebook_path(t, dest, unit=unit)),
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
            template = next(t for t in TEMPLATES if t.title == notebook.label)
            out = write_notebook(
                template,
                Path(notebook.root),
                data_path(self.parent),
                view_unit(self.parent),
            )
            if out is not None:
                logger.info(f"wrote {out} from {template.filename}")
                notebook = replace(notebook, path=str(out))
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
