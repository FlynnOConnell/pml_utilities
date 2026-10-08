"""The starter notebooks shipped in ``demos/``: which apply to a recording, and writing one beside it.

``mbo init`` and the Studio's File > Open Notebook both go through here, so a notebook lands in the
same place with the same path filled in whichever way it was asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Template:
    """One shipped notebook.

    ``path_token`` is the data path written in the template, replaced by the real one on write, and
    ``unit_token`` the ``.mesc`` measurement unit likewise. ``suffix`` limits a template to recordings
    of one file type; empty means any.
    """

    filename: str
    title: str
    path_token: str
    suffix: str = ""
    unit_token: str = ""


TEMPLATES = (
    Template("mbo_user_guide.ipynb", "mbo user guide", "D:/demo/raw"),
    Template("lsp_user_guide.ipynb", "LBM-Suite2p user guide", "D:/demo/raw"),
    Template(
        "asap7_spine_pipeline.ipynb",
        "ASAP7 spine pipeline (masknmf)",
        "D:/demo/scan.mesc",
        ".mesc",
        "MSession_0/MUnit_0",
    ),
)


def template_dir() -> Path:
    """Where the shipped notebooks are: ``demos/`` in a checkout, else the package copy a build made."""
    pkg_dir = Path(__file__).resolve().parent
    demos = pkg_dir.parent / "demos"
    return demos if demos.is_dir() else pkg_dir / "assets" / "notebooks"


def templates_for(data_path: str | Path | None) -> list[Template]:
    """The templates for ``data_path``: those for its suffix, or only the generic ones without data."""
    suffix = Path(data_path).suffix.lower() if data_path is not None else ""
    return [t for t in TEMPLATES if not t.suffix or t.suffix == suffix]


def scripts_dir(data_path: str | Path) -> Path:
    """``<data>/../scripts``: where a recording's notebooks go, beside the file or the raw folder."""
    return Path(data_path).expanduser().resolve().parent / "scripts"


def notebook_path(
    template: Template, dest: Path, day: date | None = None, unit: str = ""
) -> Path:
    """Where ``template``'s copy goes in ``dest``: dated, so a later run does not overwrite an edited one,
    and named after ``unit`` for a template that takes one, so each unit of a ``.mesc`` gets its own.
    """
    name = Path(template.filename)
    if unit and template.unit_token:
        name = name.with_stem(f"{name.stem}_{unit.replace('/', '_')}")
    return dest / f"{(day or date.today()).isoformat()}_{name}"


def write_notebook(
    template: Template,
    dest: Path,
    data_path: str | Path | None = None,
    unit: str = "",
    overwrite: bool = False,
) -> Path | None:
    """Copy ``template`` into ``dest`` with the data path and unit filled in.

    Returns the file written, or None when today's copy is already there and ``overwrite`` is off.
    """
    out = notebook_path(template, dest, unit=unit)
    if out.exists() and not overwrite:
        return None
    text = (template_dir() / template.filename).read_text(encoding="utf-8")
    if data_path is not None:
        text = text.replace(
            template.path_token, Path(data_path).expanduser().resolve().as_posix()
        )
    if unit and template.unit_token:
        text = text.replace(template.unit_token, unit)
    dest.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
