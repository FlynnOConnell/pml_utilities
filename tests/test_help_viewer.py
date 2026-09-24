"""The help viewer's markdown renderer, on a bare imgui context.

imgui has no markdown support and the viewer's renderer is a line loop, so
each block kind is pinned here. Tables are the one block that looks ahead: a
pipe row followed by a rule row opens an imgui table and every following pipe
row is one of its rows, so the shipped pages can carry tables instead of
prose. Anything it does not recognise falls through to wrapped text.
"""

from __future__ import annotations

import pytest
from imgui_bundle import imgui
from mbo_utilities.gui import _help_viewer
from mbo_utilities.gui._help_viewer import DOCS, MESC_DOC, load_doc, render_markdown

TABLE = """# Title

| column | means |
|---|---|
| `fs` | timepoints per second |
| RTMC | on or off |

after the table
"""


class Capture:
    """The tables the renderer opens and the text of each of their cells.

    ``begin_table`` answers True without a real table, so every call that
    would assert outside one is stubbed too; text outside a table is dropped.
    """

    def __init__(self):
        self.tables: list[str] = []
        self.rows: list[list[str]] = []
        self.inside = False

    def begin_table(self, label, *args, **kwargs):
        self.tables.append(label)
        self.inside = True
        return True

    def end_table(self, *args, **kwargs):
        self.inside = False

    def next_row(self, *args, **kwargs):
        self.rows.append([])

    def next_column(self, *args, **kwargs):
        return True

    def inline(self, text, *args, **kwargs):
        if self.inside and self.rows:
            self.rows[-1].append(text)

    def text(self, row):
        return "".join(self.rows[row])


@pytest.fixture
def capture(monkeypatch):
    cap = Capture()
    monkeypatch.setattr(imgui, "begin_table", cap.begin_table)
    monkeypatch.setattr(imgui, "end_table", cap.end_table)
    monkeypatch.setattr(imgui, "table_next_row", cap.next_row)
    monkeypatch.setattr(imgui, "table_next_column", cap.next_column)
    monkeypatch.setattr(imgui, "table_setup_column", lambda *a, **k: None)
    monkeypatch.setattr(_help_viewer, "_render_inline", cap.inline)
    return cap


def render(content):
    """Draw ``content`` for one frame, the way the popup does."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(900, 700)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(850, 650))
        imgui.begin("host")
        render_markdown(content)
        imgui.end()
        imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_pipe_table_becomes_one_table_with_a_row_per_pipe_line(capture):
    render(TABLE)
    assert len(capture.tables) == 1
    assert len(capture.rows) == 3
    assert capture.text(0) == "columnmeans"
    assert capture.text(1) == "`fs`timepoints per second"
    assert capture.text(2) == "RTMCon or off"


def test_the_rule_row_is_consumed_and_the_prose_after_it_is_not_a_cell(capture):
    render(TABLE)
    assert not any("---" in cell for row in capture.rows for cell in row)
    assert not any("after the table" in cell for row in capture.rows for cell in row)


def test_a_pipe_line_without_a_rule_under_it_is_not_a_table(capture):
    render("a | b\n\n| not a table\n")
    assert capture.tables == []
    assert capture.rows == []


def test_a_ragged_row_fills_the_missing_cells(capture):
    render("| a | b | c |\n|---|---|---|\n| one |\n")
    assert capture.tables == ["##md_table_3"]
    assert capture.text(1) == "one"


@pytest.mark.parametrize("doc", [MESC_DOC, *(name for _label, name in DOCS)])
def test_every_shipped_page_renders(doc):
    content = load_doc(doc)
    assert not content.startswith("*Document not found")
    render(content)


def test_the_mesc_page_is_mostly_tables():
    lines = [ln.strip() for ln in load_doc(MESC_DOC).splitlines() if ln.strip()]
    assert sum(ln.startswith("|") for ln in lines) > len(lines) / 2
