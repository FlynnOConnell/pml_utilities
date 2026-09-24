"""
embedded help viewer with markdown rendering.

renders markdown docs shipped with the package in imgui popups.
uses a simple custom renderer since imgui_md requires font setup
that only happens when using immapp.run() (not fastplotlib).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from imgui_bundle import ImVec2, imgui

from mbo_utilities.gui._imgui_helpers import PopupAutoSize

# cached doc content
_doc_cache: dict[str, str] = {}

# available docs: (display_name, filename)
DOCS = [
    ("Quick Start", "gui_quickstart.md"),
    ("File Formats", "file_formats.md"),
]

# not a file: the ROI tool writes its own guide, so it stays next to the
# code it documents instead of drifting in a shipped markdown file
ROI_DOC = "::roi::"
# what a .mesc holds; the MESc tab's (?) opens it
MESC_DOC = "mesc.md"


def docs_for(parent: Any) -> list[tuple[str, str]]:
    """The doc tabs to show: the shipped ones, the MESc page while a
    ``.mesc`` unit is on screen, plus the ROI tool's guide as a section when
    that widget is on. One Help button for the whole app.
    """
    docs = list(DOCS)
    data = getattr(getattr(parent, "image_widget", None), "data", None) or []
    md = getattr(data[0], "metadata", None) if len(data) else None
    if isinstance(md, dict) and "mesc_unit" in md:
        docs.append(("MESc files", MESC_DOC))
    if getattr(parent, "manual_roi", None) is not None:
        docs.append(("ROI Labeling", ROI_DOC))
    return docs


def get_docs_dir() -> Path:
    """Get path to embedded docs directory."""
    return Path(__file__).parent.parent / "assets" / "docs"


def load_doc(filename: str) -> str:
    """Load markdown doc from assets, with caching."""
    if filename == ROI_DOC:
        from mbo_utilities.gui.manual_roi import help_markdown

        return help_markdown()
    if filename not in _doc_cache:
        doc_path = get_docs_dir() / filename
        if doc_path.exists():
            _doc_cache[filename] = doc_path.read_text(encoding="utf-8")
        else:
            _doc_cache[filename] = f"*Document not found: {filename}*"
    return _doc_cache[filename]


def draw_help_popup(parent: Any) -> None:
    """Draw help viewer popup with markdown rendering."""
    if not hasattr(parent, "_show_help_popup"):
        parent._show_help_popup = False
    if not hasattr(parent, "_help_selected_doc"):
        parent._help_selected_doc = 0

    if not hasattr(parent, "_help_sizer"):
        parent._help_sizer = PopupAutoSize("Help##HelpViewer", auto_resize=False)

    if parent._show_help_popup:
        parent._help_sizer.before_open()
        imgui.open_popup("Help##HelpViewer")
        parent._show_help_popup = False

    # default size on first use; top anchor comes from _help_sizer.
    io = imgui.get_io()
    screen_w, screen_h = io.display_size.x, io.display_size.y
    win_w, win_h = min(650, screen_w * 0.7), min(550, screen_h * 0.7)
    imgui.set_next_window_size(ImVec2(win_w, win_h), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size_constraints(ImVec2(400, 300), ImVec2(1200, 900))

    opened, visible = imgui.begin_popup_modal(
        "Help##HelpViewer",
        p_open=True,
        flags=imgui.WindowFlags_.none,
    )

    if opened:
        if not visible:
            imgui.close_current_popup()
        else:
            # doc selector tabs
            docs = docs_for(parent)
            # a widget's (?) asks for its page by file name, for one frame
            wanted = getattr(parent, "_help_select_doc", None)
            if imgui.begin_tab_bar("##HelpTabs"):
                for i, (name, filename) in enumerate(docs):
                    flags = (
                        imgui.TabItemFlags_.set_selected
                        if filename == wanted
                        else imgui.TabItemFlags_.none
                    )
                    if imgui.begin_tab_item(name, None, flags)[0]:
                        parent._help_selected_doc = i
                        imgui.end_tab_item()
                imgui.end_tab_bar()
            parent._help_select_doc = None

            imgui.separator()
            imgui.spacing()

            # content area
            avail = imgui.get_content_region_avail()
            content_height = avail.y - 35  # space for close button

            if imgui.begin_child(
                "##HelpContent", ImVec2(0, content_height), imgui.ChildFlags_.borders
            ):
                # the ROI tab comes and goes with its widget, so the
                # remembered index can outlive the tab it pointed at
                _, filename = docs[min(parent._help_selected_doc, len(docs) - 1)]
                content = load_doc(filename)
                render_markdown(content)
                imgui.end_child()

            # close button
            imgui.separator()
            imgui.spacing()
            btn_width = 80
            imgui.set_cursor_pos_x((imgui.get_window_width() - btn_width) * 0.5)
            if imgui.button("Close", ImVec2(btn_width, 0)):
                imgui.close_current_popup()

        imgui.end_popup()


# colors for the renderer
_C_NORMAL = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_C_BOLD = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)  # brighter = "bold" surrogate
_C_CODE_INLINE = imgui.ImVec4(0.95, 0.75, 0.55, 1.0)
_C_CODE_BLOCK = imgui.ImVec4(0.7, 0.9, 0.7, 1.0)
_C_H1 = imgui.ImVec4(1.0, 0.9, 0.4, 1.0)
_C_H2 = imgui.ImVec4(0.6, 0.85, 1.0, 1.0)
_C_H3 = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_C_BULLET_TERM = imgui.ImVec4(0.9, 0.9, 0.5, 1.0)

# slack _render_inline keeps before it wraps, and a table column must grant
# on top of the text it measures
_WRAP_SLACK = 4.0

# Tokenizer for inline markdown spans:
#   **bold** | `code` | plain
_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def _render_inline(text: str, base_color: imgui.ImVec4 = _C_NORMAL) -> None:
    """Render a single line of text with inline **bold** and `code` spans,
    wrapping at the right edge of the content region.

    imgui has no native rich text and the help viewer doesn't load a
    bold-weight font, so 'bold' is rendered with a brighter tint than the
    base color. Inline `code` uses an orange-ish accent.

    Wrapping is manual because `same_line(0, 0)` (used to compose spans
    inline) blocks imgui's normal text-wrap. We tokenize each span into
    words + whitespace, measure cumulative width with `calc_text_size`,
    and emit a virtual newline (no `same_line`) when the next word would
    overflow the available region.
    """
    # 1. split into colored spans
    spans = []  # list of (text, color)
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            spans.append((part[2:-2], _C_BOLD))
        elif part.startswith("`") and part.endswith("`"):
            spans.append((part[1:-1], _C_CODE_INLINE))
        else:
            spans.append((part, base_color))

    # 2. tokenize each span into word + whitespace pieces; keep colors
    tokens = []  # list of (text, color, is_space)
    for span_text, color in spans:
        # match runs of whitespace OR runs of non-whitespace
        for m in re.finditer(r"\s+|\S+", span_text):
            piece = m.group(0)
            tokens.append((piece, color, piece.isspace()))

    if not tokens:
        return

    # 3. emit tokens with manual wrapping
    avail = imgui.get_content_region_avail().x
    start_x = imgui.get_cursor_pos_x()
    cur_x = start_x
    first_on_line = True
    space_pad = _WRAP_SLACK

    for piece, color, is_space in tokens:
        # leading whitespace at the start of a wrapped line is dropped
        if first_on_line and is_space:
            continue
        w = imgui.calc_text_size(piece).x
        # would this token overflow? if so, break to a new line first.
        # never break before a whitespace (it just gets dropped).
        if not first_on_line and (cur_x + w) > (start_x + avail - space_pad):
            if is_space:
                continue  # don't emit trailing whitespace at end of line
            # force a newline by NOT calling same_line — the next text
            # call lands on the next line at the natural cursor.
            cur_x = start_x
            first_on_line = True

        if not first_on_line:
            imgui.same_line(0, 0)
        imgui.text_colored(color, piece)
        cur_x += w
        first_on_line = False


def render_markdown(content: str) -> None:
    """Render markdown content with basic formatting."""
    in_code_block = False

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()

        # code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            # preserve leading whitespace and wrap long lines.
            imgui.push_style_color(imgui.Col_.text, _C_CODE_BLOCK)
            try:
                imgui.push_text_wrap_pos(0.0)
                try:
                    imgui.text_unformatted(line if line else " ")
                finally:
                    imgui.pop_text_wrap_pos()
            finally:
                imgui.pop_style_color()
            continue

        # empty lines
        if not stripped:
            imgui.spacing()
            continue

        # headers
        if stripped.startswith("# "):
            imgui.spacing()
            imgui.text_colored(_C_H1, stripped[2:])
            imgui.separator()
            continue
        if stripped.startswith("## "):
            imgui.spacing()
            imgui.text_colored(_C_H2, stripped[3:])
            continue
        if stripped.startswith("### "):
            imgui.spacing()
            imgui.text_colored(_C_H3, stripped[4:])
            continue

        # a pipe table: header row, a |---|---| rule, then body rows
        rule = lines[i].strip() if i < len(lines) else ""
        if (
            stripped.startswith("|")
            and rule.startswith("|")
            and "-" in rule
            and set(rule) <= set("|-: ")
        ):
            block = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
            ncol = max(len(r) for r in rows)
            # widths off the unwrapped text: an auto-fit column would measure
            # what _render_inline just wrapped to it and never widen again
            pad = 2.0 * imgui.get_style().cell_padding.x + _WRAP_SLACK
            text = [
                [
                    row[c].replace("**", "").replace("`", "") if c < len(row) else ""
                    for row in rows
                ]
                for c in range(ncol)
            ]
            natural = [
                pad + max([imgui.calc_text_size(t).x for t in col] + [1.0])
                for col in text
            ]
            floor = [
                pad
                + max(
                    [imgui.calc_text_size(w).x for t in col for w in t.split()] + [1.0]
                )
                for col in text
            ]
            # the last column takes the rest and wraps; the others keep their
            # own width, shrunk together when they crowd it out but never
            # under the longest word they hold, which would wrap a label
            head = sum(natural[:-1])
            budget = imgui.get_content_region_avail().x * 0.6
            scale = min(1.0, budget / head) if head else 1.0
            if imgui.begin_table(
                f"##md_table_{i}",
                ncol,
                imgui.TableFlags_.borders_inner | imgui.TableFlags_.row_bg,
            ):
                for c, width in enumerate(natural):
                    if c == ncol - 1:
                        imgui.table_setup_column(
                            "", imgui.TableColumnFlags_.width_stretch
                        )
                    else:
                        imgui.table_setup_column(
                            "",
                            imgui.TableColumnFlags_.width_fixed,
                            max(floor[c], width * scale),
                        )
                for r, row in enumerate(rows):
                    imgui.table_next_row()
                    for c in range(ncol):
                        imgui.table_next_column()
                        cell = row[c] if c < len(row) else ""
                        if cell:
                            _render_inline(cell, _C_BOLD if r == 0 else _C_NORMAL)
                imgui.end_table()
            imgui.spacing()
            continue

        # bullet points (definition style: - **term**: description)
        if stripped.startswith("- **") and "**:" in stripped:
            parts = stripped[2:].split("**:", 1)
            term = parts[0].replace("**", "")
            desc = parts[1].strip() if len(parts) > 1 else ""
            imgui.bullet()
            imgui.same_line()
            imgui.text_colored(_C_BULLET_TERM, term + ":")
            if desc:
                imgui.same_line()
                _render_inline(desc)
            continue
        if stripped.startswith("- "):
            imgui.bullet()
            imgui.same_line()
            _render_inline(stripped[2:])
            continue

        # regular paragraph — wrap and apply inline spans
        _render_inline(stripped)
