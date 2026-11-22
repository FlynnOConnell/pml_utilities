"""
shared ui components for timepoint, z-plane, and channel selection.

used by the save-as dialog and every pipeline's slicing popup.
"""

from imgui_bundle import hello_imgui, imgui

from mbo_utilities.arrays.features._dim_labels import find_slider_name, slider_roles
from mbo_utilities.arrays.features._slicing import (
    TimeSelection,
    parse_timepoint_selection,
)


def source_timepoints(widget) -> int:
    """Timepoints of the un-binned dataset behind a viewer widget.

    The viewer may already be showing a ``FrameAveragedView`` (possibly binned
    down to a single frame and squeezed), so count the array the binning was
    applied to when there is one.
    """
    source = widget._frame_average_source
    if source is None:
        source = widget.image_widget.data[0]
    return int(source._shape5d()[0])


def draw_frame_average_input(
    label: str,
    value: int,
    *,
    viewer_factor: int = 1,
    max_frames: int | None = None,
    width_em: float = 6.0,
) -> int:
    """The "Frame Average" option every run/save menu shows, next to Fix Phase.

    Draws an integer input for the temporal binning factor applied to the data
    a task reads (``imread(..., frame_average=N)``), clamps it to
    ``[1, max_frames]`` and returns the new value. ``viewer_factor`` is what
    the viewer's own "Apply to dataset" is currently doing, shown beside the
    input when the two differ so the user can see a run will not match the
    screen.
    """
    value = max(1, int(value or 1))
    imgui.set_next_item_width(hello_imgui.em_size(width_em))
    changed, new_value = imgui.input_int(label, value, step=1, step_fast=5)
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(
            "Average this many consecutive frames into one before processing"
            " (temporal binning). 1 = off. The frame rate is divided by the"
            " same factor, so downstream windows stay in seconds."
        )
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
    if changed:
        value = max(1, int(new_value))
        if max_frames is not None and max_frames >= 1:
            value = min(value, int(max_frames))
    viewer_factor = max(1, int(viewer_factor or 1))
    if value != viewer_factor:
        imgui.same_line()
        shown = "raw" if viewer_factor == 1 else f"{viewer_factor}x"
        imgui.text_disabled(f"(viewer: {shown})")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The viewer is showing something else; this value is what"
                " the run will use."
            )
    return value


# friendly fallbacks when a slider name is just the bare axis letter (t/c/z)
# or the axis has no slider for the loaded array.
_DEFAULT_DIM_LABELS = {"t": "Timepoints", "z": "Z-Planes", "c": "Channels"}


def resolve_dim_labels(parent) -> tuple[str, str, str]:
    """Display labels for the T / Z / C slicing rows.

    Maps each non-spatial axis to the name the viewer's slider already
    shows for the loaded array (e.g. isoview "Tile"/"Cam"/"Zplane"),
    falling back to the generic Timepoints/Z-Planes/Channels when the
    slider name is just the bare axis letter or the axis has no slider.
    The slicing only relabels the rows; the indices it returns (and the
    args downstream tasks receive) are unchanged.

    Returns ``(tp_label, z_label, c_label)``.
    """
    iw = parent.image_widget
    # sliders are the array's T, C, Z axes by position; aliases only for a host without dim names
    positional = tuple(getattr(iw, "dim_names", None) or ()) if iw is not None else ()
    roles = {role: name for name, role in slider_roles(positional).items()}
    names = getattr(iw, "_slider_dim_names", None) or () if iw is not None else ()
    out = []
    for canon in ("t", "z", "c"):
        match = roles.get(canon) or find_slider_name(names, canon)
        out.append(match if match and len(match) > 1 else _DEFAULT_DIM_LABELS[canon])
    return tuple(out)


def set_selection(
    parent, attr: str, size: int, text: str | None = None
) -> TimeSelection | None:
    """Store ``text`` (every index when None) as an axis's ``{attr}_selection``
    with its 0-based ``{attr}_parsed``, or the ``{attr}_error`` that kept it
    from selecting anything; returns the parse or None.
    """
    text = f"1:{size}" if text is None else text
    setattr(parent, f"{attr}_selection", text)
    try:
        parsed = parse_timepoint_selection(text, size)
        error = "" if parsed.count else f"selects nothing of 1:{size}"
    except ValueError as e:
        parsed, error = None, str(e)
    parsed = parsed if not error else None
    setattr(parent, f"{attr}_parsed", parsed)
    setattr(parent, f"{attr}_error", error)
    return parsed


def _draw_selection_row(parent, key, attr, size, label, mode, id_suffix) -> None:
    """One row of the selection table: the label, the text (a picker for
    ``select-one``), All, and how many indices it keeps.
    """
    disabled = mode == "all"
    if disabled:
        imgui.begin_disabled()
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text(label)
    imgui.table_next_column()
    imgui.set_next_item_width(hello_imgui.em_size(13))
    error = getattr(parent, f"{attr}_error")
    if error:
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.3, 0.1, 0.1, 1.0))
    if mode == "select-one":
        # a pipeline that reads one index gets a picker, never a range it cannot honour
        parsed = getattr(parent, f"{attr}_parsed")
        changed, picked = imgui.combo(
            f"##{key}{id_suffix}",
            parsed.final_indices[0] if parsed else 0,
            [str(i + 1) for i in range(size)],
        )
        if changed:
            set_selection(parent, attr, size, str(picked + 1))
    else:
        changed, text = imgui.input_text(
            f"##{key}{id_suffix}", getattr(parent, f"{attr}_selection")
        )
        if changed:
            set_selection(parent, attr, size, text)
    if error:
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(error)
    imgui.table_next_column()
    if imgui.small_button(f"All##{key}{id_suffix}"):
        set_selection(parent, attr, size)
    imgui.table_next_column()
    parsed = getattr(parent, f"{attr}_parsed")
    if parsed is None:
        imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "invalid")
    else:
        imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{parsed.count}/{size}")
        if parsed.exclude_str:
            imgui.same_line()
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.6, 0.4, 1.0), f"(-{len(parsed.exclude_indices)})"
            )
    if disabled:
        imgui.end_disabled()


def draw_selection_table(
    parent,
    max_frames: int,
    num_planes: int,
    tp_attr: str = "_saveas_tp",
    z_attr: str = "_saveas_z",
    id_suffix: str = "",
    suffix_attr: str | None = None,
    num_channels: int = 1,
    c_attr: str = "_saveas_c",
    tp_label: str = "Timepoints",
    z_label: str = "Z-Planes",
    c_label: str = "Channels",
    axes: dict[str, str] | None = None,
) -> tuple[TimeSelection | None, TimeSelection | None, TimeSelection | None]:
    """
    Draw the timepoint, z-plane and channel rows, each read the same way.

    Every row takes ``start:stop:step`` (1-based, inclusive, either end
    optional), a single index, and ``,exclude`` after it (``1:100,50:60``).
    Each axis keeps ``{attr}_selection``, ``{attr}_parsed`` and
    ``{attr}_error`` on ``parent`` (``set_selection``).

    Parameters
    ----------
    parent : Any
        Holds the selection state.
    max_frames, num_planes, num_channels : int
        Size of T, Z and C.
    tp_attr, z_attr, c_attr : str
        Attribute prefix of each axis's state (``"_saveas_tp"``, ``"_s2p_z"``).
    id_suffix : str
        Suffix for imgui IDs to avoid conflicts.
    suffix_attr : str, optional
        Adds a Suffix row editing this attribute (the output filename suffix).
    tp_label, z_label, c_label : str
        Row labels; callers pass the array's slider names (:func:`resolve_dim_labels`).
    axes : dict of str to str, optional
        ``PipelineWidget.axes_consumed`` by ``"T"``/``"Z"``/``"C"``: ``range``
        (default) edits, ``all`` pins the full axis, ``none`` hides the row,
        ``select-one`` draws a picker.

    Returns
    -------
    tuple
        The parsed T, Z and C selections; None for a row that does not parse.
    """
    axes = axes or {}
    rows = (
        ("t", tp_attr, max_frames, tp_label),
        ("z", z_attr, num_planes, z_label),
        ("c", c_attr, num_channels, c_label),
    )
    for key, attr, size, _label in rows:
        text = getattr(parent, f"{attr}_selection", None)
        if axes.get(key.upper()) == "all" and text != f"1:{size}":
            set_selection(parent, attr, size)
        elif getattr(parent, f"{attr}_parsed", None) is None and not getattr(
            parent, f"{attr}_error", ""
        ):
            set_selection(parent, attr, size, text)

    table_flags = (
        imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    )
    if imgui.begin_table(f"selection_table{id_suffix}", 4, table_flags):
        imgui.table_setup_column(
            "dim", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(7)
        )
        imgui.table_setup_column(
            "input", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(14)
        )
        imgui.table_setup_column(
            "all", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(3)
        )
        imgui.table_setup_column("info", imgui.TableColumnFlags_.width_stretch)

        for key, attr, size, label in rows:
            mode = axes.get(key.upper(), "range")
            if mode != "none" and (key == "t" or size > 1):
                _draw_selection_row(parent, key, attr, size, label, mode, id_suffix)

        if suffix_attr:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Suffix")

            imgui.table_next_column()
            imgui.set_next_item_width(hello_imgui.em_size(13))
            suffix_val = getattr(parent, suffix_attr, "")
            changed, new_suffix = imgui.input_text(f"##suffix{id_suffix}", suffix_val)
            if changed:
                setattr(parent, suffix_attr, new_suffix)

            imgui.table_next_column()

            imgui.table_next_column()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 25.0)
                imgui.text_unformatted(
                    "Optional suffix for output filenames.\n"
                    "Examples: 'processed', 'session1'"
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

        imgui.end_table()

    return tuple(getattr(parent, f"{attr}_parsed") for _key, attr, _size, _label in rows)
