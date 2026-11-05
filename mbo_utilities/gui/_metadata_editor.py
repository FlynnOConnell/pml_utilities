"""The Set Metadata popup: the host's ``_custom_metadata`` overrides, edited per canonical key."""

from __future__ import annotations

import contextlib
from typing import Any

from imgui_bundle import hello_imgui, imgui

from mbo_utilities.metadata import get_filename_suggestions, parse_filename_metadata

_COL_SET = imgui.ImVec4(0.5, 0.8, 0.5, 1.0)
_COL_UNSET = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
_COL_EMPTY = imgui.ImVec4(0.5, 0.5, 0.5, 1.0)
_COL_DETECTED = imgui.ImVec4(0.4, 0.8, 0.9, 1.0)
_COL_CUSTOM_HDR = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)


def _field_tooltip(field: dict) -> str:
    tooltip = field.get("description", "")
    examples = field.get("examples", [])
    if examples:
        tooltip += f"\n\nExamples: {', '.join(examples[:5])}"
    return tooltip


def _input_tooltip(dtype) -> str:
    tip = "Type a value and click Set to save"
    if dtype is str:
        tip += " (text)"
    elif dtype is float:
        tip += " (number)"
    return tip


def _push_wrap_at_edge() -> None:
    # explicit visible-edge wrap; 0.0 would track the scrollable content
    # edge inside a horizontal-scrollbar child
    imgui.push_text_wrap_pos(
        imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
    )


def _apply_set(parent: Any, current_data: Any, canonical: str, dtype) -> None:
    input_val = parent._meta_inputs.get(canonical, "").strip()
    if not input_val:
        return
    try:
        parsed = dtype(input_val)
    except ValueError:  # text that is not a number stays in the box
        return
    parent._custom_metadata[canonical] = parsed
    if current_data is not None:
        current_data.metadata[canonical] = parsed
    parent._meta_inputs[canonical] = ""


def _apply_clear(parent: Any, current_data: Any, canonical: str) -> None:
    del parent._custom_metadata[canonical]
    if current_data is not None:
        current_data.metadata.pop(canonical, None)


def _apply_custom_add(parent: Any) -> None:
    val = parent._custom_value
    with contextlib.suppress(ValueError):
        val = float(val) if "." in val else int(val)
    parent._custom_metadata[parent._custom_key.strip()] = val
    parent._custom_key = ""
    parent._custom_value = ""


def _build_suggested_fields(parent: Any, current_data: Any) -> list[dict]:
    """The fields the editor offers: the reader's required ones (a ScanImage TIFF's), dz, fs, then the filename suggestions."""
    suggested_fields = []
    if hasattr(current_data, "get_required_metadata"):  # only TiffArray has them
        suggested_fields.extend(current_data.get_required_metadata())
    existing_canonicals = {f["canonical"] for f in suggested_fields}

    z_step_canonicals = ("dz", "z_step_um", "axial_step_um")
    if not any(c in existing_canonicals for c in z_step_canonicals):
        z_step_field = {
            "canonical": "dz",
            "label": "Z Step",
            "unit": "\u03bcm",
            "dtype": float,
            "description": "Distance between Z-planes in micrometers.",
        }
        val = None if current_data is None else current_data.dz
        if val:
            z_step_field["value"] = val
        suggested_fields.append(z_step_field)
        existing_canonicals.add("dz")

    # add frame rate if not already provided (e.g. isoview XML has no fs)
    fs_canonicals = ("fs", "frame_rate", "framerate")
    is_tiled = bool(getattr(current_data, "is_tiled", False))  # isoview only
    if not is_tiled and not any(c in existing_canonicals for c in fs_canonicals):
        fs_field = {
            "canonical": "fs",
            "label": "Frame Rate",
            "unit": "Hz",
            "dtype": float,
            "description": "Volume (or frame) sampling rate in Hz.",
        }
        val = None if current_data is None else current_data.fs
        if val:
            fs_field["value"] = val
        suggested_fields.append(fs_field)
        existing_canonicals.add("fs")

    filename_meta = None
    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
    if fpath:
        filename_meta = parse_filename_metadata(str(fpath))

    user_fields = get_filename_suggestions()
    for canonical, field_def in user_fields.items():
        if canonical in existing_canonicals:
            continue

        field = dict(field_def)
        if filename_meta:
            detected_val = getattr(filename_meta, canonical, None)
            if detected_val:
                field["value"] = detected_val
                field["detected"] = True

        if current_data is not None and canonical in current_data.metadata:
            field["value"] = current_data.metadata[canonical]

        suggested_fields.append(field)
        existing_canonicals.add(canonical)

    return suggested_fields


def draw_metadata_editor_content(parent: Any):
    """Draw the editor's fields into whatever container is open."""
    data = parent.image_widget.data
    current_data = data[0] if data else None
    suggested_fields = _build_suggested_fields(parent, current_data)

    table_flags = (
        imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    )
    col_label = hello_imgui.em_size(7)
    col_value = hello_imgui.em_size(10)
    col_input = hello_imgui.em_size(8)
    col_btn = hello_imgui.em_size(6)
    # narrow container (docked side panel): stacked rows instead of the
    # fixed-width table, which would clip past the right edge
    input_w = hello_imgui.em_size(7.5)

    table_w = col_label + col_value + col_input + col_btn
    if imgui.get_content_region_avail().x < table_w + hello_imgui.em_size(1):
        _draw_editor_narrow(parent, current_data, suggested_fields)
        return

    if suggested_fields:
        if imgui.begin_table("suggested_meta", 4, table_flags):
            imgui.table_setup_column(
                "label", imgui.TableColumnFlags_.width_fixed, col_label
            )
            imgui.table_setup_column(
                "value", imgui.TableColumnFlags_.width_fixed, col_value
            )
            imgui.table_setup_column(
                "input", imgui.TableColumnFlags_.width_fixed, col_input
            )
            imgui.table_setup_column(
                "btn", imgui.TableColumnFlags_.width_fixed, col_btn
            )

            for field in suggested_fields:
                canonical = field["canonical"]
                label = field["label"]
                unit = field.get("unit", "")
                dtype = field.get("dtype", str)
                detected = field.get("detected", False)

                custom_val = parent._custom_metadata.get(canonical)
                source_val = field.get("value")
                value = custom_val if custom_val is not None else source_val
                is_set = value is not None

                imgui.table_next_row()

                imgui.table_next_column()
                imgui.text_colored(_COL_SET if is_set else _COL_UNSET, label)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(_field_tooltip(field))

                imgui.table_next_column()
                if is_set:
                    val_str = f"{value} {unit}".strip()
                    if detected and custom_val is None:
                        imgui.text_colored(_COL_DETECTED, val_str)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Detected from filename")
                    else:
                        imgui.text_colored(_COL_SET, val_str)
                else:
                    imgui.text_colored(_COL_EMPTY, "-")

                imgui.table_next_column()
                imgui.set_next_item_width(input_w)
                flags = (
                    imgui.InputTextFlags_.chars_decimal if dtype in (float, int) else 0
                )
                _, parent._meta_inputs[canonical] = imgui.input_text(
                    f"##{canonical}",
                    parent._meta_inputs.get(canonical, ""),
                    flags=flags,
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(_input_tooltip(dtype))

                imgui.table_next_column()
                if imgui.small_button(f"Set##{canonical}"):
                    _apply_set(parent, current_data, canonical, dtype)
                if custom_val is not None:
                    imgui.same_line()
                    if imgui.small_button(f"X##del_{canonical}"):
                        _apply_clear(parent, current_data, canonical)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Clear this override")

            imgui.end_table()

        imgui.spacing()

    suggested_keys = {f["canonical"] for f in suggested_fields}
    custom_entries = [
        (k, v) for k, v in parent._custom_metadata.items() if k not in suggested_keys
    ]

    imgui.spacing()
    imgui.separator()
    imgui.text_colored(_COL_CUSTOM_HDR, "Custom")
    imgui.dummy(imgui.ImVec2(0, 2))

    if imgui.begin_table("custom_meta", 4, table_flags):
        imgui.table_setup_column(
            "label", imgui.TableColumnFlags_.width_fixed, col_label
        )
        imgui.table_setup_column(
            "value", imgui.TableColumnFlags_.width_fixed, col_value
        )
        imgui.table_setup_column(
            "input", imgui.TableColumnFlags_.width_fixed, col_input
        )
        imgui.table_setup_column("btn", imgui.TableColumnFlags_.width_fixed, col_btn)

        to_remove = None
        for key, value in custom_entries:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_colored(_COL_SET, key)
            imgui.table_next_column()
            imgui.text_colored(_COL_SET, str(value))
            imgui.table_next_column()
            imgui.table_next_column()
            if imgui.small_button(f"X##custom_del_{key}"):
                to_remove = key
            if imgui.is_item_hovered():
                imgui.set_tooltip("Delete this custom entry")
        if to_remove:
            del parent._custom_metadata[to_remove]

        imgui.table_next_row()
        imgui.table_next_column()
        imgui.set_next_item_width(col_label)
        _, parent._custom_key = imgui.input_text("##custom_key", parent._custom_key)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Custom key name")

        imgui.table_next_column()
        imgui.table_next_column()
        imgui.set_next_item_width(input_w)
        _, parent._custom_value = imgui.input_text("##custom_val", parent._custom_value)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Custom value (numbers auto-detected)")

        imgui.table_next_column()
        if imgui.small_button("Set##custom_add") and parent._custom_key.strip():
            _apply_custom_add(parent)

        imgui.end_table()


def _draw_editor_narrow(
    parent: Any, current_data: Any, suggested_fields: list[dict]
) -> None:
    """Stacked per-field layout: label/value line, then input + buttons.

    Fits any panel width — text wraps and the input shrinks to leave room
    for its buttons, so nothing crosses the right edge.
    """
    style = imgui.get_style()
    set_w = imgui.calc_text_size("Set").x + style.frame_padding.x * 2
    x_w = imgui.calc_text_size("X").x + style.frame_padding.x * 2

    for field in suggested_fields:
        canonical = field["canonical"]
        dtype = field.get("dtype", str)
        custom_val = parent._custom_metadata.get(canonical)
        source_val = field.get("value")
        value = custom_val if custom_val is not None else source_val
        is_set = value is not None

        _push_wrap_at_edge()
        imgui.text_colored(_COL_SET if is_set else _COL_UNSET, field["label"])
        if imgui.is_item_hovered():
            imgui.set_tooltip(_field_tooltip(field))
        imgui.same_line()
        if is_set:
            val_str = f"{value} {field.get('unit', '')}".strip()
            if field.get("detected", False) and custom_val is None:
                imgui.text_colored(_COL_DETECTED, val_str)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Detected from filename")
            else:
                imgui.text_colored(_COL_SET, val_str)
        else:
            imgui.text_colored(_COL_EMPTY, "-")
        imgui.pop_text_wrap_pos()

        reserve = set_w + style.item_spacing.x
        if custom_val is not None:
            reserve += x_w + style.item_spacing.x
        imgui.set_next_item_width(-reserve)
        flags = imgui.InputTextFlags_.chars_decimal if dtype in (float, int) else 0
        _, parent._meta_inputs[canonical] = imgui.input_text(
            f"##{canonical}", parent._meta_inputs.get(canonical, ""), flags=flags
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(_input_tooltip(dtype))
        imgui.same_line()
        if imgui.button(f"Set##{canonical}"):
            _apply_set(parent, current_data, canonical, dtype)
        if custom_val is not None:
            imgui.same_line()
            if imgui.button(f"X##del_{canonical}"):
                _apply_clear(parent, current_data, canonical)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Clear this override")
        imgui.spacing()

    imgui.spacing()
    imgui.separator()
    imgui.text_colored(_COL_CUSTOM_HDR, "Custom")
    imgui.dummy(imgui.ImVec2(0, 2))

    suggested_keys = {f["canonical"] for f in suggested_fields}
    custom_entries = [
        (k, v) for k, v in parent._custom_metadata.items() if k not in suggested_keys
    ]
    to_remove = None
    for key, value in custom_entries:
        if imgui.small_button(f"X##custom_del_{key}"):
            to_remove = key
        if imgui.is_item_hovered():
            imgui.set_tooltip("Delete this custom entry")
        imgui.same_line()
        _push_wrap_at_edge()
        imgui.text_colored(_COL_SET, f"{key}: {value}")
        imgui.pop_text_wrap_pos()
    if to_remove:
        del parent._custom_metadata[to_remove]

    imgui.set_next_item_width(-1)
    _, parent._custom_key = imgui.input_text_with_hint(
        "##custom_key", "key", parent._custom_key
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip("Custom key name")
    imgui.set_next_item_width(-(set_w + style.item_spacing.x))
    _, parent._custom_value = imgui.input_text_with_hint(
        "##custom_val", "value", parent._custom_value
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip("Custom value (numbers auto-detected)")
    imgui.same_line()
    if imgui.button("Set##custom_add") and parent._custom_key.strip():
        _apply_custom_add(parent)


def draw_metadata_popup(parent: Any) -> None:
    """The Set Metadata modal, opened once when ``parent._show_metadata_popup`` flips on (File menu, Shift+M)."""
    if parent._show_metadata_popup:
        # default size; top anchor comes from _metadata_sizer.
        parent._metadata_sizer.before_open()
        imgui.open_popup("Set Metadata##MetadataPopup")
        parent._show_metadata_popup = False

    io = imgui.get_io()
    screen_w, screen_h = io.display_size.x, io.display_size.y
    win_w = min(560, screen_w * 0.7)
    win_h = min(620, screen_h * 0.85)
    imgui.set_next_window_size(imgui.ImVec2(win_w, win_h), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(420, 360), imgui.ImVec2(screen_w, screen_h)
    )

    opened, visible = imgui.begin_popup_modal(
        "Set Metadata##MetadataPopup",
        p_open=True,
        flags=imgui.WindowFlags_.no_saved_settings,
    )
    if not opened:
        return
    try:
        # scrollable content area so the close button stays anchored at
        # the bottom no matter how many fields the editor renders.
        if not visible:
            imgui.close_current_popup()
            return
        imgui.dummy(imgui.ImVec2(0, 4))
        avail = imgui.get_content_region_avail()
        content_h = max(0.0, avail.y - 36.0)
        if imgui.begin_child(
            "##MetadataContent",
            imgui.ImVec2(0, content_h),
            imgui.ChildFlags_.borders,
        ):
            draw_metadata_editor_content(parent)
        imgui.end_child()

        imgui.spacing()
        btn_w = 80
        imgui.set_cursor_pos_x((imgui.get_window_width() - btn_w) * 0.5)
        if imgui.button("Close", imgui.ImVec2(btn_w, 0)):
            imgui.close_current_popup()
    finally:
        imgui.end_popup()
