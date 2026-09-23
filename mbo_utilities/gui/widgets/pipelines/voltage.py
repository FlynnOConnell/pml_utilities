"""Voltage pipeline widget: the spatial JEDI pipeline from the Run tab.

Same run experience as Suite2p and MaskNMF: current-dataset block, output
folder, the slicing popup, a settings popup with (?) hints and
modified-orange tinting, the modified table and a green Run button that
spawns the "voltage" worker. What is new is the scans-and-domains block:
which AOD ROI units of the file (line scans, chessboard or ribbon patches)
become scans and which ROIs make each domain. Settings are written for the
archive's frame rate and scaled to the scans'. Results are the run's
results zarr beside the input (``pkl`` writes a PF folder instead), which the
Curate button opens in the curation window (``mbo curate``, its own process).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities.arrays.mesc import ROI_LAYOUTS
from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    button_width,
    checkbox_with_tooltip,
    draw_boxed_label,
    fit_width,
    set_tooltip,
    tooltip_marks_right,
)
from mbo_utilities.gui._selection_ui import draw_selection_table, resolve_dim_labels
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui.widgets.pipelines.settings import (
    _MISSING_COLOR,
    _MODIFIED_COLOR,
    _SUBSECTION_COLOR,
    _dataset_size_bytes,
    _draw_md_field,
    _format_size,
    _truncate_to_width,
)
from mbo_utilities.install import VNOISER_HINT
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.reader import widget_reader_kwargs
from mbo_utilities.results import (
    EXCLUDED_DOMAINS,
    PROVENANCE_FILE,
    ResultsArray,
    newest_results,
    pipeline_files,
)

_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
_DIM_COLOR = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
_IMPORTANT_FIELDS = {
    "thres_type",
    "soft_levels",
    "thres_bp_sd",
    "thres_amp_sd",
    "sigma_dfof",
}
DOMAINS_FILE = "domains.json"


def parse_roi_text(text: str, n_lines: int) -> list[int]:
    """``"0,1,2"`` or ``"0:2"`` (inclusive) to 0-based line indices; raises on a bad one."""
    rois = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi = part.split(":", 1)
            rois.extend(range(int(lo), int(hi) + 1))
        else:
            rois.append(int(part))
    if not rois:
        raise ValueError("no lines")
    bad = [r for r in rois if not 0 <= r < n_lines]
    if bad:
        raise ValueError(f"line {bad[0]} is outside 0..{n_lines - 1}")
    return sorted(set(rois))


def _field_default(obj, name: str):
    for f in dataclasses.fields(obj):
        if f.name == name:
            return (
                f.default
                if f.default is not dataclasses.MISSING
                else f.default_factory()
            )
    return None


def _is_default(obj, name: str) -> bool:
    cur, default = getattr(obj, name), _field_default(obj, name)
    if isinstance(cur, float) and isinstance(default, (int, float)):
        return math.isclose(cur, float(default), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(cur, (tuple, list)) and isinstance(default, (tuple, list)):
        return len(cur) == len(default) and all(
            math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
            for a, b in zip(cur, default, strict=True)
        )
    return cur == default


class VoltagePipelineWidget(PipelineWidget):
    """AOD ROI .mesc units (line scans, chessboard or ribbon patches) to a voltage results file with vnoiser."""

    name = "Voltage"
    is_available = HAS_VNOISER
    install_command = VNOISER_HINT
    # frames are a window; Z is the unit's ROI index (mesc_z_axis_meaning "roi_index"), so the
    # Z range picks which ROIs are read and cuts the domains down to them; one channel is averaged
    axes_consumed = {"T": "range", "Z": "range", "C": "select-one"}

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """True for any unit of a .mesc that holds an AOD ROI unit (the unit on screen need not be one)
        and for a PF folder whose source unit is reachable.
        """
        arr = base_array(arr)
        if isinstance(arr, ResultsArray):
            return arr.pipeline == "voltage" and arr.source_recording is not None
        if (getattr(arr, "metadata", None) or {}).get("mesc_layout") in ROI_LAYOUTS:
            return True
        filenames = getattr(arr, "filenames", None) or []
        if not filenames or Path(str(filenames[0])).suffix.lower() != ".mesc":
            return False
        from mbo_utilities.arrays.mesc import list_mesc_units

        try:
            return any(
                u.get("kind") in ROI_LAYOUTS for u in list_mesc_units(filenames[0])
            )
        except Exception:
            return False

    def __init__(self, parent: Any):
        super().__init__(parent)
        from mbo_utilities.vnoiser.params import VoltageSettings

        self.settings = VoltageSettings()
        self._outdir = ""
        self._outdir_dialog = None
        self._last_status = ""
        self._status_color = _DIM_COLOR
        self._show_settings_popup = False
        self._settings_sizer: PopupAutoSize | None = None
        self._show_slice_popup = False
        self._last_fpath = None
        self._units: list[dict] = []
        self._scans: dict[str, bool] = {}
        self._first_env: dict[str, bool] = {}
        self._domain_rows: list[list[str]] = []
        self._domain_error = ""
        self._domains_path = ""

    def _array(self):
        iw = getattr(self.parent, "image_widget", None)
        if iw is None or not iw.data:
            return None
        return base_array(iw.data[0])

    def _dims(self) -> tuple[int, int, int]:
        """(frames, ROIs, channels) of the scans to process: the shortest ticked unit, else the first ROI unit."""
        ticked = [u for u in self._units if self._scans.get(u["key"])] or self._units[
            :1
        ]
        if not ticked:
            return 1000, 1, 1
        unit = min(ticked, key=lambda u: int(u["nframes"]))
        return (
            int(unit["nframes"]),
            int(unit["nrois"]) or 1,
            int(unit.get("nchannels") or 1),
        )

    def _mesc_path(self) -> Path | None:
        arr = self._array()
        if isinstance(arr, ResultsArray):
            return arr.source_recording
        fpath = getattr(self.parent, "fpath", None)
        if isinstance(fpath, (list, tuple)):
            fpath = fpath[0] if fpath else None
        if not fpath or Path(str(fpath)).suffix.lower() != ".mesc":
            return None
        return Path(str(fpath))

    def _set_status(self, text: str, error: bool = False) -> None:
        self._last_status = text
        self._status_color = _MISSING_COLOR if error else _DIM_COLOR

    def _ensure_state(self) -> None:
        """Seed units, scans, domains, output folder and slicing when the dataset changes."""
        from mbo_utilities.arrays.mesc import list_mesc_units
        from mbo_utilities.vnoiser import voltage_run_for_mesc
        from mbo_utilities.vnoiser.params import VoltageSettings
        from mbo_utilities.vnoiser.pipeline import read_domains

        fpath = getattr(self.parent, "fpath", None)
        if fpath == self._last_fpath:
            return
        self._last_fpath = fpath
        self._units, self._scans, self._first_env, self._domain_rows = [], {}, {}, []
        self._domain_error, self._domains_path = "", ""
        mesc = self._mesc_path()
        if mesc is None:
            return
        shown = getattr(self._array(), "unit_key", None)
        try:
            self._units = [
                u for u in list_mesc_units(mesc) if u.get("kind") in ROI_LAYOUTS
            ]
        except Exception as e:
            self._set_status(f"Cannot list the file's units: {e}", error=True)
            return
        # the unit on screen when it has ROIs, else every ROI unit in the file
        on_screen = any(u["key"] == shown for u in self._units)
        for u in self._units:
            self._scans[u["key"]] = u["key"] == shown or not on_screen
        if self._units:
            self._first_env[self._units[0]["key"]] = True
        max_frames, n_lines, _ = self._dims()
        self._voltage_tp_selection = f"1:{max_frames}"
        self._voltage_tp_parsed = None
        self._voltage_tp_error = ""
        self._voltage_z_selection = f"1:{n_lines}"
        self._voltage_z_error = ""
        self._voltage_c_selection = "1"
        self._voltage_c_error = ""
        arr = self._array()
        run = arr.path if isinstance(arr, ResultsArray) else voltage_run_for_mesc(mesc)
        domains, scan_ids, first_env = {}, [], []
        if run is not None:
            prov_file = pipeline_files(run) / PROVENANCE_FILE
            try:
                if prov_file.is_file():
                    prov = json.loads(prov_file.read_text())
                    domains = prov.get("domains") or {}
                    scan_ids = [str(s) for s in prov.get("scan_ids") or []]
                    first_env = [str(s) for s in prov.get("first_env") or []]
                else:
                    from vnoiser import read_pf

                    files = read_pf(run)
                    prov = files.provenance or {}
                    domains, scan_ids = files.domains, files.scan_ids
                    first_env = [str(s) for s in files.rois.get("scanID_1st_env", [])]
                self.settings = VoltageSettings.from_provenance(prov)
                domains = {
                    k: v for k, v in domains.items() if k not in EXCLUDED_DOMAINS
                }
                self._set_status(
                    f"Loaded the previous run's scans and domains from {run.name}"
                )
            except (OSError, ValueError, KeyError) as e:
                self._set_status(
                    f"The previous run at {run.name} could not be read: {e}", error=True
                )
        elif (mesc.parent / DOMAINS_FILE).exists():
            try:
                spec = read_domains(mesc.parent / DOMAINS_FILE)
                domains, scan_ids, first_env = (
                    spec["domains"],
                    spec["scan_ids"],
                    spec["first_env"],
                )
                self._domains_path = str(mesc.parent / DOMAINS_FILE)
            except Exception as e:
                self._domain_error = f"{DOMAINS_FILE}: {e}"
        # after the previous run's settings land: the folder follows the output format
        self._outdir = self._default_outdir()
        if not domains:
            domains = {f"roi{i}": [i] for i in range(n_lines)}
        self._domain_rows = [
            [name, ",".join(str(r) for r in rois)] for name, rois in domains.items()
        ]
        if scan_ids:
            for u in self._units:
                munit = u["key"].rsplit("_", 1)[-1]
                self._scans[u["key"]] = munit in scan_ids
                self._first_env[u["key"]] = munit in first_env

    def _domains(self) -> dict[str, list[int]]:
        """The domain table as ``{name: lines}``; sets ``_domain_error`` and returns {} when invalid."""
        _, n_lines, _ = self._dims()
        out = {}
        for name, text in self._domain_rows:
            name = name.strip()
            if not name:
                self._domain_error = "A domain has no name."
                return {}
            if name in out:
                self._domain_error = f"Domain {name} is listed twice."
                return {}
            try:
                out[name] = parse_roi_text(text, n_lines)
            except ValueError as e:
                self._domain_error = f"{name}: {e}."
                return {}
        if not out:
            self._domain_error = "No domains."
            return {}
        self._domain_error = ""
        return out

    def _frame_window(self) -> tuple[int, int] | None:
        """``(start, stop)`` of the frame selection; None means every frame; raises on a stride."""
        parsed = getattr(self, "_voltage_tp_parsed", None)
        idx = (
            list(getattr(parsed, "final_indices", []) or [])
            if parsed is not None
            else []
        )
        max_frames = self._dims()[0]
        if not idx or len(idx) == max_frames:
            return None
        if np.any(np.diff(idx) != 1):
            raise ValueError(
                "The frame selection must be one contiguous window; the z-score and baselines use the whole trace."
            )
        return int(idx[0]), int(idx[-1]) + 1

    def _planes(self) -> list[int] | None:
        """1-based ROIs of the slice popup's ROI row (the unit's Z axis); None means every ROI; raises on a bad entry."""
        from mbo_utilities.gui._selection_ui import _parse_z_selection

        _, n_lines, _ = self._dims()
        text = str(getattr(self, "_voltage_z_selection", "") or f"1:{n_lines}")
        start, stop, step, error = _parse_z_selection(text, n_lines)
        if error:
            raise ValueError(f"ROIs: {error}")
        planes = list(range(int(start), int(stop) + 1, int(step)))
        return None if planes == list(range(1, n_lines + 1)) else planes

    def _channel(self) -> int:
        text = str(getattr(self, "_voltage_c_selection", "1")).split(":")[0].strip()
        return max(int(text) - 1, 0) if text.isdigit() else 0

    def draw_config(self) -> None:
        self._ensure_state()
        with (
            imgui_ctx.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)),
            fit_width(),
        ):
            imgui.spacing()
            self._draw_dataset_block()
            imgui.separator()
            self._draw_output_row()
            self._draw_slice_row()
            imgui.spacing()
            self._draw_scans_block()
            imgui.spacing()
            self._draw_domains_block()
            imgui.spacing()
            if imgui.button(
                "Pipeline Settings##voltage_settings",
                imgui.ImVec2(hello_imgui.em_size(11), 0),
            ):
                self._show_settings_popup = True
            set_tooltip(
                "dF/F, denoiser and peak parameters. Defaults are the archive's, so an untouched run reproduces its PF folders."
            )
            self._draw_settings_popup()
            imgui.spacing()
            self._draw_modified_table()
            imgui.spacing()
            self._draw_run()

    def _draw_dataset_block(self) -> None:
        imgui.text_colored(_SUBSECTION_COLOR, "Current dataset")
        imgui.spacing()
        arr = self._array()
        if arr is None:
            imgui.text_disabled("(no file loaded)")
            return
        mesc = self._mesc_path()
        path_str = str(mesc) if mesc else "(in-memory)"
        shown = _truncate_to_width(path_str, imgui.get_content_region_avail().x)
        imgui.text_unformatted(shown)
        if shown != path_str and imgui.is_item_hovered():
            imgui.set_tooltip(path_str)
        filenames = getattr(arr, "filenames", None) or ([str(mesc)] if mesc else [])
        if filenames:
            imgui.text(
                f"Size on disk: {_format_size(_dataset_size_bytes(self, filenames))}"
            )
        imgui.text(f"On screen: {getattr(arr, 'unit_key', '?')}")
        set_tooltip(
            "The measurement unit on screen; the Scans block below picks which units become scans."
        )
        imgui.text_disabled(f"{len(self._units)} unit(s) with AOD ROIs in the file")
        ticked = [u for u in self._units if self._scans.get(u["key"])]
        if ticked:
            fs = float(ticked[0]["fs"])
            _draw_md_field("Frame rate", fs, "Hz")
            ref = float(self.settings.runtime.reference_fs or 0)
            if ref > 0 and not math.isclose(fs, ref, rel_tol=1e-6):
                imgui.text_disabled(
                    f"Sample-count settings scale by {fs / ref:.3f} from {ref:.1f} Hz"
                )
                set_tooltip(
                    "dF/F sigmas, start-up samples, wavelet scales and peak spacing are written for the "
                    "reference rate and scaled to this scan's; the peak band-pass is capped below Nyquist."
                )
            comment = str(ticked[0].get("comment") or "")
            if comment:
                imgui.text_disabled(f"Comment: {comment}")

    def _default_outdir(self) -> str:
        """Where this format's output belongs: the folder beside the file for
        a zarr run, which the runner names its file in, else the PF folder.
        """
        from mbo_utilities.vnoiser.pipeline import default_pf_dir

        mesc = self._mesc_path()
        if mesc is None:
            return ""
        if self.settings.runtime.output_format == "zarr":
            return str(mesc.parent)
        arr = self._array()
        if isinstance(arr, ResultsArray) and arr.path.suffix != ".zarr":
            return str(arr.path)
        return str(default_pf_dir(mesc))

    def _draw_output_row(self) -> None:
        if self._outdir_dialog is not None and self._outdir_dialog.ready():
            result = self._outdir_dialog.result()
            if result:
                self._outdir = result
                set_last_dir("voltage_outdir", result)
            self._outdir_dialog = None
        zarr_out = self.settings.runtime.output_format == "zarr"
        imgui.text_colored(
            _SUBSECTION_COLOR, "Output folder" if zarr_out else "Output folder (PF)"
        )
        set_tooltip(
            "The folder the results file goes in; it is named after the input with a timestamp, "
            "so a run never overwrites an earlier one."
            if zarr_out
            else "The PF folder the curation window opens: <animal>/<expt>/PF for the archive layout, else PF beside the file."
        )
        btn_w = hello_imgui.em_size(6)
        imgui.set_next_item_width(
            max(
                imgui.get_content_region_avail().x
                - btn_w
                - imgui.get_style().item_spacing.x,
                hello_imgui.em_size(6),
            )
        )
        _, self._outdir = imgui.input_text("##voltage_outdir", self._outdir)
        if self._outdir and imgui.is_item_hovered():
            imgui.set_tooltip(self._outdir)
        imgui.same_line()
        if imgui.button("Browse##voltage_outdir_btn", imgui.ImVec2(btn_w, 0)):
            start = str(get_last_dir("voltage_outdir") or Path.home())
            self._outdir_dialog = pfd.select_folder("Select output folder", start)

    def _draw_slice_row(self) -> None:
        max_frames, n_lines, num_channels = self._dims()
        if imgui.button(
            "Set slice##voltage_slice", imgui.ImVec2(hello_imgui.em_size(6), 0)
        ):
            self._show_slice_popup = True
        set_tooltip(
            "Frame window, ROIs and channel. The ROI row is this unit's Z axis: only the chosen ROIs "
            "are read and every domain is cut down to them.",
            show_mark=False,
        )
        imgui.same_line()
        try:
            window = self._frame_window()
            planes = self._planes()
            n_frames = max_frames if window is None else window[1] - window[0]
            n_rois = n_lines if planes is None else len(planes)
            imgui.text_colored(
                _DIM_COLOR,
                f"{n_frames} frames · {n_rois}/{n_lines} ROIs · ch {self._channel() + 1}",
            )
        except ValueError as e:
            imgui.text_colored(_MISSING_COLOR, str(e))
        if self._show_slice_popup:
            imgui.open_popup("Frames & Channel##voltage_slice")
            self._show_slice_popup = False
        imgui.set_next_window_size(
            imgui.ImVec2(hello_imgui.em_size(36), 0), imgui.Cond_.first_use_ever
        )
        if imgui.begin_popup("Frames & Channel##voltage_slice"):
            tp_label, _z_label, c_label = resolve_dim_labels(self.parent)
            draw_selection_table(
                self,
                max_frames,
                n_lines,
                tp_attr="_voltage_tp",
                z_attr="_voltage_z",
                id_suffix="_voltage",
                num_channels=num_channels,
                c_attr="_voltage_c",
                tp_label=tp_label,
                z_label="ROIs",
                c_label=c_label,
                axes=self.axes_consumed,
            )
            imgui.spacing()
            if imgui.button(
                "Close##voltage_slice_close", imgui.ImVec2(hello_imgui.em_size(6), 0)
            ):
                imgui.close_current_popup()
            imgui.end_popup()

    def _draw_scans_block(self) -> None:
        imgui.text_colored(_SUBSECTION_COLOR, "Scans")
        set_tooltip(
            "Each ticked unit becomes one scan of the PF folder, keyed by its MUnit number. "
            "env marks the first scan of each environment (scanID_1st_env)."
        )
        if not self._units:
            imgui.text_disabled("No units with AOD ROIs in this file.")
            return
        for u in self._units:
            key = u["key"]
            munit = key.rsplit("/", 1)[-1]
            seconds = u.get("duration_s")
            label = (
                f"{munit} ({u['nrois']} ROIs, {seconds:.0f} s)"
                if seconds
                else f"{munit} ({u['nrois']} ROIs)"
            )
            _, self._scans[key] = imgui.checkbox(
                f"{label}##voltage_scan_{key}", self._scans.get(key, False)
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(key)
            if self._scans[key]:
                imgui.same_line()
                _, self._first_env[key] = imgui.checkbox(
                    f"env##voltage_env_{key}", self._first_env.get(key, False)
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip("First scan of an environment.")

    def _draw_domains_block(self) -> None:
        imgui.text_colored(_SUBSECTION_COLOR, "Domains")
        set_tooltip(
            "Domains are groups of ROIs, letting you average signals from the same "
            "structure (a soma, a branch, a cell).\n\n"
            "By default, each domain has 1 ROI.\n\n"
            "Femtonics comments like 'bas1-3, api1-5' are parsed automatically to name "
            "them. If you expect different parsing than what you see here, file an issue.\n\n"
            "ROIs are 0-based, in drawing order: 0,1,2 or 0:2."
        )
        _, n_lines, _ = self._dims()
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_stretch_prop
        )
        remove = None
        if imgui.begin_table("##voltage_domains", 3, flags):
            imgui.table_setup_column(
                "Domain", imgui.TableColumnFlags_.width_stretch, 2.0
            )
            imgui.table_setup_column("ROIs", imgui.TableColumnFlags_.width_stretch, 3.0)
            imgui.table_setup_column(
                "", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(1.8)
            )
            imgui.table_headers_row()
            for i, row in enumerate(self._domain_rows):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.set_next_item_width(-1)
                _, row[0] = imgui.input_text(f"##voltage_dom_name_{i}", row[0])
                imgui.table_set_column_index(1)
                try:
                    parse_roi_text(row[1], n_lines)
                    bad = False
                except ValueError:
                    bad = True
                if bad:
                    imgui.push_style_color(imgui.Col_.text, _MISSING_COLOR)
                imgui.set_next_item_width(-1)
                _, row[1] = imgui.input_text(f"##voltage_dom_rois_{i}", row[1])
                if bad:
                    imgui.pop_style_color()
                imgui.table_set_column_index(2)
                if imgui.small_button(f"x##voltage_dom_rm_{i}"):
                    remove = i
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Remove this domain.")
            imgui.end_table()
        if remove is not None:
            del self._domain_rows[remove]
        # the three buttons wrap to a second line when the panel is too narrow for one
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x
        if imgui.button("Add##voltage_dom_add"):
            self._domain_rows.append([f"domain{len(self._domain_rows) + 1}", ""])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add a domain row.")
        if imgui.get_item_rect_max().x + spacing + button_width("Load") <= right:
            imgui.same_line()
        if imgui.button("Load##voltage_dom_load"):
            self._load_domains_file()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Load {DOMAINS_FILE} from beside the file.")
        if imgui.get_item_rect_max().x + spacing + button_width("Save") <= right:
            imgui.same_line()
        if imgui.button("Save##voltage_dom_save"):
            self._save_domains_file()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Save the table, scans and env flags to {DOMAINS_FILE} beside the file."
            )
        domains = self._domains()
        if self._domain_error:
            imgui.text_colored(_MISSING_COLOR, self._domain_error)
        else:
            imgui.text_disabled(
                f"{len(domains)} domains, {sum(len(v) for v in domains.values())} ROIs"
            )
        if self._domains_path:
            shown = _truncate_to_width(
                self._domains_path, imgui.get_content_region_avail().x
            )
            imgui.text_disabled(shown)
            if imgui.is_item_hovered():
                imgui.set_tooltip(self._domains_path)

    def _load_domains_file(self) -> None:
        from mbo_utilities.vnoiser.pipeline import read_domains

        mesc = self._mesc_path()
        path = (mesc.parent if mesc else Path.home()) / DOMAINS_FILE
        if not path.exists():
            self._domain_error = f"{path} not found."
            return
        try:
            spec = read_domains(path)
        except Exception as e:
            self._domain_error = f"{path.name}: {e}"
            return
        self._domain_rows = [
            [n, ",".join(str(r) for r in rois)]
            for n, rois in spec["domains"].items()
            if n not in ("All_domains", "bg")
        ]
        for u in self._units:
            munit = u["key"].rsplit("_", 1)[-1]
            if spec["scan_ids"]:
                self._scans[u["key"]] = munit in spec["scan_ids"]
                self._first_env[u["key"]] = munit in spec["first_env"]
        self._domains_path = str(path)
        self._set_status(f"Loaded {path.name}")

    def _save_domains_file(self) -> None:
        domains = self._domains()
        if not domains:
            return
        mesc = self._mesc_path()
        path = (mesc.parent if mesc else Path.home()) / DOMAINS_FILE
        scans = [
            u["key"].rsplit("_", 1)[-1]
            for u in self._units
            if self._scans.get(u["key"])
        ]
        first_env = [
            u["key"].rsplit("_", 1)[-1]
            for u in self._units
            if self._scans.get(u["key"]) and self._first_env.get(u["key"])
        ]
        doc = {
            "mesc": mesc.name if mesc else "",
            "scans": scans,
            "first_env": first_env,
            "domains": domains,
        }
        path.write_text(json.dumps(doc, indent=2))
        self._domains_path = str(path)
        self._set_status(f"Saved {path.name}")

    def _mod_push(self, obj, field: str) -> bool:
        if not _is_default(obj, field):
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
            return True
        return False

    def _row_tail(self, obj, field: str, label: str, tooltip: str) -> None:
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        pushed = self._mod_push(obj, field)
        if field in _IMPORTANT_FIELDS:
            draw_boxed_label(label, font=getattr(self.parent, "_bold_font", None))
        else:
            imgui.text(label)
        if pushed:
            imgui.pop_style_color()
        set_tooltip(tooltip)

    def _f_int(self, obj, field: str, label: str, tooltip: str, lo=0) -> None:
        pushed = self._mod_push(obj, field)
        imgui.set_next_item_width(hello_imgui.em_size(6))
        _, val = imgui.input_int(f"##voltage_{field}", int(getattr(obj, field)))
        if pushed:
            imgui.pop_style_color()
        setattr(obj, field, max(lo, val))
        self._row_tail(obj, field, label, tooltip)

    def _f_float(
        self, obj, field: str, label: str, tooltip: str, step=0.1, fmt="%.2f"
    ) -> None:
        pushed = self._mod_push(obj, field)
        imgui.set_next_item_width(hello_imgui.em_size(6))
        _, val = imgui.input_float(
            f"##voltage_{field}", float(getattr(obj, field)), step, step * 10, fmt
        )
        if pushed:
            imgui.pop_style_color()
        setattr(obj, field, val)
        self._row_tail(obj, field, label, tooltip)

    def _f_check(self, obj, field: str, label: str, tooltip: str) -> None:
        pushed = self._mod_push(obj, field)
        val = checkbox_with_tooltip(
            f"{label}##voltage_{field}", bool(getattr(obj, field)), tooltip
        )
        if pushed:
            imgui.pop_style_color()
        setattr(obj, field, bool(val))

    def _draw_settings_popup(self) -> None:
        popup_title = "Voltage Pipeline Settings##voltage_settings_popup"
        if self._settings_sizer is None:
            self._settings_sizer = PopupAutoSize(popup_title)
        if self._show_settings_popup:
            imgui.open_popup(popup_title)
            self._show_settings_popup = False
        viewport = imgui.get_main_viewport()
        self._settings_sizer.before_open()
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(
                min(hello_imgui.em_size(62), viewport.size.x * 0.9),
                hello_imgui.em_size(12),
            ),
            imgui.ImVec2(viewport.size.x * 0.98, viewport.size.y * 0.98),
        )
        opened, visible = imgui.begin_popup_modal(
            popup_title,
            p_open=True,
            flags=self._settings_sizer.flags(imgui.WindowFlags_.no_saved_settings),
        )
        if not opened:
            return
        try:
            if not visible:
                imgui.close_current_popup()
                return
            with imgui_ctx.push_style_var(
                imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)
            ):
                self._draw_settings_columns()
                imgui.spacing()
                imgui.separator()
                imgui.text_colored(_SUBSECTION_COLOR, "Runtime")
                from mbo_utilities.vnoiser.params import OUTPUT_FORMATS

                rt = self.settings.runtime
                self._f_check(
                    rt,
                    "convert",
                    "Apply the file's conversion offset",
                    "Zero then means no photons. Off reproduces the archive, which worked on raw counts.",
                )
                self._f_check(
                    rt,
                    "save_cwt",
                    "Save cwts.h5",
                    "The wavelet coefficients, about 20 bytes per sample per domain.",
                )
                self._f_check(
                    rt,
                    "overwrite",
                    "Overwrite PF",
                    "Replace an existing PF folder's pipeline files; .curation stays.",
                )
                pushed = self._mod_push(rt, "output_format")
                imgui.set_next_item_width(hello_imgui.em_size(6))
                current = (
                    OUTPUT_FORMATS.index(rt.output_format)
                    if rt.output_format in OUTPUT_FORMATS
                    else 0
                )
                changed, chosen = imgui.combo(
                    "##voltage_output_format", current, list(OUTPUT_FORMATS)
                )
                if pushed:
                    imgui.pop_style_color()
                if changed:
                    was = self._default_outdir()
                    rt.output_format = OUTPUT_FORMATS[chosen]
                    # an untouched folder follows the format; a chosen one stays
                    if self._outdir in ("", was):
                        self._outdir = self._default_outdir()
                self._row_tail(
                    rt,
                    "output_format",
                    "Output format",
                    "zarr: one <input>.<timestamp>.voltage.zarr results file beside the input (the shape "
                    "every pipeline's results share), the run's own files in a voltage/ folder inside it. "
                    "pkl: the archive's PF folder of pickles.",
                )
                imgui.spacing()
                imgui.separator()
                btn_w = hello_imgui.em_size(6)
                imgui.push_style_color(
                    imgui.Col_.button, imgui.ImVec4(0.60, 0.35, 0.10, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_hovered, imgui.ImVec4(0.70, 0.42, 0.14, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_active, imgui.ImVec4(0.50, 0.28, 0.08, 1.0)
                )
                if imgui.button("Defaults##voltage_defaults", imgui.ImVec2(btn_w, 0)):
                    from mbo_utilities.vnoiser.params import VoltageSettings

                    self.settings = VoltageSettings()
                imgui.pop_style_color(3)
                set_tooltip(
                    "Reset every parameter to the archive's settings (Noguchi & Terada).",
                    show_mark=False,
                )
                imgui.same_line()
                imgui.set_cursor_pos_x(
                    imgui.get_window_width()
                    - btn_w
                    - imgui.get_style().window_padding.x
                )
                imgui.push_style_color(
                    imgui.Col_.button, imgui.ImVec4(0.55, 0.13, 0.13, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_hovered, imgui.ImVec4(0.65, 0.18, 0.18, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.button_active, imgui.ImVec4(0.45, 0.10, 0.10, 1.0)
                )
                if imgui.button(
                    "Close##voltage_settings_close", imgui.ImVec2(btn_w, 0)
                ):
                    imgui.close_current_popup()
                imgui.pop_style_color(3)
        finally:
            imgui.end_popup()

    def _draw_settings_columns(self) -> None:
        columns = (
            ("dF/F and z-score", "dfof", self._draw_dfof_params),
            ("Wavelet denoiser", "denoiser", self._draw_denoiser_params),
            ("Peaks", "events", self._draw_event_params),
        )
        avail_w = imgui.get_content_region_avail().x
        col_w = max(
            (avail_w - 2 * imgui.get_style().item_spacing.x) / 3,
            hello_imgui.em_size(18),
        )
        for i, (title, key, body) in enumerate(columns):
            if i:
                imgui.same_line()
            imgui.begin_child(
                f"##voltage_col_{key}",
                imgui.ImVec2(col_w, 0),
                imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
            )
            try:
                imgui.text_colored(_TITLE_COLOR, title)
                imgui.separator()
                with tooltip_marks_right():
                    body()
            finally:
                imgui.end_child()

    def _draw_dfof_params(self) -> None:
        d = self.settings.dfof
        self._f_float(
            d,
            "sigma_dfof",
            "dF/F baseline sigma",
            "Gaussian baseline under dF/F, in samples (1500 = 1.4 s at 1075 Hz).",
            step=100,
            fmt="%.0f",
        )
        self._f_float(
            d,
            "sigma_baseline",
            "z baseline sigma",
            "Slower Gaussian baseline removed before the z-score, in samples.",
            step=100,
            fmt="%.0f",
        )
        self._f_int(
            d,
            "n_startup",
            "Start-up samples",
            "Leading samples replaced by the mean of the rest (filter start-up).",
        )
        self._f_check(
            d,
            "negative",
            "Flip sign",
            "JEDI darkens on depolarisation; on makes spikes positive.",
        )

    def _draw_denoiser_params(self) -> None:
        den = self.settings.denoiser
        pushed = self._mod_push(den, "thres_type")
        idx = 0 if den.thres_type == "soft" else 1
        imgui.set_next_item_width(hello_imgui.em_size(6))
        changed, idx = imgui.combo("##voltage_thres_type", idx, ["Soft", "Hard"])
        if pushed:
            imgui.pop_style_color()
        if changed:
            den.thres_type = "soft" if idx == 0 else "hard"
        self._row_tail(
            den,
            "thres_type",
            "Mask type",
            "Soft keeps a fraction of every band outside events; the archive used soft.",
        )
        pushed = self._mod_push(den, "soft_levels")
        imgui.set_next_item_width(hello_imgui.em_size(12))
        _, levels = imgui.input_float4(
            "##voltage_soft_levels", list(den.soft_levels), "%.2f"
        )
        if pushed:
            imgui.pop_style_color()
        den.soft_levels = tuple(float(v) for v in levels)
        self._row_tail(
            den,
            "soft_levels",
            "Soft levels",
            "Attenuation outside events for bands below 5, 5-30, 30-80 and above 80 Hz.",
        )
        self._f_float(
            den,
            "hard_floor",
            "Hard floor",
            "Attenuation outside events in hard mode.",
            step=0.01,
        )
        self._f_check(
            den,
            "complex_bands",
            "Complex bands above 75 Hz",
            "As the original code; off takes the real part first.",
        )
        self._f_int(
            den,
            "n_scales",
            "Scales",
            "Wavelet scales, log-spaced between the two below.",
            lo=8,
        )
        self._f_float(
            den,
            "scale_min",
            "Scale min",
            "Smallest wavelet scale in samples.",
            step=0.5,
            fmt="%.1f",
        )
        self._f_float(
            den,
            "scale_max",
            "Scale max",
            "Largest wavelet scale in samples.",
            step=50,
            fmt="%.0f",
        )
        self._f_float(
            den,
            "lp_cutoff_hz",
            "Baseline cutoff Hz",
            "FIR low-pass of the z-score saved as lp_FIR1Hz; not added to the curated trace.",
            step=0.5,
            fmt="%.1f",
        )
        self._f_float(
            den,
            "fir_window_ms",
            "FIR window ms",
            "Hamming window length of the baseline filter.",
            step=100,
            fmt="%.0f",
        )
        self._f_check(
            den,
            "fir_odd_taps",
            "Odd FIR taps",
            "Trim the tap count to odd. Off reproduces the archive.",
        )
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.collapsing_header("Clustering and windows##voltage_adv"):
            self._f_int(
                den,
                "n_components",
                "PCA components",
                "PCA components of the wavelet rows.",
                lo=2,
            )
            self._f_int(
                den,
                "n_comp_clu",
                "Components clustered",
                "Leading components the clustering uses.",
                lo=1,
            )
            self._f_int(
                den,
                "n_clusters",
                "Clusters",
                "Coarse cut of the frequency tree (unused downstream).",
                lo=1,
            )
            self._f_int(
                den, "n_subclusters", "Bands", "Frequency bands reconstructed.", lo=1
            )
            self._f_float(
                den,
                "slow_upthres",
                "Open level, slow (SD)",
                "Event window opens at this many SD for bands below 30 Hz.",
            )
            self._f_float(
                den,
                "fast_upthres",
                "Open level, fast (SD)",
                "Event window opens at this many SD for bands at or above 30 Hz.",
            )

    def _draw_event_params(self) -> None:
        ev = self.settings.events
        self._f_check(
            ev,
            "detect",
            "Detect peaks",
            "Write detected_events_peaks.pkl; the curation window does not need it.",
        )
        if not ev.detect:
            imgui.begin_disabled()
        try:
            self._f_float(
                ev,
                "bp_low",
                "Band-pass low Hz",
                "Low edge of the Butterworth band-pass.",
                step=1,
                fmt="%.0f",
            )
            self._f_float(
                ev,
                "bp_high",
                "Band-pass high Hz",
                "High edge of the Butterworth band-pass.",
                step=10,
                fmt="%.0f",
            )
            self._f_float(
                ev,
                "thres_bp_sd",
                "Band-passed SD",
                "Peaks of the band-passed trace above mean + this many SD.",
            )
            self._f_float(
                ev,
                "thres_amp_sd",
                "Amplitude SD",
                "Peak height on the trace itself above mean + this many SD.",
            )
            self._f_float(
                ev,
                "duration_thres_ms",
                "Min duration ms",
                "Shortest stretch above the on/off level a peak may sit on.",
                step=1,
                fmt="%.1f",
            )
            self._f_int(
                ev,
                "distance_samples",
                "Min spacing",
                "Samples between peaks; 3 reproduces the archive.",
                lo=1,
            )
        finally:
            if not ev.detect:
                imgui.end_disabled()

    def _draw_modified_table(self) -> None:
        mods = []
        for prefix, obj in (
            ("dfof", self.settings.dfof),
            ("denoiser", self.settings.denoiser),
            ("events", self.settings.events),
            ("runtime", self.settings.runtime),
        ):
            for f in dataclasses.fields(obj):
                if not _is_default(obj, f.name):
                    mods.append(
                        (
                            f"{prefix}.{f.name}",
                            getattr(obj, f.name),
                            _field_default(obj, f.name),
                        )
                    )
        imgui.text(f"Modified parameters ({len(mods)})")
        set_tooltip(
            "Every parameter that differs from the archive's value, with the default beside it."
        )
        if not mods:
            imgui.text_disabled("All parameters at the archive's defaults")
            return
        mod_h = min(
            hello_imgui.em_size(10),
            imgui.get_frame_height_with_spacing() * (len(mods) + 2),
        )
        if imgui.begin_child(
            "##voltage_mod_params", imgui.ImVec2(-1, mod_h), imgui.ChildFlags_.borders
        ):
            flags = (
                imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h
                | imgui.TableFlags_.sizing_stretch_prop
            )
            if imgui.begin_table("##voltage_mod_tbl", 3, flags):
                imgui.table_setup_column(
                    "Parameter", imgui.TableColumnFlags_.width_stretch, 4.0
                )
                imgui.table_setup_column(
                    "Current", imgui.TableColumnFlags_.width_stretch, 2.5
                )
                imgui.table_setup_column(
                    "Default", imgui.TableColumnFlags_.width_stretch, 2.5
                )
                imgui.table_headers_row()
                for field, cur, default in mods:
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    imgui.text_colored(_MODIFIED_COLOR, field)
                    imgui.table_set_column_index(1)
                    imgui.text(f"{cur:.3g}" if isinstance(cur, float) else str(cur))
                    imgui.table_set_column_index(2)
                    imgui.text_disabled(
                        f"{default:.3g}" if isinstance(default, float) else str(default)
                    )
                imgui.end_table()
        imgui.end_child()

    def _draw_run(self) -> None:
        mesc = self._mesc_path()
        scans = [u["key"] for u in self._units if self._scans.get(u["key"])]
        rates = sorted(
            {
                round(float(u["fs"] or 0), 3)
                for u in self._units
                if self._scans.get(u["key"])
            }
        )
        domains = self._domains()
        try:
            window = self._frame_window()
            planes = self._planes()
            window_error = ""
        except ValueError as e:
            window, planes, window_error = None, None, str(e)
        # what the runner will keep: every domain cut down to the chosen ROIs
        chosen = None if planes is None else {p - 1 for p in planes}
        kept = {
            n: [r for r in rs if chosen is None or r in chosen]
            for n, rs in domains.items()
        }
        reason = ""
        if mesc is None:
            reason = "Load a .mesc with AOD ROI units first."
        elif not self._outdir:
            reason = "Set the output folder."
        elif not scans:
            reason = "Tick at least one scan."
        elif len(rates) > 1:
            reason = f"Ticked scans have different frame rates ({', '.join(f'{r:g}' for r in rates)} Hz); run them separately."
        elif not domains:
            reason = self._domain_error
        elif window_error:
            reason = window_error
        elif not any(kept.values()):
            reason = "No domain has an ROI in the ROI selection."
        ready = not reason
        run_w = hello_imgui.em_size(14)
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
        imgui.push_style_color(
            imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0)
        )
        imgui.push_style_color(
            imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0)
        )
        if not ready:
            imgui.begin_disabled()
        run_avail = imgui.get_content_region_avail().x
        if run_avail > run_w:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (run_avail - run_w) * 0.5)
        clicked = imgui.button("Run Voltage", imgui.ImVec2(run_w, 0))
        if not ready:
            imgui.end_disabled()
        imgui.pop_style_color(3)
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                reason
                or f"Spawn a worker for {len(scans)} scan(s); the process console tracks it."
            )
        if self._last_status:
            imgui.text_colored(self._status_color, self._last_status)
        if clicked and ready:
            self._submit(mesc, scans, domains, window, planes)
        pf_done = (
            bool(self._outdir)
            and (Path(self._outdir) / "denoised_trace_scans.pkl").exists()
        )
        results = (
            newest_results(self._outdir, "voltage")
            if self._outdir and not pf_done
            else None
        )
        if results is not None:
            imgui.text_disabled(f"Results: {results.name}")
            if imgui.button(
                "Load into Traces##voltage_load_traces",
                imgui.ImVec2(hello_imgui.em_size(11), 0),
            ):
                self._load_traces(results)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Load every scan's denoised and line traces into the Traces tab (Manual ROI Labeling)."
                )
        if mesc is None:
            return
        # the curation window on the folder once it is written (pickles or
        # zarr), else on the file itself (every line raw, denoised when clicked)
        written = pf_done or results is not None
        if imgui.button(
            "Curate##voltage_curate", imgui.ImVec2(hello_imgui.em_size(11), 0)
        ):
            self._curate(self._outdir if written else str(mesc))
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Open {'this PF folder' if written else mesc.name + ', its lines raw for the denoiser,'} "
                "in the curation window: its own window, what `mbo curate` opens."
            )

    def _submit(
        self, mesc: Path, scans: list[str], domains: dict, window, planes=None
    ) -> None:
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        first_env = [k.rsplit("_", 1)[-1] for k in scans if self._first_env.get(k)]
        args = {
            "input_path": str(mesc),
            "reader_kwargs": widget_reader_kwargs(
                getattr(self.parent, "image_widget", None)
            ),
            "output_dir": self._outdir,
            "units": scans,
            "first_env": first_env,
            "domains": domains,
            "channel": self._channel(),
            "frames": None if window is None else [window[0], window[1]],
            "planes": planes,
            "settings": self.settings.to_dict(),
            "custom_metadata": dict(getattr(self.parent, "_custom_metadata", {}) or {}),
        }
        rois = "" if planes is None else f", {len(planes)} ROI(s)"
        description = f"Voltage: {len(scans)} scan(s), {len(domains)} domain(s){rois}"
        pid = get_process_manager().spawn(
            task_type="voltage",
            args=args,
            description=description,
            output_path=self._outdir,
        )
        if pid:
            self._set_status(f"Started (PID {pid}); progress in the process console.")
        else:
            self._set_status("Failed to start the worker.", error=True)

    def _load_traces(self, results: Path) -> None:
        """Switch Manual ROI Labeling on and load a results file into its Traces tab."""
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        sync = getattr(self.parent, "sync_manual_roi", None)
        if sync is not None:
            set_widget_enabled("manual_roi", True)
            sync(True)
        widget = getattr(self.parent, "manual_roi", None)
        if widget is None:
            self._set_status(
                "Manual ROI Labeling is unavailable for this view.", error=True
            )
            return
        if widget.load_run(results):
            self._set_status(f"Loaded {results.name} into the Traces tab.")
        else:
            self._set_status(
                widget._run_error or f"Could not load {results.name}.", error=True
            )

    def _curate(self, path: str) -> None:
        # the window module brings hello_imgui and fastplotlib's edge windows
        from mbo_utilities.gui.curation_viewer import launch_curation_window

        pid = launch_curation_window(path, channel=self._channel())
        self._set_status(f"Curation window opened on {Path(path).name} (PID {pid}).")

    def cleanup(self) -> None:
        self._outdir_dialog = None
