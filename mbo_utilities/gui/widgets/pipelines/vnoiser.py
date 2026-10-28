"""vnoiser pipeline widget: domain traces from a .mesc, then the wavelet denoiser.

Top to bottom: the dataset, the output folder and frame window, the Scans
table (every recording of the file traces come from: line scans, chessboard
and ribbon patches, a single patch), the Domains table (each domain's ROIs as
toggles, the pixels it averages, whether this run takes it, and presets that
build the table: one per ROI, all in one, groups, the unit's comment, the ROIs
drawn in the ROI tool), the Trace block (channel, polarity, photon counts,
background, reference channel, dF/F and z-score), a live preview of the
selected domain on the scan on screen (its pixels over the ROI's mean image,
F, dF/F and z, computed off the draw thread by
:func:`~mbo_utilities.vnoiser.pipeline.preview_domain`), the denoiser and peak
settings popup with (?) hints and modified-orange tinting, the modified table
and a green Run button that spawns the "vnoiser" worker. Settings are written
for the archive's frame rate and scaled to the scans'. Results are the run's
results zarr beside the input (``pkl`` writes a PF folder instead), which the
Curate button opens in the curation window (``mbo curate``, its own process).
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, implot
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities.analysis.roi_traces import (
    PIXEL_RULES,
    Domain,
)
from mbo_utilities.arrays.features._dim_labels import slider_roles
from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    button_width,
    checkbox_with_tooltip,
    draw_boxed_label,
    fit_width,
    selected_button_style,
    set_tooltip,
    settings_row,
    settings_table,
    tooltip_marks_right,
)
from mbo_utilities.gui._selection_ui import draw_selection_table, resolve_dim_labels
from mbo_utilities.gui.imgui.lines import (
    X_AXIS_HIDDEN,
    decimate_minmax,
    line,
    plot_style,
    subplots,
)
from mbo_utilities.gui.imgui.panels import draw_keybinds_popup
from mbo_utilities.gui.imgui.vnoiser_help import (
    KEYBINDS,
    TOOLTIP,
    draw_vnoiser_help,
)
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
_MASK_COLOR = (1.0, 0.35, 0.25, 0.9)
_BACKGROUND_COLOR = (0.95, 0.65, 0.2)
_PIXEL_MEAN_COLOR = (0.55, 0.55, 0.6)
_TRACE_COLOR = (0.9, 0.92, 0.95)
_F0_COLOR = (0.35, 0.65, 1.0)
_IMPORTANT_FIELDS = {
    "thres_type",
    "soft_levels",
    "thres_bp_sd",
    "thres_amp_sd",
}
DOMAINS_FILE = "domains.json"
# ROIs drawn as one toggle each up to this many; more take a typed list
MAX_ROI_TOGGLES = 48
PREVIEW_SECONDS = (5.0, 15.0, 30.0, 60.0, 0.0)
# what the Scans table calls a MESc modality
KIND_LABELS = {
    "linescan": "lines",
    "multiline": "lines",
    "chessboard": "chessboard",
    "ribbon_transverse": "ribbon",
    "ribbon_longitudinal": "ribbon",
    "timeseries": "raster",
    "multicube": "cubes",
}

PIXEL_LABELS = {
    "all": "every pixel",
    "bright": "brightest",
    "active": "most active",
    "correlated": "most correlated",
    "mask": "drawn mask",
}
PIXEL_HELP = (
    "Which pixels of each of the domain's ROIs are averaged.\n\n"
    "every pixel: the whole line or patch (what MESc reports).\n"
    "brightest: the top fraction by mean brightness (the stained membrane).\n"
    "most active: the top fraction by fast fluctuation (SD after a high-pass).\n"
    "most correlated: the top fraction whose fast fluctuation follows the ROI's "
    "mean best (the pixels that share the cell's signal).\n"
    "drawn mask: the pixels of an ROI drawn in the ROI tool ('Drawn ROIs')."
)
POLARITY_LABELS = {
    "flip_z": "dims: flip z (vnoiser)",
    "invert_f": "dims: invert F (pml_utilities)",
    "none": "brightens: as recorded",
}
POLARITY_HELP = (
    "Which way the indicator's events point, and who turns them up.\n\n"
    "dims: flip z (vnoiser): JEDI and ASAP dim on depolarization; vnoiser negates "
    "the z-score, the archive's way. dF/F stays negative-going.\n"
    "dims: invert F (pml_utilities): the trace is turned over about its mean "
    "(2 x mean - F) before dF/F, which is the viewer's Invert Deflection on these "
    "pixels; dF/F and z both rise.\n"
    "brightens: as recorded: an indicator that brightens.\n\n"
    "Exactly one inversion is ever applied. The viewer's own Invert Deflection is "
    "display only and never reaches the run."
)
BACKGROUND_LABELS = {
    "none": "none",
    "dim": "dimmest pixels",
    "outside": "outside the domain",
    "domain": "a background domain",
}
BACKGROUND_HELP = (
    "Subtracted from each domain's mean before dF/F.\n\n"
    "dimmest pixels: a low percentile of the ROI's pixels, frame by frame.\n"
    "outside the domain: the mean of the ROI's pixels the domain leaves out "
    "(a domain of every pixel gets none).\n"
    "a background domain: the F of one domain of the table (often 'bg'), "
    "which then gets no trace of its own."
)
REFERENCE_LABELS = {"none": "none", "ratio": "divide", "regress": "regress out"}
REFERENCE_HELP = (
    "A structural channel recorded beside the indicator (mScarlet beside ASAP7), "
    "read through the same pixels.\n\n"
    "divide: F / R x mean(R), for brightness changes both channels share.\n"
    "regress out: removes the least-squares fit of R from F, for motion and "
    "hemodynamic artifacts."
)
BASELINE_LABELS = {
    "gaussian": "Gaussian (archive)",
    "percentile": "rolling percentile",
    "maxmin": "rolling max-min",
    "exponential": "bleaching fit",
    "window": "fixed window (MESc F0)",
    "none": "none (z of F)",
}
BASELINE_HELP = (
    "F0 under dF/F = (F - F0) / F0.\n\n"
    "Gaussian: a Gaussian low-pass of F, the archive's.\n"
    "rolling percentile / max-min: a low envelope over a window, robust to spikes.\n"
    "bleaching fit: a decaying exponential plus a constant.\n"
    "fixed window: the mean over a stretch you name, as MESc's dF/F does.\n"
    "none: no division; the z-score is taken of F itself."
)


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


@dataclass
class DomainRow:
    """One row of the Domains table: a domain and whether this run takes it."""

    name: str
    rois: list[int]
    pixels: str = "all"
    fraction: float = 0.3
    mask: dict[int, list[int]] | None = None
    run: bool = True

    def domain(self) -> Domain:
        return Domain(self.name, self.rois, self.pixels, self.fraction, self.mask)

    def entry(self):
        return self.domain().to_json()

    @classmethod
    def from_entry(cls, name, value, run: bool = True) -> DomainRow:
        d = Domain.from_json(name, value)
        return cls(d.name, list(d.rois), d.pixels, d.fraction, d.mask, run)


def one_per_roi(n_rois: int, names=None) -> list[DomainRow]:
    """``roi0: [0]``, ``roi1: [1]``, ... (or ``names`` when it has one per ROI)."""
    names = names if names and len(names) == n_rois else None
    return [DomainRow(names[i] if names else f"roi{i}", [i]) for i in range(n_rois)]


class TracePreview:
    """One domain's trace on one scan, computed on a worker thread so the tab
    never waits for a read: ``result`` is :func:`preview_domain`'s dict for
    the last request finished, ``error`` why it failed. A request while one
    runs replaces any queued one, so dragging a slider costs one run at most.
    """

    def __init__(self):
        self.key = None
        self.result: dict | None = None
        self.error = ""
        self.busy = False
        self._queued: dict | None = None
        self._lock = threading.Lock()

    def request(self, key, kwargs: dict) -> None:
        with self._lock:
            if key == self.key:
                return
            self.key = key
            if self.busy:
                self._queued = kwargs
                return
            self.busy = True
        threading.Thread(target=self._work, args=(key, kwargs), daemon=True).start()

    def _work(self, key, kwargs: dict) -> None:
        from mbo_utilities.vnoiser.pipeline import preview_domain

        while True:
            try:
                result, error = preview_domain(**kwargs), ""
            except Exception as e:  # noqa: BLE001 - shown in the preview
                result, error = None, str(e)
            with self._lock:
                if key == self.key:
                    self.result, self.error = result, error
                if self._queued is None:
                    self.busy = False
                    return
                kwargs, self._queued = self._queued, None
                key = self.key

    def wait(self, timeout: float = 30.0) -> None:
        """Block until no request is running (tests)."""
        end = time.monotonic() + timeout
        while self.busy and time.monotonic() < end:
            time.sleep(0.01)


class VnoiserPipelineWidget(PipelineWidget):
    """Recordings of a .mesc (line scans, chessboard or ribbon patches, single patches) to a vnoiser results file."""

    name = "vnoiser"
    is_available = HAS_VNOISER
    install_command = VNOISER_HINT
    # frames are a window; the domains say which ROIs (the unit's Z) are read
    # and the Trace block which channel, so neither is a slice row
    axes_consumed = {"T": "range", "Z": "none", "C": "none"}
    seeds_from_view = True

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """True for any unit of a .mesc that holds a recording traces come
        from (the unit on screen need not be one) and for a vnoiser run
        whose source recording is reachable.
        """
        from mbo_utilities.arrays.mesc import ROI_LAYOUTS

        arr = base_array(arr)
        if isinstance(arr, ResultsArray):
            return arr.pipeline == "vnoiser" and arr.source_recording is not None
        if (getattr(arr, "metadata", None) or {}).get("mesc_layout") in ROI_LAYOUTS:
            return True
        filenames = getattr(arr, "filenames", None) or []
        if not filenames or Path(str(filenames[0])).suffix.lower() != ".mesc":
            return False
        from mbo_utilities.arrays.mesc import is_trace_unit, list_mesc_units

        try:
            return any(is_trace_unit(u) for u in list_mesc_units(filenames[0]))
        except Exception:
            return False

    def __init__(self, parent: Any):
        super().__init__(parent)
        from mbo_utilities.vnoiser.params import VnoiserSettings

        self.settings = VnoiserSettings()
        self._outdir = ""
        self._outdir_dialog = None
        self._last_status = ""
        self._status_color = _DIM_COLOR
        # the spike-average windows this tab started, until they exit
        self._windows: list = []
        self._show_settings_popup = False
        self._settings_sizer: PopupAutoSize | None = None
        self._show_slice_popup = False
        self._last_fpath = None
        self._last_unit = None
        self._units: list[dict] = []
        self._scans: dict[str, bool] = {}
        self._first_env: dict[str, bool] = {}
        self._domains: list[DomainRow] = []
        self._selected = 0
        self._group_n = 3
        self._domain_error = ""
        self._domains_path = ""
        self._channel_index = 0
        self._preview = TracePreview()
        self._preview_open = True
        self._preview_seconds = 15.0
        self._preview_roi = 0
        # the preview result last drawn, so a new one refits the time axis
        self._drawn: dict | None = None
        # F, dF/F and z; the bottom row also holds the shared time axis
        self._preview_ratios = implot.SubplotsRowColRatios(row_ratios=[1.0, 1.0, 1.35])
        self._help_open = False
        self._help_keys = False

    def _array(self):
        iw = getattr(self.parent, "image_widget", None)
        if iw is None or not iw.data:
            return None
        return base_array(iw.data[0])

    def _dims(self) -> tuple[int, int, int]:
        """(frames, ROIs, channels) of the scans to process: the shortest ticked unit, else the first."""
        ticked = [u for u in self._units if self._scans.get(u["key"])] or self._units[
            :1
        ]
        if not ticked:
            return 1000, 1, 1
        unit = min(ticked, key=lambda u: int(u["nframes"]))
        return (
            int(unit["nframes"]),
            min(int(u["nrois"]) or 1 for u in ticked),
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
        from mbo_utilities.vnoiser import vnoiser_run_for_mesc
        from mbo_utilities.vnoiser.params import VnoiserSettings
        from mbo_utilities.vnoiser.pipeline import read_domains, scan_units

        fpath = getattr(self.parent, "fpath", None)
        shown = getattr(self._array(), "unit_key", None)
        if fpath == self._last_fpath:
            # another unit of the same file on screen: the selection follows it
            if shown != self._last_unit and any(u["key"] == shown for u in self._units):
                was = self._dims()[1]
                self._last_unit = shown
                self._scans = {u["key"]: u["key"] == shown for u in self._units}
                self._first_env = {shown: True}
                self._seed_slicing()
                self._fit_domains(was)
            return
        self._last_fpath = fpath
        self._last_unit = shown
        self._units, self._scans, self._first_env, self._domains = [], {}, {}, []
        self._domain_error, self._domains_path = "", ""
        mesc = self._mesc_path()
        if mesc is None:
            return
        try:
            self._units = scan_units(mesc)
        except Exception as e:
            self._set_status(f"Cannot list the file's units: {e}", error=True)
            return
        # the unit on screen when it has traces, else every such unit in the file
        on_screen = any(u["key"] == shown for u in self._units)
        for u in self._units:
            self._scans[u["key"]] = u["key"] == shown or not on_screen
        if self._units:
            self._first_env[shown if on_screen else self._units[0]["key"]] = True
        self._seed_slicing()
        arr = self._array()
        run = arr.path if isinstance(arr, ResultsArray) else vnoiser_run_for_mesc(mesc)
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
                # the table the run was given; its own is cut down to the ROIs it read
                domains = (prov.get("source") or {}).get("domains") or domains
                self.settings = VnoiserSettings.from_provenance(prov)
                domains = {
                    k: v for k, v in domains.items() if k not in EXCLUDED_DOMAINS
                }
                # a table made for scans with another number of ROIs is not the shown scan's
                ran = set(((prov.get("source") or {}).get("units") or {}).values())
                counts = {int(u["nrois"]) for u in self._units if u["key"] in ran}
                if on_screen and counts and self._dims()[1] not in counts:
                    domains = {}
                self._set_status(
                    f"Loaded the previous run's {'scans and domains' if domains else 'settings'} from {run.name}"
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
        if run is None and getattr(self.parent, "invert_deflection", False):
            # the viewer shows the movie inverted: the run inverts the same way, once
            self.settings.polarity = "invert_f"
        # after the previous run's settings land: the folder follows the output format
        self._outdir = self._default_outdir()
        rows = []
        for name, value in domains.items():
            try:
                rows.append(DomainRow.from_entry(name, value))
            except ValueError as e:
                self._domain_error = f"{name}: {e}"
        self._domains = rows
        # a previous run's scans only when no unit of the file is on screen
        if scan_ids and not on_screen:
            for u in self._units:
                munit = u["key"].rsplit("_", 1)[-1]
                self._scans[u["key"]] = munit in scan_ids
                self._first_env[u["key"]] = munit in first_env
        self._fit_domains(0)

    def _fit_domains(self, was: int) -> None:
        """One domain per ROI of the ticked scans, unless the table was loaded
        or edited and names only ROIs they have. ``was`` is the ROI count the
        table was last fitted to.
        """
        _, n_rois, _ = self._dims()
        default = [(f"roi{i}", [i]) for i in range(was)]
        fits = [(r.name, r.rois) for r in self._domains] != default and bool(
            self._domains
        )
        if any(not 0 <= roi < n_rois for r in self._domains for roi in r.rois):
            fits = False
        if not fits:
            self._domains = one_per_roi(n_rois)
        self._selected = min(self._selected, max(len(self._domains) - 1, 0))

    def _seed_slicing(self) -> None:
        """Every frame of the ticked scans; the channel stays as chosen when the scans have it."""
        max_frames, _, num_channels = self._dims()
        self._vnoiser_tp_selection = f"1:{max_frames}"
        self._vnoiser_tp_parsed = None
        self._vnoiser_tp_error = ""
        self._channel_index = min(self._channel_index, max(num_channels - 1, 0))

    def seed_from_view(self) -> None:
        """Tick only the recording on screen, run only the domains holding
        the ROI its slider is on (one of its own when none does), read the
        channel on screen, over every frame. A unit without traces on screen
        (a picture) leaves the scans as seeded.
        """
        self._ensure_state()
        shown = getattr(self._array(), "unit_key", None)
        if not any(u["key"] == shown for u in self._units):
            self._seed_slicing()
            return
        was = self._dims()[1]
        self._scans = {u["key"]: u["key"] == shown for u in self._units}
        self._seed_slicing()
        self._fit_domains(was)
        iw = getattr(self.parent, "image_widget", None)
        names = tuple(getattr(iw, "dim_names", None) or ())
        # the sliders are the array's T, C, Z axes by position; Z is the ROI index here
        roles = {role: name for name, role in slider_roles(names).items()}
        if roles.get("z") is not None:
            roi = int(iw.indices[roles["z"]])
            if not any(roi in r.rois for r in self._domains):
                self._domains.append(DomainRow(f"roi{roi}", [roi]))
            for r in self._domains:
                r.run = roi in r.rois
            self._selected = next(i for i, r in enumerate(self._domains) if r.run)
        if roles.get("c") is not None:
            self._channel_index = int(iw.indices[roles["c"]])

    def _check_domains(self) -> list[DomainRow]:
        """The rows as checked domains; sets ``_domain_error`` and returns [] when one is invalid."""
        _, n_rois, _ = self._dims()
        seen = set()
        for r in self._domains:
            name = r.name.strip()
            if not name:
                self._domain_error = "A domain has no name."
                return []
            if name in seen:
                self._domain_error = f"Domain {name} is listed twice."
                return []
            seen.add(name)
            bad = [roi for roi in r.rois if not 0 <= roi < n_rois]
            if not r.rois:
                self._domain_error = f"{name}: no ROI."
                return []
            if bad:
                self._domain_error = f"{name}: ROI {bad[0]} is outside 0..{n_rois - 1}."
                return []
            try:
                r.domain()
            except ValueError as e:
                self._domain_error = str(e)
                return []
        if not self._domains:
            self._domain_error = "No domains."
            return []
        self._domain_error = ""
        return list(self._domains)

    def _domains_to_run(self) -> dict:
        """``{name: entry}`` of the ticked rows, and the background domain when the trace uses one."""
        rows = self._check_domains()
        background = (
            self.settings.traces.background_domain
            if self.settings.traces.background == "domain"
            else None
        )
        out = {
            r.name.strip(): r.entry()
            for r in rows
            if r.run or r.name.strip() == background
        }
        if rows and not [n for n in out if n != background]:
            self._domain_error = "Tick at least one domain to run."
            return {}
        if background is not None and background not in out:
            self._domain_error = (
                f"The background domain {background!r} is not in the table."
            )
            return {}
        return out

    def _frame_window(self) -> tuple[int, int] | None:
        """``(start, stop)`` of the frame selection; None means every frame; raises on a stride."""
        parsed = getattr(self, "_vnoiser_tp_parsed", None)
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

    def _channel(self) -> int:
        return int(self._channel_index)

    def draw_config(self) -> None:
        self._ensure_state()
        with (
            imgui_ctx.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)),
            fit_width(),
        ):
            imgui.spacing()
            if imgui.small_button(
                f"{fa.ICON_FA_CIRCLE_QUESTION} vnoiser guide##vnoiser_help"
            ):
                self._help_open = not self._help_open
            set_tooltip(TOOLTIP, show_mark=False)
            self._draw_dataset_block()
            imgui.separator()
            self._draw_output_row()
            self._draw_slice_row()
            imgui.spacing()
            self._draw_scans_block()
            imgui.spacing()
            self._draw_domains_block()
            imgui.spacing()
            self._draw_trace_block()
            imgui.spacing()
            self._draw_preview()
            imgui.spacing()
            if imgui.button(
                "Denoiser and peaks##vnoiser_settings",
                imgui.ImVec2(hello_imgui.em_size(11), 0),
            ):
                self._show_settings_popup = True
            set_tooltip(
                "The wavelet denoiser, the peak detector, start-up samples and output options. "
                "Defaults are the archive's, so an untouched run reproduces its PF folders."
            )
            self._draw_settings_popup()
            imgui.spacing()
            self._draw_modified_table()
            imgui.spacing()
            self._draw_run()
        self._help_open, self._help_keys = draw_vnoiser_help(
            self._help_open, self._help_keys
        )
        self._help_keys = draw_keybinds_popup(
            KEYBINDS, self._help_keys, "Curation keybinds"
        )

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
        imgui.text_disabled(f"{len(self._units)} recording(s) with traces in the file")
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
                set_last_dir("vnoiser_outdir", result)
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
        _, self._outdir = imgui.input_text("##vnoiser_outdir", self._outdir)
        if self._outdir and imgui.is_item_hovered():
            imgui.set_tooltip(self._outdir)
        imgui.same_line()
        if imgui.button("Browse##vnoiser_outdir_btn", imgui.ImVec2(btn_w, 0)):
            start = str(get_last_dir("vnoiser_outdir") or Path.home())
            self._outdir_dialog = pfd.select_folder("Select output folder", start)

    def _draw_slice_row(self) -> None:
        max_frames, _, _ = self._dims()
        if imgui.button(
            "Frames##vnoiser_slice", imgui.ImVec2(hello_imgui.em_size(6), 0)
        ):
            self._show_slice_popup = True
        set_tooltip(
            "The window of frames read: one contiguous stretch, since the baselines and the "
            "z-score use the whole trace.",
            show_mark=False,
        )
        imgui.same_line()
        try:
            window = self._frame_window()
            n_frames = max_frames if window is None else window[1] - window[0]
            imgui.text_colored(_DIM_COLOR, f"{n_frames} of {max_frames} frames")
        except ValueError as e:
            imgui.text_colored(_MISSING_COLOR, str(e))
        if self._show_slice_popup:
            imgui.open_popup("Frames##vnoiser_slice_popup")
            self._show_slice_popup = False
        imgui.set_next_window_size(
            imgui.ImVec2(hello_imgui.em_size(36), 0), imgui.Cond_.first_use_ever
        )
        if imgui.begin_popup("Frames##vnoiser_slice_popup"):
            tp_label, _z_label, _c_label = resolve_dim_labels(self.parent)
            draw_selection_table(
                self,
                max_frames,
                1,
                tp_attr="_vnoiser_tp",
                z_attr="_vnoiser_z",
                id_suffix="_vnoiser",
                tp_label=tp_label,
                axes=self.axes_consumed,
            )
            imgui.spacing()
            if imgui.button(
                "Close##vnoiser_slice_close", imgui.ImVec2(hello_imgui.em_size(6), 0)
            ):
                imgui.close_current_popup()
            imgui.end_popup()

    def _draw_scans_block(self) -> None:
        imgui.text_colored(_SUBSECTION_COLOR, "Scans")
        set_tooltip(
            "The recordings in this file traces come from: line scans, chessboard and "
            "ribbon scans, a single patch. Ticked ones are processed together with the "
            "same domains; the one on screen is ticked by default."
        )
        if not self._units:
            imgui.text_disabled("No recording with traces in this file.")
            return
        shown = getattr(self._array(), "unit_key", None)
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_fixed_fit
        )
        if imgui.begin_table("##vnoiser_scans", 5, flags):
            fixed = imgui.TableColumnFlags_.width_fixed
            imgui.table_setup_column("Unit", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column(
                "Kind", fixed, imgui.calc_text_size("chessboard").x
            )
            imgui.table_setup_column("ROIs", fixed)
            imgui.table_setup_column("Length", fixed)
            imgui.table_setup_column("New env.", fixed)
            imgui.table_headers_row()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "New env.: tick when this scan is the first one recorded after the "
                    "animal was moved to a different environment (another arena or track). "
                    "The pipeline does not use it; it is written to the output "
                    "(scanID_1st_env) so later analysis can group scans by environment."
                )
            for u in self._units:
                key = u["key"]
                munit = key.rsplit("/", 1)[-1]
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                label = f"{munit} (on screen)" if key == shown else munit
                _, self._scans[key] = imgui.checkbox(
                    f"{label}##vnoiser_scan_{key}", self._scans.get(key, False)
                )
                if imgui.is_item_hovered():
                    comment = str(u.get("comment") or "")
                    imgui.set_tooltip(f"{key}\n{comment}" if comment else key)
                imgui.table_set_column_index(1)
                kind = str(u.get("modality_name") or u.get("kind") or "")
                imgui.text(KIND_LABELS.get(kind, kind))
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"{kind}, {u.get('outline_kind') or 'frame'} ROIs"
                    )
                imgui.table_set_column_index(2)
                imgui.text(str(u["nrois"]))
                imgui.table_set_column_index(3)
                seconds = u.get("duration_s")
                imgui.text(f"{seconds:.0f} s" if seconds else "-")
                imgui.table_set_column_index(4)
                if not self._scans[key]:
                    imgui.begin_disabled()
                _, self._first_env[key] = imgui.checkbox(
                    f"##vnoiser_env_{key}", self._first_env.get(key, False)
                )
                if not self._scans[key]:
                    imgui.end_disabled()
            imgui.end_table()
        if imgui.small_button("On screen only##vnoiser_scans_shown"):
            self._scans = {u["key"]: u["key"] == shown for u in self._units}
        imgui.same_line()
        if imgui.small_button("All##vnoiser_scans_all"):
            self._scans = {u["key"]: True for u in self._units}

    def _comment_names(self) -> list[str] | None:
        from mbo_utilities.vnoiser.pipeline import comment_names

        ticked = [u for u in self._units if self._scans.get(u["key"])]
        if not ticked:
            return None
        return comment_names(ticked[0].get("comment"), int(ticked[0]["nrois"]))

    def drawn_domains(self) -> list[DomainRow]:
        """The ROI tool's ROIs on the recording on screen, one mask domain
        each, named by their note or ``drawn<uid>``; [] without the tool or
        another recording on screen.
        """
        from mbo_utilities.analysis.roi_traces import roi_geometry

        widget = getattr(self.parent, "manual_roi", None)
        arr = self._array()
        if widget is None or arr is None:
            return []
        if not self._scans.get(getattr(arr, "unit_key", None)):
            return []
        store = widget.store
        geometry = roi_geometry(arr)
        rows = []
        for i, record in enumerate(store.rois):
            z = int(store.roi_z(i))
            if z not in geometry:
                continue
            name = (record.note or "").strip() or f"drawn{record.uid}"
            try:
                d = Domain.from_frame_mask(name, z, store.mask(i), geometry)
            except ValueError:
                continue
            rows.append(DomainRow(d.name, d.rois, d.pixels, d.fraction, d.mask))
        return rows

    @staticmethod
    def _inline(label: str, right: float) -> None:
        """Keep the next button on the line when ``label`` fits before ``right``."""
        spacing = imgui.get_style().item_spacing.x
        if imgui.get_item_rect_max().x + spacing + button_width(label) <= right:
            imgui.same_line()

    def _draw_domain_presets(self, n_rois: int) -> None:
        """Buttons that rebuild the table, wrapping to new lines when the panel is narrow."""
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        if imgui.small_button("One per ROI##vnoiser_dom_each"):
            self._domains = one_per_roi(n_rois, self._comment_names())
            self._selected = 0
        set_tooltip("Every ROI its own domain, every pixel.", show_mark=False)
        self._inline("All in one", right)
        if imgui.small_button("All in one##vnoiser_dom_all"):
            self._domains = [DomainRow("all", list(range(n_rois)))]
            self._selected = 0
        set_tooltip("One domain of every ROI.", show_mark=False)
        self._inline("Groups of 00", right)
        if imgui.small_button(f"Groups of {self._group_n}##vnoiser_dom_groups"):
            k = max(1, int(self._group_n))
            self._domains = [
                DomainRow(f"domain{i + 1}", list(range(lo, min(lo + k, n_rois))))
                for i, lo in enumerate(range(0, n_rois, k))
            ]
            self._selected = 0
        set_tooltip(
            "Consecutive ROIs in groups (the archive's three lines per soma or branch). "
            "Right-click to change the size.",
            show_mark=False,
        )
        if imgui.begin_popup_context_item("##vnoiser_dom_groups_ctx"):
            imgui.set_next_item_width(hello_imgui.em_size(6))
            _, self._group_n = imgui.input_int("ROIs per group", int(self._group_n))
            self._group_n = max(1, self._group_n)
            imgui.end_popup()
        names = self._comment_names()
        if names:
            self._inline("From comment", right)
            if imgui.small_button("From comment##vnoiser_dom_comment"):
                self._domains = one_per_roi(n_rois, names)
                self._selected = 0
            set_tooltip(f"One per ROI, named {', '.join(names)}.", show_mark=False)
        drawn = self.drawn_domains()
        self._inline("Drawn ROIs (00)", right)
        if not drawn:
            imgui.begin_disabled()
        if imgui.small_button(f"Drawn ROIs ({len(drawn)})##vnoiser_dom_drawn"):
            have = {r.name for r in self._domains}
            for row in drawn:
                if row.name in have:
                    self._domains = [
                        row if r.name == row.name else r for r in self._domains
                    ]
                else:
                    self._domains.append(row)
            self._selected = len(self._domains) - 1
        if not drawn:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "Add every ROI drawn in the ROI tool (Manual ROI Labeling) on the recording "
                "on screen as a domain of exactly its pixels, named by its note."
                if drawn
                else "Draw ROIs with the ROI tool on the recording on screen (a to draw, r to "
                "add) and they appear here as domains of exactly their pixels."
            )
        self._inline("Load", right)
        if imgui.small_button("Load##vnoiser_dom_load"):
            self._load_domains_file()
        set_tooltip(f"Load {DOMAINS_FILE} from beside the file.", show_mark=False)
        self._inline("Save", right)
        if imgui.small_button("Save##vnoiser_dom_save"):
            self._save_domains_file()
        set_tooltip(
            f"Save the domains, ticked scans and new-environment ticks to {DOMAINS_FILE} beside the file.",
            show_mark=False,
        )

    def _draw_roi_toggles(self, i: int, row: DomainRow, n_rois: int) -> None:
        """One toggle per ROI (a typed list past ``MAX_ROI_TOGGLES``); a mask row's ROIs are fixed."""
        if row.pixels == "mask":
            imgui.text(", ".join(str(r) for r in row.rois))
            if imgui.is_item_hovered():
                imgui.set_tooltip("A drawn mask belongs to the ROI it was drawn on.")
            return
        if n_rois == 1:
            imgui.text_disabled("0 (the frame)")
            return
        if n_rois > MAX_ROI_TOGGLES:
            imgui.set_next_item_width(-1)
            text = ",".join(str(r) for r in row.rois)
            changed, text = imgui.input_text(f"##vnoiser_dom_rois_{i}", text)
            if changed:
                try:
                    row.rois = parse_roi_text(text, n_rois)
                except ValueError:
                    pass
            return
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x * 0.5
        for roi in range(n_rois):
            label = f"{roi}##vnoiser_dom_{i}_roi_{roi}"
            if (
                roi
                and imgui.get_item_rect_max().x + spacing + button_width(str(roi))
                <= right
            ):
                imgui.same_line(0, spacing)
            with selected_button_style(roi in row.rois):
                if imgui.small_button(label):
                    rois = set(row.rois) ^ {roi}
                    if rois:
                        row.rois = sorted(rois)

    def _draw_domains_block(self) -> None:
        imgui.text_colored(_SUBSECTION_COLOR, "Domains")
        set_tooltip(
            "A domain is one trace: the pixels of one or more ROIs averaged together "
            "(the lines of a soma, a patch, the part of a patch the cell covers).\n\n"
            "Run: whether this run takes the domain. The dot picks the domain the preview "
            "shows. ROIs: click to add or remove a line or patch (0-based, drawing order). "
            "Pixels: which of each ROI's pixels are averaged.\n\n"
            "The buttons rebuild the table; Drawn ROIs adds the ROI tool's ROIs."
        )
        _, n_rois, _ = self._dims()
        self._draw_domain_presets(n_rois)
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_stretch_prop
        )
        remove = None
        pick_w = imgui.get_frame_height()
        if imgui.begin_table("##vnoiser_domains", 6, flags):
            imgui.table_setup_column("", imgui.TableColumnFlags_.width_fixed, pick_w)
            imgui.table_setup_column("Run", imgui.TableColumnFlags_.width_fixed, pick_w)
            imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch, 2.0)
            imgui.table_setup_column("ROIs", imgui.TableColumnFlags_.width_stretch, 3.0)
            imgui.table_setup_column(
                "Pixels", imgui.TableColumnFlags_.width_stretch, 3.0
            )
            imgui.table_setup_column(
                "", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(1.8)
            )
            imgui.table_headers_row()
            for i, row in enumerate(self._domains):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                if imgui.radio_button(f"##vnoiser_dom_pick_{i}", self._selected == i):
                    self._selected = i
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Preview this domain.")
                imgui.table_set_column_index(1)
                _, row.run = imgui.checkbox(f"##vnoiser_dom_run_{i}", row.run)
                imgui.table_set_column_index(2)
                imgui.set_next_item_width(-1)
                _, row.name = imgui.input_text(f"##vnoiser_dom_name_{i}", row.name)
                imgui.table_set_column_index(3)
                bad = any(not 0 <= r < n_rois for r in row.rois)
                if bad:
                    imgui.push_style_color(imgui.Col_.text, _MISSING_COLOR)
                self._draw_roi_toggles(i, row, n_rois)
                if bad:
                    imgui.pop_style_color()
                imgui.table_set_column_index(4)
                self._draw_pixel_rule(i, row)
                imgui.table_set_column_index(5)
                if imgui.small_button(f"x##vnoiser_dom_rm_{i}"):
                    remove = i
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Remove this domain.")
            imgui.end_table()
        if remove is not None:
            del self._domains[remove]
            self._selected = min(self._selected, max(len(self._domains) - 1, 0))
        if imgui.small_button("Add##vnoiser_dom_add"):
            self._domains.append(DomainRow(f"domain{len(self._domains) + 1}", [0]))
            self._selected = len(self._domains) - 1
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add a domain of ROI 0; click ROIs to change it.")
        imgui.same_line()
        self._domains_to_run()
        if self._domain_error:
            imgui.text_colored(_MISSING_COLOR, self._domain_error)
        else:
            ticked = [r for r in self._domains if r.run]
            imgui.text_disabled(
                f"{len(ticked)} of {len(self._domains)} domains run, "
                f"{len({roi for r in ticked for roi in r.rois})} ROIs read"
            )
        if self._domains_path:
            shown = _truncate_to_width(
                self._domains_path, imgui.get_content_region_avail().x
            )
            imgui.text_disabled(shown)
            if imgui.is_item_hovered():
                imgui.set_tooltip(self._domains_path)

    def _draw_pixel_rule(self, i: int, row: DomainRow) -> None:
        rules = [p for p in PIXEL_RULES if p != "mask" or row.mask]
        current = rules.index(row.pixels) if row.pixels in rules else 0
        scored = row.pixels in ("bright", "active", "correlated")
        imgui.set_next_item_width(
            -1 if not scored else imgui.get_content_region_avail().x * 0.6
        )
        changed, current = imgui.combo(
            f"##vnoiser_dom_px_{i}", current, [PIXEL_LABELS[p] for p in rules]
        )
        if changed:
            row.pixels = rules[current]
        if imgui.is_item_hovered():
            imgui.set_tooltip(PIXEL_HELP)
        if scored:
            imgui.same_line()
            imgui.set_next_item_width(-1)
            _, percent = imgui.slider_float(
                f"##vnoiser_dom_frac_{i}", row.fraction * 100, 1.0, 100.0, "%.0f %%"
            )
            row.fraction = min(1.0, max(0.01, percent / 100))
            if imgui.is_item_hovered():
                imgui.set_tooltip("The share of each ROI's pixels kept.")

    def _f_combo(
        self, obj, field: str, labels: dict, help_text: str, width=None
    ) -> bool:
        """A combo over ``labels`` (value -> shown) for ``obj.field``, orange when modified."""
        values = list(labels)
        current = (
            values.index(getattr(obj, field)) if getattr(obj, field) in values else 0
        )
        pushed = self._mod_push(obj, field)
        imgui.set_next_item_width(width or hello_imgui.em_size(13))
        changed, current = imgui.combo(
            f"##vnoiser_{field}", current, [labels[v] for v in values]
        )
        if pushed:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(help_text)
        if changed:
            setattr(obj, field, values[current])
        return changed

    def _seconds_field(self, obj, field: str, tooltip: str) -> None:
        """A sample-count field shown and edited in seconds at the reference rate."""
        fs = float(self.settings.runtime.reference_fs or 1.0)
        pushed = self._mod_push(obj, field)
        imgui.set_next_item_width(hello_imgui.em_size(6))
        changed, seconds = imgui.input_float(
            f"##vnoiser_{field}_s", float(getattr(obj, field)) / fs, 0.1, 1.0, "%.2f s"
        )
        if pushed:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(tooltip)
        if changed:
            setattr(obj, field, max(0.0, seconds) * fs)

    def _draw_trace_block(self) -> None:
        from mbo_utilities.vnoiser.params import POLARITIES, VnoiserSettings

        imgui.text_colored(_SUBSECTION_COLOR, "Trace")
        set_tooltip(
            "What each domain's pixel mean goes through before the denoiser, in order: "
            "the conversion offset, the background, the reference channel, the polarity, "
            "then dF/F and the z-score. The preview below shows the result."
        )
        tr, dfof = self.settings.traces, self.settings.dfof
        _, _, n_channels = self._dims()
        captions = (
            "Channel",
            "Polarity",
            "Background",
            "Reference",
            "dF/F baseline",
            "z-score",
            "Stages",
        )
        with settings_table("##vnoiser_trace", captions) as shown:
            if not shown:
                return
            settings_row("Channel")
            imgui.set_next_item_width(hello_imgui.em_size(6))
            changed, index = imgui.combo(
                "##vnoiser_channel",
                min(self._channel_index, n_channels - 1),
                [f"ch {c + 1}" for c in range(n_channels)],
            )
            if changed:
                self._channel_index = index
            if imgui.is_item_hovered():
                imgui.set_tooltip("The channel the indicator is on.")
            imgui.same_line()
            self._f_check(
                tr,
                "convert",
                "photon counts",
                "Apply the file's conversion offset so zero means no photons. MESc's raw "
                "counts sit about 1000 above it, which makes dF/F several times too small. "
                "Off reproduces the archive.",
            )

            settings_row("Polarity")
            pushed = self.settings.polarity != "flip_z"
            if pushed:
                imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
            values = list(POLARITIES)
            imgui.set_next_item_width(hello_imgui.em_size(15))
            changed, index = imgui.combo(
                "##vnoiser_polarity",
                values.index(self.settings.polarity),
                [POLARITY_LABELS[v] for v in values],
            )
            if pushed:
                imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(POLARITY_HELP)
            if changed:
                self.settings.polarity = values[index]
            if getattr(self.parent, "invert_deflection", False):
                imgui.same_line()
                imgui.text_disabled(fa.ICON_FA_CIRCLE_INFO)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "The viewer shows the movie inverted (Invert Deflection). That is "
                        "display only; this setting is the run's one inversion."
                    )

            settings_row("Background")
            self._f_combo(tr, "background", BACKGROUND_LABELS, BACKGROUND_HELP)
            if tr.background == "dim":
                imgui.same_line()
                imgui.set_next_item_width(hello_imgui.em_size(6))
                _, tr.background_percentile = imgui.slider_float(
                    "##vnoiser_bg_pct",
                    float(tr.background_percentile),
                    1.0,
                    50.0,
                    "%.0f th",
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "The percentile of the ROI's pixels taken each frame."
                    )
            elif tr.background == "domain":
                imgui.same_line()
                names = [r.name for r in self._domains] or ["bg"]
                current = (
                    names.index(tr.background_domain)
                    if tr.background_domain in names
                    else 0
                )
                imgui.set_next_item_width(hello_imgui.em_size(7))
                changed, current = imgui.combo("##vnoiser_bg_domain", current, names)
                if changed or tr.background_domain not in names:
                    tr.background_domain = names[current]
            if tr.background != "none":
                imgui.same_line()
                imgui.set_next_item_width(hello_imgui.em_size(5))
                _, tr.background_coeff = imgui.input_float(
                    "##vnoiser_bg_coeff", float(tr.background_coeff), 0.0, 0.0, "x %.2f"
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "The background is multiplied by this before it is subtracted."
                    )

            settings_row("Reference")
            if n_channels < 2:
                imgui.text_disabled("one channel")
                tr.reference = "none"
            else:
                self._f_combo(
                    tr,
                    "reference",
                    REFERENCE_LABELS,
                    REFERENCE_HELP,
                    hello_imgui.em_size(8),
                )
                if tr.reference != "none":
                    imgui.same_line()
                    others = [c for c in range(n_channels) if c != self._channel_index]
                    if tr.reference_channel not in others:
                        tr.reference_channel = others[0]
                    imgui.set_next_item_width(hello_imgui.em_size(6))
                    changed, index = imgui.combo(
                        "##vnoiser_ref_channel",
                        others.index(tr.reference_channel),
                        [f"ch {c + 1}" for c in others],
                    )
                    if changed:
                        tr.reference_channel = others[index]
                else:
                    tr.reference_channel = None

            settings_row("dF/F baseline")
            self._f_combo(dfof, "method", BASELINE_LABELS, BASELINE_HELP)
            if dfof.method == "gaussian":
                imgui.same_line()
                self._seconds_field(
                    dfof,
                    "sigma_dfof",
                    "The Gaussian's sigma (the archive's 1500 samples at 1075 Hz, 1.4 s).",
                )
            elif dfof.method in ("percentile", "maxmin"):
                imgui.same_line()
                imgui.set_next_item_width(hello_imgui.em_size(6))
                _, dfof.window_s = imgui.input_float(
                    "##vnoiser_window_s", float(dfof.window_s), 0.5, 5.0, "%.1f s"
                )
                dfof.window_s = max(0.05, dfof.window_s)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("The rolling window.")
                if dfof.method == "percentile":
                    imgui.same_line()
                    imgui.set_next_item_width(hello_imgui.em_size(6))
                    _, dfof.percentile = imgui.slider_float(
                        "##vnoiser_pct", float(dfof.percentile), 1.0, 50.0, "%.0f th"
                    )
            elif dfof.method == "window":
                imgui.same_line()
                imgui.set_next_item_width(hello_imgui.em_size(9))
                _, span = imgui.input_float2(
                    "##vnoiser_f0",
                    [float(dfof.f0_start_s), float(dfof.f0_stop_s)],
                    "%.2f s",
                )
                dfof.f0_start_s, dfof.f0_stop_s = (
                    max(0.0, span[0]),
                    max(span[0] + 1e-3, span[1]),
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "F0 is the mean of F from the first to the second time."
                    )

            settings_row("z-score")
            self._f_combo(
                dfof,
                "noise",
                {"sd": "/ SD (archive)", "mad": "/ robust SD (MAD)"},
                "What the z-score divides by: the SD of dF/F, or 1.4826 x the median "
                "absolute deviation, which spikes do not inflate.",
                hello_imgui.em_size(10),
            )
            imgui.same_line()
            imgui.text_disabled("slow baseline")
            imgui.same_line()
            self._seconds_field(
                dfof,
                "sigma_baseline",
                "The slow Gaussian removed from dF/F before the z-score (archive: 5000 samples, 4.7 s).",
            )

            settings_row("Stages")
            rt = self.settings.runtime
            denoise = not rt.denoise
            pushed = self._mod_push(rt, "denoise")
            changed, denoise = imgui.checkbox("traces only##vnoiser_denoise", denoise)
            if pushed:
                imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Stop after dF/F and the z-score: no wavelet denoising, no peaks. "
                    "Writes the results zarr."
                )
            if changed:
                rt.denoise = not denoise
                if not rt.denoise:
                    rt.output_format = "zarr"
        if imgui.small_button("Archive defaults##vnoiser_preset_archive"):
            self.settings = VnoiserSettings()
            self._outdir = self._default_outdir()
        set_tooltip(
            "Every setting as the archive (Noguchi & Terada): raw counts, Gaussian dF/F, "
            "SD z-score, z flipped.",
            show_mark=False,
        )
        imgui.same_line()
        if imgui.small_button("Recommended##vnoiser_preset_recommended"):
            self.settings = VnoiserSettings.recommended()
            self._outdir = self._default_outdir()
        set_tooltip(
            "For new recordings: the archive's settings on photon counts (the conversion "
            "offset applied, so dF/F has its real size) with a robust z-score.",
            show_mark=False,
        )

    def _preview_unit(self) -> dict | None:
        """The scan the preview reads: the one on screen when ticked, else the first ticked."""
        shown = getattr(self._array(), "unit_key", None)
        ticked = [u for u in self._units if self._scans.get(u["key"])]
        return next(
            (u for u in ticked if u["key"] == shown), ticked[0] if ticked else None
        )

    def _request_preview(self) -> None:
        mesc = self._mesc_path()
        unit = self._preview_unit()
        if mesc is None or unit is None or not self._domains:
            return
        row = self._domains[min(self._selected, len(self._domains) - 1)]
        table = {r.name.strip(): r for r in self._domains}
        try:
            entries = {row.name.strip(): row.entry()}
            bg = self.settings.traces.background_domain
            if self.settings.traces.background == "domain" and bg in table:
                entries[bg] = table[bg].entry()
            window = self._frame_window()
        except ValueError as e:
            self._preview.error, self._preview.result = str(e), None
            return
        start, stop = window or (0, int(unit["nframes"]))
        if self._preview_seconds > 0 and unit.get("fs"):
            stop = min(
                stop, start + int(round(self._preview_seconds * float(unit["fs"])))
            )
        settings = self.settings.to_dict()
        key = json.dumps(
            [
                str(mesc),
                unit["key"],
                entries,
                row.name,
                settings,
                self._channel_index,
                start,
                stop,
            ],
            sort_keys=True,
            default=str,
        )
        from mbo_utilities.vnoiser.params import VnoiserSettings

        self._preview.request(
            key,
            {
                "mesc_path": mesc,
                "unit": unit["key"],
                "domains": entries,
                "name": row.name.strip(),
                "settings": VnoiserSettings.from_dict(settings),
                "channel": self._channel_index,
                "frames": (start, stop),
            },
        )

    def _draw_preview(self) -> None:
        imgui.set_next_item_open(self._preview_open, imgui.Cond_.once)
        self._preview_open = imgui.collapsing_header("Preview##vnoiser_preview")
        if not self._preview_open:
            return
        unit = self._preview_unit()
        if unit is None or not self._domains:
            imgui.text_disabled("Tick a scan and add a domain to see its trace.")
            return
        row = self._domains[min(self._selected, len(self._domains) - 1)]
        try:
            row.domain()
        except ValueError as e:
            imgui.text_colored(_MISSING_COLOR, str(e))
            return
        # a slider being dragged waits for its release before the next read
        if not imgui.is_any_item_active():
            self._request_preview()
        imgui.text(f"{row.name} on {unit['key'].rsplit('/', 1)[-1]}")
        imgui.same_line()
        lengths = [f"{s:g} s" if s else "whole" for s in PREVIEW_SECONDS]
        current = (
            PREVIEW_SECONDS.index(self._preview_seconds)
            if self._preview_seconds in PREVIEW_SECONDS
            else 1
        )
        imgui.set_next_item_width(hello_imgui.em_size(6))
        changed, current = imgui.combo("##vnoiser_preview_len", current, lengths)
        if changed:
            self._preview_seconds = PREVIEW_SECONDS[current]
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "How much of the recording the preview reads, from the start of the frame window."
            )
        if self._preview.busy:
            imgui.same_line()
            imgui.text_disabled("reading...")
        if self._preview.error:
            imgui.text_colored(_MISSING_COLOR, self._preview.error)
            return
        result = self._preview.result
        if result is None:
            return
        self._draw_preview_image(result)
        imgui.same_line()
        self._draw_preview_traces(result)
        z = result["z"]
        dfof = result["dfof"]
        pixels = sum(len(v) for v in result["mask"].values())
        total = sum(h * w for h, w in result["shape"].values())
        imgui.text_disabled(
            f"{pixels} of {total} pixels · dF/F {np.nanpercentile(dfof, 1) * 100:.1f} to "
            f"{np.nanpercentile(dfof, 99.9) * 100:.1f} % · z peak {np.nanmax(z):.1f}"
        )
        if result["f0"] is not None and np.nanmin(result["f0"]) <= 0:
            imgui.text_colored(
                _MODIFIED_COLOR,
                "The dF/F baseline reaches zero: dF/F is undefined there (shown as 0). "
                "Use photon counts, lower the background coefficient, or take no baseline.",
            )

    def _draw_preview_image(self, result: dict) -> None:
        rois = result["rois"]
        roi = rois[min(self._preview_roi, len(rois) - 1)]
        h, w = result["shape"][roi]
        image = np.ascontiguousarray(result["images"][roi]["mean"], dtype=np.float64)
        side = hello_imgui.em_size(11)
        imgui.begin_group()
        if len(rois) > 1:
            imgui.set_next_item_width(side)
            changed, index = imgui.combo(
                "##vnoiser_preview_roi",
                rois.index(roi),
                [f"ROI {r}" for r in rois],
            )
            if changed:
                self._preview_roi = index
        height = side * min(4.0, max(0.25, h / max(w, 1)))
        with plot_style():
            flags = (
                implot.Flags_.no_title
                | implot.Flags_.no_legend
                | implot.Flags_.no_menus
                | implot.Flags_.no_box_select
                | implot.Flags_.no_mouse_text
                | implot.Flags_.equal
            )
            if implot.begin_plot(
                "##vnoiser_preview_img", imgui.ImVec2(side, height), flags
            ):
                axis = implot.AxisFlags_.no_decorations
                implot.setup_axes("", "", axis, axis)
                implot.setup_axes_limits(0, w, 0, h, implot.Cond_.always)
                lo, hi = (float(v) for v in np.nanpercentile(image, (1, 99.5)))
                # implot's greys run white to black: negated, the bright pixels are white
                implot.push_colormap(implot.Colormap_.greys)
                implot.plot_heatmap(
                    "##mean",
                    np.ascontiguousarray(-image),
                    -(hi if hi > lo else lo + 1),
                    -lo,
                    "",
                    implot.Point(0, 0),
                    implot.Point(w, h),
                )
                implot.pop_colormap()
                mask = np.asarray(result["mask"][roi])
                if mask.size < h * w:
                    rows, cols = np.divmod(mask, w)
                    spec = implot.Spec()
                    spec.marker = implot.Marker_.square
                    spec.marker_size = max(1.0, 0.25 * side / max(w, h))
                    spec.marker_fill_color = imgui.ImVec4(*_MASK_COLOR)
                    spec.marker_line_color = imgui.ImVec4(*_MASK_COLOR)
                    implot.plot_scatter(
                        "##mask",
                        np.ascontiguousarray(cols + 0.5, dtype=np.float64),
                        np.ascontiguousarray(h - rows - 0.5, dtype=np.float64),
                        spec,
                    )
                implot.end_plot()
        imgui.end_group()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"ROI {roi}'s mean image ({h} x {w}); the red pixels are the domain's."
            )

    def _draw_preview_traces(self, result: dict) -> None:
        """F (with the pixel mean, background and F0 when there are), dF/F and
        z stacked on one time axis, refitted when a new result lands.
        """
        t = result["t"]
        fresh = result is not self._drawn
        self._drawn = result
        height = hello_imgui.em_size(17)
        flags = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_title
        inverted = result["polarity"] == "invert_f"
        rows = (
            (
                "F",
                (
                    ("pixel mean", result["pixel_mean"], _PIXEL_MEAN_COLOR),
                    ("background", result["background"], _BACKGROUND_COLOR),
                    ("F", result["measured"], _TRACE_COLOR),
                    ("F0", None if inverted else result["f0"], _F0_COLOR),
                ),
            ),
            ("dF/F", (("dF/F", result["dfof"], _TRACE_COLOR),)),
            ("z", (("z", result["z"], _TRACE_COLOR),)),
        )
        with (
            plot_style(),
            subplots(
                "##vnoiser_preview_traces",
                3,
                1,
                height,
                -1,
                flags,
                self._preview_ratios,
            ) as shown,
        ):
            if not shown:
                return
            for k, (label, lines) in enumerate(rows):
                drawn = [
                    item
                    for item in lines
                    if item[1] is not None
                    and not (item[0] == "pixel mean" and result["background"] is None)
                ]
                legend = len(drawn) > 1
                plot_flags = implot.Flags_.no_title | (
                    0 if legend else implot.Flags_.no_legend
                )
                if not implot.begin_plot(
                    f"##vnoiser_prev_{k}", imgui.ImVec2(-1, -1), plot_flags
                ):
                    continue
                implot.setup_axes(
                    "s" if k == 2 else "",
                    label,
                    implot.AxisFlags_.none if k == 2 else X_AXIS_HIDDEN,
                    implot.AxisFlags_.auto_fit,
                )
                if fresh:
                    implot.setup_axis_limits(
                        implot.ImAxis_.x1,
                        float(t[0]),
                        float(t[-1]),
                        implot.Cond_.always,
                    )
                if legend:
                    implot.setup_legend(
                        implot.Location_.north_east, implot.LegendFlags_.horizontal
                    )
                for name, y, color in drawn:
                    idx, values = decimate_minmax(y, 2000)
                    line(name, values, x=t[idx.astype(int)], color=color, legend=legend)
                implot.end_plot()

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
        _, val = imgui.input_int(f"##vnoiser_{field}", int(getattr(obj, field)))
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
            f"##vnoiser_{field}", float(getattr(obj, field)), step, step * 10, fmt
        )
        if pushed:
            imgui.pop_style_color()
        setattr(obj, field, val)
        self._row_tail(obj, field, label, tooltip)

    def _f_check(self, obj, field: str, label: str, tooltip: str) -> None:
        pushed = self._mod_push(obj, field)
        val = checkbox_with_tooltip(
            f"{label}##vnoiser_{field}", bool(getattr(obj, field)), tooltip
        )
        if pushed:
            imgui.pop_style_color()
        setattr(obj, field, bool(val))

    def _draw_settings_popup(self) -> None:
        popup_title = "vnoiser Settings##vnoiser_settings_popup"
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
                    "##vnoiser_output_format", current, list(OUTPUT_FORMATS)
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
                    "zarr: one <input>.<timestamp>.vnoiser.zarr results file beside the input (the shape "
                    "every pipeline's results share), the run's own files in a vnoiser/ folder inside it. "
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
                if imgui.button("Defaults##vnoiser_defaults", imgui.ImVec2(btn_w, 0)):
                    from mbo_utilities.vnoiser.params import VnoiserSettings

                    self.settings = VnoiserSettings()
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
                    "Close##vnoiser_settings_close", imgui.ImVec2(btn_w, 0)
                ):
                    imgui.close_current_popup()
                imgui.pop_style_color(3)
        finally:
            imgui.end_popup()

    def _draw_settings_columns(self) -> None:
        columns = (
            ("Start-up and pixel scoring", "dfof", self._draw_dfof_params),
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
                f"##vnoiser_col_{key}",
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
        d, tr = self.settings.dfof, self.settings.traces
        self._f_int(
            d,
            "n_startup",
            "Start-up samples",
            "Leading samples replaced by the mean of the rest (filter start-up).",
        )
        self._f_float(
            d,
            "sigma_dfof",
            "Gaussian sigma",
            "The Gaussian dF/F baseline's sigma in samples at the reference rate "
            "(1500 = 1.4 s at 1075 Hz); the Trace block shows it in seconds.",
            step=100,
            fmt="%.0f",
        )
        self._f_float(
            tr,
            "score_seconds",
            "Pixel scoring s",
            "How much of the middle of the recording the brightest / most active / most "
            "correlated pixel rules rank pixels on.",
            step=5,
            fmt="%.0f",
        )
        self._f_float(
            tr,
            "score_highpass_hz",
            "Scoring high-pass Hz",
            "Fluctuations below this are left out of the active and correlated scores, "
            "so slow drift and motion shared by every pixel do not count.",
            step=1,
            fmt="%.0f",
        )

    def _draw_denoiser_params(self) -> None:
        den = self.settings.denoiser
        pushed = self._mod_push(den, "thres_type")
        idx = 0 if den.thres_type == "soft" else 1
        imgui.set_next_item_width(hello_imgui.em_size(6))
        changed, idx = imgui.combo("##vnoiser_thres_type", idx, ["Soft", "Hard"])
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
            "##vnoiser_soft_levels", list(den.soft_levels), "%.2f"
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
        if imgui.collapsing_header("Clustering and windows##vnoiser_adv"):
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
            ("traces", self.settings.traces),
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
            "##vnoiser_mod_params", imgui.ImVec2(-1, mod_h), imgui.ChildFlags_.borders
        ):
            flags = (
                imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h
                | imgui.TableFlags_.sizing_stretch_prop
            )
            if imgui.begin_table("##vnoiser_mod_tbl", 3, flags):
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
        domains = self._domains_to_run()
        try:
            window = self._frame_window()
            window_error = ""
        except ValueError as e:
            window, window_error = None, str(e)
        reason = ""
        if mesc is None:
            reason = "Load a .mesc with a recording to trace first."
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
        clicked = imgui.button("Run vnoiser", imgui.ImVec2(run_w, 0))
        if not ready:
            imgui.end_disabled()
        imgui.pop_style_color(3)
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                reason
                or f"Spawn a worker for {len(scans)} scan(s) and {len(domains)} domain(s); "
                "the process console tracks it."
            )
        if self._last_status:
            imgui.text_colored(self._status_color, self._last_status)
        if clicked and ready:
            self._submit(mesc, scans, domains, window)
        pf_done = (
            bool(self._outdir)
            and (Path(self._outdir) / "denoised_trace_scans.pkl").exists()
        )
        results = (
            newest_results(self._outdir, "vnoiser", source=mesc)
            if self._outdir and not pf_done
            else None
        )
        if results is not None:
            imgui.text_disabled(f"Results: {results.name}")
            if imgui.button(
                "Load into Traces##vnoiser_load_traces",
                imgui.ImVec2(hello_imgui.em_size(11), 0),
            ):
                self._load_traces(results)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Load every scan's domain traces (F, dF/F, z, denoised) and line traces "
                    "into the Traces tab (Manual ROI Labeling)."
                )
            imgui.same_line()
            if imgui.button(
                "Spike average##vnoiser_sta",
                imgui.ImVec2(hello_imgui.em_size(11), 0),
            ):
                self._spike_average(results)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "The ROI on screen averaged around its detected events (or the "
                    "events its curation accepts, or a threshold): the line's movie, "
                    "its trace and RTMC. Its own window."
                )
        for window in list(self._windows):
            state = window.poll()
            if state is not None:
                self._windows.remove(window)
                if state:
                    self._set_status(state, error=True)
        if mesc is None:
            return
        # the curation window on the folder once it is written (pickles or
        # zarr), else on the file itself (every line raw, denoised when clicked)
        written = pf_done or results is not None
        if imgui.button(
            "Curate##vnoiser_curate", imgui.ImVec2(hello_imgui.em_size(11), 0)
        ):
            self._curate(self._outdir if written else str(mesc))
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Open {'this run' if written else mesc.name + ', its lines raw for the denoiser,'} "
                "in the curation window: its own window, what `mbo curate` opens."
            )

    def _submit(self, mesc: Path, scans: list[str], domains: dict, window) -> None:
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
            "invert_deflection": bool(getattr(self.parent, "invert_deflection", False)),
            "settings": self.settings.to_dict(),
            "custom_metadata": dict(getattr(self.parent, "_custom_metadata", {}) or {}),
        }
        description = f"vnoiser: {len(scans)} scan(s), {len(domains)} domain(s)"
        pid = get_process_manager().spawn(
            task_type="vnoiser",
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

    def _spike_average(self, results: Path) -> None:
        """Open the spike-triggered average on the scan and ROI on screen:
        the results unit that processed the shown recording unit, and the ROI
        whose lines hold the line on the ROI slider.
        """
        from mbo_utilities.gui.launch import LaunchedWindow
        from mbo_utilities.gui.spike_average_viewer import launch_spike_average
        from mbo_utilities.results import Results, unit_for_source

        found = Results.read(results)
        iw = getattr(self.parent, "image_widget", None)
        shown = base_array(iw.data[0]) if iw is not None else None
        unit = unit_for_source(found, str(getattr(shown, "unit_key", "") or ""))
        roi = None
        view = getattr(self.parent, "slice", None)
        if unit is not None and view is not None:
            k = found.units[unit].member_roi(view.z)
            roi = None if k is None else found.units[unit].roi_names[k]
        pid, log_name = launch_spike_average(results, unit, roi, self._channel())
        self._windows.append(LaunchedWindow(pid, log_name, "Spike average"))
        self._set_status(
            f"Spike average of {unit or 'the first scan'}{f' {roi}' if roi else ''} "
            f"opened in its own window (PID {pid})."
        )

    def _curate(self, path: str) -> None:
        # the window module brings hello_imgui and fastplotlib's edge windows
        from mbo_utilities.gui.curation_viewer import launch_curation_window

        pid = launch_curation_window(path, channel=self._channel())
        self._set_status(f"Curation window opened on {Path(path).name} (PID {pid}).")

    def cleanup(self) -> None:
        self._outdir_dialog = None
