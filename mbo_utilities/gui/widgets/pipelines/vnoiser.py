"""vnoiser pipeline widget: domain traces from a .mesc, then the wavelet denoiser.

The tab is a summary, top to bottom: the dataset, the output (folder, frame
window, channel), the Scans table (every recording of the file traces come
from: line scans, chessboard and ribbon patches, a single patch; their blocks
behind the Blocks button), the Domains summary (one line per domain with its
run tick), the Parameters button over the table of changed values, and a green
Run button that spawns the "vnoiser" worker. Two modals hold the editing. The
domain editor is one row per domain with a box per ROI, the pixels it averages
and presets that build the table (one per ROI, all in one, groups, the unit's
comment, the ROIs drawn in the ROI tool), over a live preview of the picked
domain on the scan on screen (its pixels over the ROI's mean image, F, dF/F
and z, computed off the draw thread by
:func:`~mbo_utilities.vnoiser.pipeline.preview_domain`). The Parameters popup
is laid out as masknmf's: one bordered column per stage (trace, dF/F and
z-score, wavelet denoiser, peaks) from
:mod:`~mbo_utilities.gui.widgets.pipelines.vnoiser_hints`, a changed value in
orange with a reset button and its default beside it, advanced and stable
fields in folds, the runtime under the columns. Settings are written for the
archive's frame rate and scaled to the scans'. Results are the run's results
zarr beside the input (``pkl`` writes a PF folder instead), which the Curate
button opens in the curation window (``mbo curate``, its own process).
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
import time
from contextlib import contextmanager
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
from mbo_utilities.annotation.domains import (
    DomainFile,
    Layout,
    domains_path,
    find_domains,
    scan_of,
)
from mbo_utilities.arrays.features._dim_labels import slider_roles
from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui._imgui_helpers import (
    button_width,
    draw_boxed_label,
    fit_width,
    selected_button_style,
    set_tooltip,
    settings_row,
    settings_table,
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
    draw_vnoiser_help,
)
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui.widgets.pipelines.settings import (
    _MISSING_COLOR,
    _MODIFIED_COLOR,
    _SUBSECTION_COLOR,
    _dataset_size_bytes,
    _format_size,
    _truncate_to_width,
)
from mbo_utilities.gui.widgets.pipelines.vnoiser_hints import (
    COLUMNS,
    HIDDEN,
    HINTS,
    POLARITY,
    POLARITY_DEFAULT,
    RUNTIME,
    STAGE_DENOISER,
    STAGE_PEAKS,
    applies,
)
from mbo_utilities.install import VNOISER_HINT
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import get_last_dir, get_roi_folder, set_last_dir
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
SCAN_LABELS = ("New env", "Reward switch")  # what an experiment starts with
SCAN_COLUMN_HELP = {
    "New block": "the first scan after a condition changed; a run's first scan always is one",
    "Change": "what changed at this block's first scan; Labels below sets the words",
}
MATRIX_MAX = 32  # one box per ROI up to this many; more take a typed list
WIDTH_INPUT_EM = 7.5
WIDTH_COLUMN_EM = (19.0, 27.0)  # a Parameters column, narrowest and widest
POPUP_PARAMETERS = "vnoiser parameters##vnoiser_parameters"
POPUP_DOMAINS = "vnoiser domains##vnoiser_domains"
COLORS_DEFAULTS = ((0.60, 0.35, 0.10), (0.70, 0.42, 0.14), (0.50, 0.28, 0.08))
COLORS_CLOSE = ((0.55, 0.13, 0.13), (0.65, 0.18, 0.18), (0.45, 0.10, 0.10))
PREVIEW_SECONDS = (5.0, 15.0, 30.0, 60.0, 0.0)
KIND_LABELS = {  # the Scans table's word for a MESc modality
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
DOMAIN_COLUMN_HELP = {
    "View": "The domain the preview below reads.",
    "Run": "Whether this run takes the domain.",
    "Name": "The trace's name everywhere: soma, basal1, apical1, bg.",
    "ROIs": "The ROIs averaged into this trace, as 0,1,2 or 0:2.",
    "Pixels": PIXEL_HELP,
    "Share": "The fraction of each ROI's pixels the rule keeps.",
}


@contextmanager
def button_colors(button, hovered, active):
    """Draw the buttons inside the block in these three RGB colours."""
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*button, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*hovered, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(*active, 1.0))
    yield
    imgui.pop_style_color(3)


def draw_divider() -> None:
    """A rule with air on both sides, between the parts of a panel."""
    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    imgui.spacing()


def same_line_if_fits(width: float) -> None:
    """Keep the next item on the line when ``width`` fits in what is left of it."""
    spacing = imgui.get_style().item_inner_spacing.x
    right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
    if imgui.get_item_rect_max().x + spacing + width <= right:
        imgui.same_line(0, spacing)


def roi_ranges(rois) -> str:
    """ROI indices with runs folded: ``[0, 1, 2, 5]`` is ``0-2, 5``."""
    parts, rois = [], sorted(rois)
    start = 0
    for i in range(1, len(rois) + 1):
        if i == len(rois) or rois[i] != rois[i - 1] + 1:
            first, last = rois[start], rois[i - 1]
            parts.append(str(first) if first == last else f"{first}-{last}")
            start = i
    return ", ".join(parts)


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


def one_per_roi(n_rois: int) -> list[DomainRow]:
    """``roi0: [0]``, ``roi1: [1]``, ..."""
    return [DomainRow(f"roi{i}", [i]) for i in range(n_rois)]


def groups_of(n_rois: int, size: int) -> list[DomainRow]:
    """``domain1: [0, 1, 2]``, ``domain2: [3, 4, 5]``, ...; the last takes what is left."""
    return [
        DomainRow(f"domain{i + 1}", list(range(lo, min(lo + size, n_rois))))
        for i, lo in enumerate(range(0, n_rois, size))
    ]


class TracePreview:
    """One domain's trace on one scan, computed on a worker thread so the tab never waits for a read.

    ``result`` is :func:`preview_domain`'s dict for the last request finished
    and ``error`` its failure message. A request while one runs replaces any
    queued one, so dragging a slider costs one run at most.
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
    axes_consumed = {
        "T": "range",
        "Z": "none",
        "C": "none",
    }  # the domains pick the ROIs (Z), the Trace block the channel
    seeds_from_view = True

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """True for any unit of a .mesc holding a recording traces come from (not necessarily the unit on screen), and for a vnoiser run whose source recording is reachable."""
        from mbo_utilities.arrays.mesc import ROI_LAYOUTS

        arr = base_array(arr)
        if arr is None:
            return False
        if isinstance(arr, ResultsArray):
            return arr.pipeline == "vnoiser" and arr.source_recording is not None
        if (arr.metadata or {}).get("mesc_layout") in ROI_LAYOUTS:
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
        self._windows: list = []  # spike-average windows this tab started, until they exit
        self._spawned: set[int] = (
            set()
        )  # workers this tab started, until each run is loaded
        self._polled = 0.0
        self._show_settings_popup = False
        self._parameters_chrome = (
            0.0  # height of the Parameters popup around its columns
        )
        self._show_domains_popup = False
        self._show_blocks_popup = False
        self._show_slice_popup = False
        self._last_fpath = None
        self._last_unit = None
        self._units: list[dict] = []
        self._scans: dict[str, bool] = {}
        self._first_env: dict[str, bool] = {}
        self._labels: list[str] = list(SCAN_LABELS)
        self._scan_labels: dict[str, list[str]] = {}
        self._new_label = ""
        self._domains: list[DomainRow] = []
        self._selected = 0
        self._group_n = 3
        self._domain_error = ""
        self._domains_path = ""
        self._book_seen: tuple | None = (
            None  # the domain file as last polled: (mtime, the ticked scan's layout)
        )
        self._channel_index = 0
        self._vnoiser_tp_selection = ""
        self._vnoiser_tp_parsed = None
        self._vnoiser_tp_error = ""
        self._preview = TracePreview()
        self._preview_open = True
        self._preview_seconds = 15.0
        self._preview_roi = 0
        self._drawn: dict | None = (
            None  # the result last drawn; a new one refits the time axis
        )
        self._preview_ratios = implot.SubplotsRowColRatios(
            row_ratios=[
                1.0,
                1.0,
                1.35,
            ]  # F, dF/F, z; the bottom row holds the shared time axis
        )
        self._help_open = False
        self._help_keys = False

    def _array(self):
        iw = self.parent.image_widget
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
        fpath = self.parent.fpath
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
        from mbo_utilities.vnoiser.pipeline import scan_units

        fpath = self.parent.fpath
        shown = getattr(self._array(), "unit_key", None)
        if fpath == self._last_fpath:
            # another unit of the same file on screen: the selection follows it
            if shown != self._last_unit and any(u["key"] == shown for u in self._units):
                was = self._dims()[1]
                book = self._read_book()
                left = book and book.layout_of(str(self._last_unit).rsplit("_", 1)[-1])
                entered = book and book.layout_of(str(shown).rsplit("_", 1)[-1])
                self._last_unit = shown
                self._scans = {u["key"]: u["key"] == shown for u in self._units}
                self._seed_slicing()
                if entered and entered is not left:
                    # another layout's scan: its own table, edits to this one stay unsaved
                    self._domains, self._selected = self._layout_rows(entered), 0
                self._fit_domains(was)
            return
        self._last_fpath = fpath
        self._last_unit = shown
        self._units, self._scans, self._first_env, self._domains = [], {}, {}, []
        self._labels, self._scan_labels = list(SCAN_LABELS), {}
        self._domain_error, self._domains_path = "", ""
        self._book_seen = None
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
        self._seed_slicing()
        arr = self._array()
        run = arr.path if isinstance(arr, ResultsArray) else vnoiser_run_for_mesc(mesc)
        domains, scan_ids, first_env, labels, scan_labels = {}, [], [], None, {}
        ticked = None
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
                labels, scan_labels = prov.get("labels"), prov.get("scan_labels")
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
        # the domain file is the table every window reads; it wins over the run's copy
        book = self._read_book()
        if book is not None:
            layout = (
                book.layout_of(str(shown).rsplit("_", 1)[-1])
                if on_screen
                else next(iter(book.layouts), None)
            )
            if layout is not None:
                domains, ticked = layout.domains, layout.run
                scan_ids = layout.scans or scan_ids
            first_env, labels, scan_labels = (
                book.first_env,
                book.labels,
                book.scan_labels,
            )
            self._set_status(f"Domains from {Path(self._domains_path).name}")
        if run is None and self.parent.invert_deflection:
            # the viewer shows the movie inverted: the run inverts the same way, once
            self.settings.polarity = "invert_f"
        # after the previous run's settings land: the folder follows the output format
        self._outdir = self._default_outdir()
        rows = []
        for name, value in domains.items():
            try:
                rows.append(
                    DomainRow.from_entry(name, value, ticked is None or name in ticked)
                )
            except ValueError as e:
                self._domain_error = f"{name}: {e}"
        self._domains = rows
        self._set_blocks(first_env, labels, scan_labels)
        # a previous run's scans only when no unit of the file is on screen
        if scan_ids and not on_screen:
            for u in self._units:
                self._scans[u["key"]] = u["key"].rsplit("_", 1)[-1] in scan_ids
        self._fit_domains(0)

    def _fit_domains(self, was: int) -> None:
        """One domain per ROI of the ticked scans, unless the table was loaded or edited and names only ROIs they have; ``was`` is the ROI count it was last fitted to."""
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
        """Tick only the recording on screen, run only the domains holding the ROI its slider is on (one of its own when none does), read the channel on screen, over every frame.

        A unit without traces on screen (a picture) leaves the scans as seeded.
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
        iw = self.parent.image_widget
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
        parsed = self._vnoiser_tp_parsed
        idx = list(parsed.final_indices) if parsed is not None else []
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
            self._draw_dataset_block()
            self._draw_output_block()
            self._draw_scans_block()
            self._draw_domains_block()
            self._draw_parameters_block()
            draw_divider()
            self._draw_run()
            self._draw_blocks_popup()
            self._draw_domains_popup()
            self._draw_parameters()
        self._help_open, self._help_keys = draw_vnoiser_help(
            self._help_open, self._help_keys
        )
        self._help_keys = draw_keybinds_popup(
            KEYBINDS, self._help_keys, "Curation keybinds"
        )

    def _draw_dataset_block(self) -> None:
        imgui.separator_text("Dataset")
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
        facts = [f"{str(getattr(arr, 'unit_key', '?')).rsplit('/', 1)[-1]} on screen"]
        ticked = [u for u in self._units if self._scans.get(u["key"])]
        if ticked:
            facts.append(f"{float(ticked[0]['fs']):.1f} Hz")
        if filenames:
            facts.append(_format_size(_dataset_size_bytes(filenames)))
        facts.append(f"{len(self._units)} recordings with traces")
        imgui.text_disabled(" · ".join(facts))
        if not ticked:
            return
        fs = float(ticked[0]["fs"])
        ref = float(self.settings.runtime.reference_fs or 0)
        if ref > 0 and not math.isclose(fs, ref, rel_tol=1e-6):
            imgui.text_disabled(
                f"Sample counts scale by {fs / ref:.3f} from {ref:.1f} Hz"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Start-up samples, wavelet scales and peak spacing are written for the "
                    "reference rate and scaled to this scan's."
                )
        comment = str(ticked[0].get("comment") or "")
        if comment:
            imgui.text_disabled(f"Comment: {comment}")

    def _default_outdir(self) -> str:
        """The folder a run of this format writes into: the file's own for a zarr run (the runner names the file), else the PF folder."""
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

    def _draw_output_block(self) -> None:
        """The output folder, the frame window and the channel read."""
        if self._outdir_dialog is not None and self._outdir_dialog.ready():
            result = self._outdir_dialog.result()
            if result:
                self._outdir = result
                set_last_dir("vnoiser_outdir", result)
            self._outdir_dialog = None
        zarr_out = self.settings.runtime.output_format == "zarr"
        imgui.spacing()
        imgui.separator_text("Output")
        max_frames, _, n_channels = self._dims()
        with settings_table(
            "##vnoiser_output", ("Folder", "Frames", "Channel")
        ) as shown:
            if not shown:
                return
            settings_row("Folder")
            btn_w = button_width("Browse")
            imgui.set_next_item_width(
                max(
                    imgui.get_content_region_avail().x
                    - btn_w
                    - imgui.get_style().item_spacing.x,
                    hello_imgui.em_size(6),
                )
            )
            _, self._outdir = imgui.input_text("##vnoiser_outdir", self._outdir)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"{self._outdir}\nThe results file lands here, named after the input and the time."
                    if zarr_out
                    else f"{self._outdir}\nThe PF folder the curation window opens."
                )
            imgui.same_line()
            if imgui.button("Browse##vnoiser_outdir_btn"):
                start = str(get_last_dir("vnoiser_outdir") or Path.home())
                self._outdir_dialog = pfd.select_folder("Select output folder", start)

            settings_row("Frames")
            try:
                window = self._frame_window()
                n_frames = max_frames if window is None else window[1] - window[0]
                text, color = f"{n_frames} of {max_frames}", None
            except ValueError as e:
                text, color = str(e), _MISSING_COLOR
            if color is not None:
                imgui.push_style_color(imgui.Col_.text, color)
            if imgui.button(f"{text}##vnoiser_slice"):
                self._show_slice_popup = True
            if color is not None:
                imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "The window of frames read: one contiguous stretch, since the baselines "
                    "and the z-score use the whole trace."
                )
            self._draw_slice_popup(max_frames)

            settings_row("Channel")
            imgui.set_next_item_width(hello_imgui.em_size(WIDTH_INPUT_EM))
            changed, index = imgui.combo(
                "##vnoiser_channel",
                min(self._channel_index, n_channels - 1),
                [f"ch {c + 1}" for c in range(n_channels)],
            )
            if changed:
                self._channel_index = index
            if imgui.is_item_hovered():
                imgui.set_tooltip("The channel the indicator is on.")

    def _draw_slice_popup(self, max_frames: int) -> None:
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
            if imgui.button("Close##vnoiser_slice_close"):
                imgui.close_current_popup()
            imgui.end_popup()

    def _draw_scans_block(self) -> None:
        """The recordings a run takes, ticked; their blocks sit behind the Blocks button."""
        imgui.spacing()
        imgui.separator_text("Scans")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The recordings of this file traces come from. Ticked ones are processed "
                "together with the same domains."
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
        if imgui.begin_table("##vnoiser_scans", 4, flags):
            kinds = [
                KIND_LABELS.get(kind, kind)
                for kind in (
                    str(u.get("modality_name") or u.get("kind") or "")
                    for u in self._units
                )
            ]
            fixed = imgui.TableColumnFlags_.width_fixed
            imgui.table_setup_column("Unit", imgui.TableColumnFlags_.width_stretch)
            ## NOTE: sized by hand, since the tab's text wrap breaks a word in a fit-to-content column
            imgui.table_setup_column(
                "Kind",
                fixed,
                max(imgui.calc_text_size(kind).x for kind in [*kinds, "Kind"]),
            )
            imgui.table_setup_column("ROIs", fixed)
            imgui.table_setup_column("Length", fixed)
            imgui.table_headers_row()
            for u, kind in zip(self._units, kinds, strict=True):
                key = u["key"]
                imgui.table_next_row()
                imgui.table_next_column()
                on_screen = key == shown
                if on_screen:
                    imgui.push_style_color(imgui.Col_.text, _SUBSECTION_COLOR)
                _, self._scans[key] = imgui.checkbox(
                    f"{key.rsplit('/', 1)[-1]}##vnoiser_scan_{key}",
                    self._scans.get(key, False),
                )
                if on_screen:
                    imgui.pop_style_color()
                if imgui.is_item_hovered():
                    comment = str(u.get("comment") or "")
                    imgui.set_tooltip(
                        "\n".join(
                            t
                            for t in (key, "on screen" if on_screen else "", comment)
                            if t
                        )
                    )
                imgui.table_next_column()
                imgui.text(kind)
                imgui.table_next_column()
                imgui.text(str(u["nrois"]))
                imgui.table_next_column()
                seconds = u.get("duration_s")
                imgui.text(f"{seconds:.0f} s" if seconds else "-")
            imgui.end_table()
        if imgui.small_button("On screen##vnoiser_scans_shown"):
            self._scans = {u["key"]: u["key"] == shown for u in self._units}
        imgui.same_line()
        if imgui.small_button("All##vnoiser_scans_all"):
            self._scans = {u["key"]: True for u in self._units}
        ticked = [u["key"] for u in self._units if self._scans.get(u["key"])]
        starts = self._block_args(ticked)["first_env"]
        imgui.same_line()
        if imgui.small_button(f"Blocks ({len(starts)})##vnoiser_blocks"):
            self._show_blocks_popup = True
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Mark the scans where a condition changed and what changed there."
            )

    def _draw_blocks_popup(self) -> None:
        """Which ticked scans start a block and what changed there, then the experiment's labels."""
        if self._show_blocks_popup:
            imgui.open_popup("##vnoiser_blocks_popup")
            self._show_blocks_popup = False
        imgui.set_next_window_size(
            imgui.ImVec2(hello_imgui.em_size(34), 0), imgui.Cond_.appearing
        )
        if not imgui.begin_popup("##vnoiser_blocks_popup"):
            return
        imgui.text_colored(_TITLE_COLOR, "Blocks")
        imgui.text_disabled(
            "A block starts at the first scan after a condition changed; a run's first scan always starts one."
        )
        ticked = [u["key"] for u in self._units if self._scans.get(u["key"])]
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
        if imgui.begin_table("##vnoiser_blocks", 3, flags):
            fixed = imgui.TableColumnFlags_.width_fixed
            imgui.table_setup_column("Scan", fixed)
            imgui.table_setup_column("New block", fixed)
            imgui.table_setup_column("Change", imgui.TableColumnFlags_.width_stretch)
            imgui.table_headers_row()
            for i, key in enumerate(ticked):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
                imgui.text(key.rsplit("/", 1)[-1])
                imgui.table_next_column()
                imgui.begin_disabled(i == 0)
                changed, start = imgui.checkbox(
                    f"##vnoiser_env_{key}", i == 0 or self._first_env.get(key, False)
                )
                imgui.end_disabled()
                if changed:
                    self._first_env[key] = start
                    if not start:
                        self._scan_labels.pop(key, None)
                imgui.table_next_column()
                if i == 0 or self._first_env.get(key):
                    self._draw_scan_labels(key)
            imgui.end_table()
        if not ticked:
            imgui.text_disabled("Tick a scan first.")
        imgui.spacing()
        self._draw_label_list()
        imgui.end_popup()

    def _draw_scan_labels(self, key: str) -> None:
        """One toggle per label; a picked label keeps New block ticked if this stops being the run's first scan."""
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x * 0.5
        picked = self._scan_labels.get(key, [])
        for i, label in enumerate(self._labels):
            if (
                i
                and imgui.get_item_rect_max().x + spacing + button_width(label) <= right
            ):
                imgui.same_line(0, spacing)
            with selected_button_style(label in picked):
                if imgui.small_button(f"{label}##vnoiser_label_{key}_{i}"):
                    self._scan_labels[key] = [
                        name
                        for name in self._labels
                        if (name in picked) != (name == label)
                    ]
                    self._first_env[key] = True

    def _draw_label_list(self) -> None:
        """The experiment's labels: x removes one from the list and its scans, Enter adds one."""
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x * 0.5
        imgui.align_text_to_frame_padding()
        imgui.text("Labels")
        set_tooltip(
            "this experiment's words for the Change column; Enter adds one",
            show_mark=False,
        )
        for i, label in enumerate(self._labels):
            text = f"{label} x"
            if imgui.get_item_rect_max().x + spacing + button_width(text) <= right:
                imgui.same_line(0, spacing)
            if imgui.small_button(f"{text}##vnoiser_label_rm_{i}"):
                self._labels.remove(label)
                self._scan_labels = {
                    k: kept
                    for k, picked in self._scan_labels.items()
                    if (kept := [name for name in picked if name != label])
                }
                break
            set_tooltip(f"Remove {label} from the list and its scans.", show_mark=False)
        field = hello_imgui.em_size(9)
        if imgui.get_item_rect_max().x + spacing + field <= right:
            imgui.same_line(0, spacing)
        imgui.set_next_item_width(field)
        entered, self._new_label = imgui.input_text_with_hint(
            "##vnoiser_new_label",
            "add a label",
            self._new_label,
            imgui.InputTextFlags_.enter_returns_true,
        )
        label = self._new_label.strip()
        if entered and label:
            if label not in self._labels:
                self._labels.append(label)
            self._new_label = ""

    def _comment_map(self) -> tuple[str, dict[str, list[int]]] | None:
        """``(unit, {name: rois})`` from the nearest comment at or before the first ticked scan; the rig writes a map once per layout, so the search stops at another ROI count."""
        from mbo_utilities.vnoiser.pipeline import comment_groups

        ticked = [i for i, u in enumerate(self._units) if self._scans.get(u["key"])]
        if not ticked:
            return None
        n_rois = int(self._units[ticked[0]]["nrois"])
        for u in reversed(self._units[: ticked[0] + 1]):
            if int(u["nrois"]) != n_rois:
                return None
            groups = comment_groups(u.get("comment"), n_rois)
            if groups:
                return u["key"].rsplit("/", 1)[-1], groups
        return None

    def drawn_domains(self) -> list[DomainRow]:
        """The ROI tool's ROIs on the recording on screen as one mask domain each, named by note or ``drawn<uid>``; [] without the tool or with another recording on screen."""
        from mbo_utilities.analysis.roi_traces import roi_geometry

        widget = self.parent.manual_roi
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

    def _set_blocks(self, first_env, labels, scan_labels) -> None:
        """Block ticks and labels by scan id, from a run or a ``domains.json``; a run without them takes the ticks alone."""
        scan_labels = scan_labels or {}
        used = [label for picked in scan_labels.values() for label in picked]
        given = SCAN_LABELS if labels is None else labels
        self._labels = list(dict.fromkeys([*given, *used]))
        self._scan_labels = {}
        for u in self._units:
            munit = u["key"].rsplit("_", 1)[-1]
            self._first_env[u["key"]] = munit in first_env
            if scan_labels.get(munit):
                self._scan_labels[u["key"]] = list(scan_labels[munit])

    def _block_args(self, scans: list[str]) -> dict:
        """``first_env``, ``labels`` and ``scan_labels`` of ``scans``, as a run and ``domains.json`` take them.

        A run's first scan always starts a block: analysis cuts blocks at this
        list, and scans before its first entry would belong to none.
        """
        starts = [k for i, k in enumerate(scans) if i == 0 or self._first_env.get(k)]
        return {
            "first_env": [k.rsplit("_", 1)[-1] for k in starts],
            "labels": list(self._labels),
            "scan_labels": {
                k.rsplit("_", 1)[-1]: list(self._scan_labels[k])
                for k in starts
                if self._scan_labels.get(k)
            },
        }

    def _read_book(self) -> DomainFile | None:
        """The file's domain file (:func:`find_domains`), or None when there is none or it does not read (the reason in ``_domain_error``)."""
        mesc = self._mesc_path()
        found = None if mesc is None else find_domains(mesc, get_roi_folder() or None)
        if found is None:
            return None
        try:
            book = DomainFile.read(found)
        except (OSError, ValueError, KeyError) as e:
            self._domain_error = f"{found.name}: {e}"
            return None
        self._domains_path = str(found)
        return book

    @staticmethod
    def _layout_rows(layout: Layout) -> list[DomainRow]:
        return [
            DomainRow.from_entry(n, v, layout.run is None or n in layout.run)
            for n, v in layout.domains.items()
        ]

    def _load_domains_file(self) -> None:
        """The table of the first ticked scan's layout in the domain file, its scans and every block tick."""
        mesc = self._mesc_path()
        found = find_domains(mesc, get_roi_folder() or None)
        if found is None:
            self._set_status(
                f"No {domains_path(mesc, get_roi_folder() or None).name} yet: Save writes one.",
                error=True,
            )
            return
        ticked = [u["key"] for u in self._units if self._scans.get(u["key"])]
        try:
            book = DomainFile.read(found)
            layout = (
                book.layout_of(ticked[0].rsplit("_", 1)[-1])
                if ticked
                else next(iter(book.layouts), None)
            )
            if layout is None:
                raise ValueError(f"no table for scan {ticked[0].rsplit('_', 1)[-1]}")
            rows = self._layout_rows(layout)
        except (OSError, ValueError, KeyError) as e:
            self._set_status(f"{found.name}: {e}", error=True)
            return
        self._domains, self._selected = rows, 0
        self._set_blocks(book.first_env, book.labels, book.scan_labels)
        if layout.scans:
            for u in self._units:
                self._scans[u["key"]] = u["key"].rsplit("_", 1)[-1] in layout.scans
        self._seed_slicing()
        self._domains_path = str(found)
        self._set_status(f"Loaded {found.name}")

    def _adopt_domain_file(self) -> None:
        """Take the first ticked scan's table from the domain file when another window rewrote it (the MESc tab's ROI names)."""
        mesc = self._mesc_path() if self._units else None
        found = None if mesc is None else find_domains(mesc, get_roi_folder() or None)
        if found is None:
            return
        stamp = found.stat().st_mtime_ns
        if self._book_seen is not None and self._book_seen[0] == stamp:
            return
        ticked = [u["key"] for u in self._units if self._scans.get(u["key"])]
        try:
            layout = ticked and DomainFile.read(found).layout_of(scan_of(ticked[0]))
            rows = self._layout_rows(layout) if layout else None
        except (OSError, ValueError, KeyError):
            return
        table = layout.to_json() if layout else None
        was, self._book_seen = self._book_seen, (stamp, table)
        ## NOTE: a trim or a block written by another window leaves the table alone, so unsaved edits stay
        if rows is None or was is None or was[1] == table:
            return
        self._domains, self._selected = rows, 0
        self._fit_domains(self._dims()[1])
        self._domains_path = str(found)
        self._set_status(f"Domains from {found.name}")

    def _save_domains_file(self) -> None:
        """Write the table and its run ticks as the ticked scans' layout in the domain file, with their block ticks and the label list; other layouts and scans keep theirs."""
        rows = self._check_domains()
        if not rows:
            return
        mesc = self._mesc_path()
        root = get_roi_folder() or None
        path = domains_path(mesc, root)
        found = find_domains(mesc, root)
        try:
            book = DomainFile() if found is None else DomainFile.read(found)
        except (OSError, ValueError, KeyError) as e:
            self._set_status(
                f"{found.name} does not read, not overwritten: {e}", error=True
            )
            return
        scans = [u["key"] for u in self._units if self._scans.get(u["key"])]
        numbers = [k.rsplit("_", 1)[-1] for k in scans]
        book.put(
            Layout(
                numbers,
                {r.name.strip(): r.entry() for r in rows},
                [r.name.strip() for r in rows if r.run],
            )
        )
        # only the ticks given: a layout's first scan starts a run's block, not the file's
        file_first = self._units[0]["key"] if self._units else None
        blocks = self._block_args(scans)
        book.set_blocks(
            numbers,
            [
                k.rsplit("_", 1)[-1]
                for k in scans
                if self._first_env.get(k) or k == file_first
            ],
            blocks["scan_labels"],
        )
        book.labels = blocks["labels"]
        try:
            book.write(path)
        except OSError as e:
            self._set_status(f"Cannot write {path}: {e}", error=True)
            return
        self._domains_path = str(path)
        self._set_status(f"Saved {path.name}")

    def _draw_domain_presets(self, n_rois: int) -> None:
        """Buttons that rebuild the table."""
        if imgui.button("One per ROI##vnoiser_dom_each"):
            self._domains = one_per_roi(n_rois)
            self._selected = 0
        if imgui.is_item_hovered():
            imgui.set_tooltip("Every ROI its own domain, every pixel.")
        imgui.same_line()
        if imgui.button("All in one##vnoiser_dom_all"):
            self._domains = [DomainRow("all", list(range(n_rois)))]
            self._selected = 0
        if imgui.is_item_hovered():
            imgui.set_tooltip("One domain of every ROI.")
        imgui.same_line()
        if imgui.button("Groups of##vnoiser_dom_groups"):
            self._domains = groups_of(n_rois, self._group_n)
            self._selected = 0
        if imgui.is_item_hovered():
            imgui.set_tooltip("Consecutive ROIs, this many to a domain.")
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.set_next_item_width(hello_imgui.em_size(5))
        _, self._group_n = imgui.input_int("##vnoiser_dom_group_n", self._group_n)
        self._group_n = max(1, self._group_n)
        mapped = self._comment_map()
        if mapped:
            unit, groups = mapped
            imgui.same_line()
            if imgui.button("From comment##vnoiser_dom_comment"):
                # the pipeline gives bg no trace of its own; it stays a background
                self._domains = [
                    DomainRow(name, rois, run=name != "bg")
                    for name, rois in groups.items()
                ]
                self._selected = 0
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"The groups in {unit}'s comment: "
                    + ", ".join(f"{name} {rois}" for name, rois in groups.items())
                )
        drawn = self.drawn_domains()
        imgui.same_line()
        imgui.begin_disabled(not drawn)
        if imgui.button(f"Drawn ROIs ({len(drawn)})##vnoiser_dom_drawn"):
            have = {r.name for r in self._domains}
            for row in drawn:
                if row.name in have:
                    self._domains = [
                        row if r.name == row.name else r for r in self._domains
                    ]
                else:
                    self._domains.append(row)
            self._selected = len(self._domains) - 1
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "Add every ROI drawn in the ROI tool on the recording on screen as a domain "
                "of exactly its pixels, named by its note."
            )

    def _draw_domains_block(self) -> None:
        """The table as it stands, one line per domain with its run tick; Edit opens the editor."""
        imgui.spacing()
        imgui.separator_text("Domains")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "A domain is one trace: the pixels of one or more ROIs averaged together "
                "(the lines of a soma, a patch, the part of a patch the cell covers)."
            )
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit
        if imgui.begin_table("##vnoiser_domains_summary", 3, flags):
            imgui.table_setup_column("##run", imgui.TableColumnFlags_.width_fixed)
            ## NOTE: sized by hand, since the tab's text wrap breaks a word in a fit-to-content column
            imgui.table_setup_column(
                "##name",
                imgui.TableColumnFlags_.width_fixed,
                max(
                    (imgui.calc_text_size(r.name).x for r in self._domains), default=0.0
                ),
            )
            imgui.table_setup_column("##rois", imgui.TableColumnFlags_.width_stretch)
            for i, row in enumerate(self._domains):
                imgui.table_next_row()
                imgui.table_next_column()
                _, row.run = imgui.checkbox(f"##vnoiser_dom_sum_run_{i}", row.run)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Whether this run takes the domain.")
                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
                if row.run:
                    imgui.text(row.name)
                else:
                    imgui.text_disabled(row.name)
                imgui.table_next_column()
                rule = "" if row.pixels == "all" else f", {PIXEL_LABELS[row.pixels]}"
                imgui.align_text_to_frame_padding()
                imgui.text_disabled(f"ROI {roi_ranges(row.rois)}{rule}")
            imgui.end_table()
        if imgui.button(f"{fa.ICON_FA_PEN}  Edit domains##vnoiser_dom_edit"):
            self._show_domains_popup = True
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Name the domains, pick their ROIs and pixels, and preview each one."
            )
        self._draw_domains_status()

    def _draw_domains_status(self) -> None:
        """How many domains run and how many ROIs they read, or what is wrong with the table."""
        self._domains_to_run()
        if self._domain_error:
            imgui.text_colored(_MISSING_COLOR, self._domain_error)
            return
        ticked = [r for r in self._domains if r.run]
        imgui.text_disabled(
            f"{len(ticked)} of {len(self._domains)} domains run, "
            f"{len({roi for r in ticked for roi in r.rois})} ROIs read"
        )

    def _draw_domains_popup(self) -> None:
        """The domain editor: presets, one row per domain with an ROI's box per column, the preview of the picked row, then Load, Save and Close."""
        if self._show_domains_popup:
            imgui.open_popup(POPUP_DOMAINS)
            self._show_domains_popup = False
        viewport = imgui.get_main_viewport()
        ## NOTE: centered every frame, since an auto-resized popup is 0 wide on the frame it appears
        imgui.set_next_window_pos(
            viewport.get_center(), imgui.Cond_.always, imgui.ImVec2(0.5, 0.5)
        )
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(hello_imgui.em_size(44), 0),
            imgui.ImVec2(viewport.size.x * 0.96, viewport.size.y * 0.94),
        )
        flags = (
            imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_move
        )
        opened, visible = imgui.begin_popup_modal(POPUP_DOMAINS, True, flags)
        if not opened:
            return
        if not visible or (
            imgui.is_key_pressed(imgui.Key.escape) and not imgui.is_any_item_active()
        ):
            imgui.close_current_popup()
        _, n_rois, _ = self._dims()
        scans = [
            u["key"].rsplit("/", 1)[-1]
            for u in self._units
            if self._scans.get(u["key"])
        ]
        imgui.text_colored(_TITLE_COLOR, "Domains")
        imgui.same_line()
        imgui.text_disabled(
            f"of {', '.join(scans) or 'no scan'}: each row is one trace, the pixels of its ticked ROIs averaged"
        )
        counts = {int(u["nrois"]) for u in self._units if self._scans.get(u["key"])}
        if len(counts) > 1:
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
            imgui.text_wrapped(
                f"The ticked scans have {min(counts)} to {max(counts)} ROIs. This table "
                f"applies to all of them, up to ROI {n_rois - 1}, and an ROI number may "
                "not be the same line in each. Run each layout on its own."
            )
            imgui.pop_style_color()
        imgui.spacing()
        imgui.separator_text("Start from")
        self._draw_domain_presets(n_rois)
        imgui.spacing()
        imgui.separator_text("Table")
        self._draw_domain_table(n_rois)
        if imgui.button(f"{fa.ICON_FA_PLUS}  Add domain##vnoiser_dom_add"):
            self._domains.append(DomainRow(f"domain{len(self._domains) + 1}", [0]))
            self._selected = len(self._domains) - 1
        imgui.same_line()
        imgui.align_text_to_frame_padding()
        self._draw_domains_status()
        imgui.spacing()
        imgui.separator_text("Preview")
        self._draw_preview()
        draw_divider()
        if imgui.button("Load##vnoiser_dom_load"):
            self._load_domains_file()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Load the ticked scan's table from the domain file.")
        imgui.same_line()
        if imgui.button("Save##vnoiser_dom_save"):
            self._save_domains_file()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Save the table, the ticked scans and their blocks to the domain file; a run saves it too."
            )
        if self._domains_path:
            imgui.same_line()
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(Path(self._domains_path).name)
            if imgui.is_item_hovered():
                imgui.set_tooltip(self._domains_path)
        imgui.same_line()
        close_w = button_width("Close")
        imgui.set_cursor_pos_x(
            max(
                imgui.get_cursor_pos_x(),
                imgui.get_window_width() - imgui.get_style().window_padding.x - close_w,
            )
        )
        with button_colors(*COLORS_CLOSE):
            if imgui.button("Close##vnoiser_dom_close"):
                imgui.close_current_popup()
        imgui.end_popup()

    def _draw_domain_table(self, n_rois: int) -> None:
        """One row per domain: the row the preview shows, its run tick, its name, a box per ROI (a typed list past ``MATRIX_MAX`` ROIs), the pixels it averages and their share."""
        matrix = 1 < n_rois <= MATRIX_MAX
        roi_columns = n_rois if matrix else 1
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner
            | imgui.TableFlags_.borders_outer
            | imgui.TableFlags_.sizing_fixed_fit
        )
        remove = None
        if not imgui.begin_table("##vnoiser_domains", 6 + roi_columns, flags):
            return
        fixed = imgui.TableColumnFlags_.width_fixed
        imgui.table_setup_column("View", fixed)
        imgui.table_setup_column("Run", fixed)
        imgui.table_setup_column("Name", fixed, hello_imgui.em_size(9))
        if matrix:
            for roi in range(n_rois):
                imgui.table_setup_column(str(roi), fixed)
        else:
            imgui.table_setup_column("ROIs", fixed, hello_imgui.em_size(14))
        imgui.table_setup_column("Pixels", fixed, hello_imgui.em_size(10))
        imgui.table_setup_column("Share", fixed, hello_imgui.em_size(6))
        imgui.table_setup_column("##remove", fixed)
        imgui.table_next_row(imgui.TableRowFlags_.headers)
        for column in range(6 + roi_columns):
            imgui.table_set_column_index(column)
            name = imgui.table_get_column_name(column)
            imgui.table_header(name)
            if imgui.is_item_hovered() and name in DOMAIN_COLUMN_HELP:
                imgui.set_tooltip(DOMAIN_COLUMN_HELP[name])
            elif imgui.is_item_hovered() and matrix and name.isdigit():
                imgui.set_tooltip(
                    f"ROI {name}: tick it in a row to add it to that domain."
                )
        for i, row in enumerate(self._domains):
            imgui.table_next_row()
            imgui.table_next_column()
            if imgui.radio_button(f"##vnoiser_dom_pick_{i}", self._selected == i):
                self._selected = i
            imgui.table_next_column()
            _, row.run = imgui.checkbox(f"##vnoiser_dom_run_{i}", row.run)
            imgui.table_next_column()
            imgui.set_next_item_width(-1)
            _, row.name = imgui.input_text(f"##vnoiser_dom_name_{i}", row.name)
            bad = any(not 0 <= r < n_rois for r in row.rois)
            if matrix:
                for roi in range(n_rois):
                    imgui.table_next_column()
                    # a drawn mask belongs to the ROI it was drawn on
                    imgui.begin_disabled(row.pixels == "mask")
                    changed, on = imgui.checkbox(
                        f"##vnoiser_dom_{i}_roi_{roi}", roi in row.rois
                    )
                    imgui.end_disabled()
                    if changed and (on or len(row.rois) > 1):
                        row.rois = sorted(set(row.rois) ^ {roi})
            else:
                imgui.table_next_column()
                if bad:
                    imgui.push_style_color(imgui.Col_.text, _MISSING_COLOR)
                if n_rois == 1:
                    imgui.text_disabled("0 (the frame)")
                else:
                    imgui.set_next_item_width(-1)
                    imgui.begin_disabled(row.pixels == "mask")
                    changed, text = imgui.input_text(
                        f"##vnoiser_dom_rois_{i}", ",".join(str(r) for r in row.rois)
                    )
                    imgui.end_disabled()
                    if changed:
                        try:
                            row.rois = parse_roi_text(text, n_rois)
                        except ValueError:
                            pass
                if bad:
                    imgui.pop_style_color()
            imgui.table_next_column()
            rules = [p for p in PIXEL_RULES if p != "mask" or row.mask]
            imgui.set_next_item_width(-1)
            changed, current = imgui.combo(
                f"##vnoiser_dom_px_{i}",
                rules.index(row.pixels) if row.pixels in rules else 0,
                [PIXEL_LABELS[p] for p in rules],
            )
            if changed:
                row.pixels = rules[current]
            imgui.table_next_column()
            if row.pixels in ("bright", "active", "correlated"):
                imgui.set_next_item_width(-1)
                _, percent = imgui.slider_float(
                    f"##vnoiser_dom_frac_{i}", row.fraction * 100, 1.0, 100.0, "%.0f %%"
                )
                row.fraction = min(1.0, max(0.01, percent / 100))
            else:
                imgui.align_text_to_frame_padding()
                imgui.text_disabled("all" if row.pixels == "all" else "mask")
            imgui.table_next_column()
            if imgui.small_button(f"{fa.ICON_FA_XMARK}##vnoiser_dom_rm_{i}"):
                remove = i
            if imgui.is_item_hovered():
                imgui.set_tooltip("Remove this domain.")
        imgui.end_table()
        if remove is not None:
            del self._domains[remove]
            self._selected = min(self._selected, max(len(self._domains) - 1, 0))

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
        """The picked domain on the scan on screen: its pixels over an ROI's mean image, then F, dF/F and z."""
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
        imgui.set_next_item_width(hello_imgui.em_size(9))
        _, self._selected = imgui.combo(
            "##vnoiser_preview_domain",
            min(self._selected, len(self._domains) - 1),
            [r.name for r in self._domains],
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("The domain the preview reads; its ROIs are averaged.")
        imgui.same_line()
        imgui.text(f"on {unit['key'].rsplit('/', 1)[-1]}, first")
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
        result = self._preview.result
        imgui.same_line()
        if self._preview.busy:
            imgui.text_disabled("reading...")
        elif result is not None and not self._preview.error:
            pixels = sum(len(v) for v in result["mask"].values())
            total = sum(h * w for h, w in result["shape"].values())
            imgui.text_disabled(
                f"{pixels} of {total} pixels · dF/F {np.nanpercentile(result['dfof'], 1) * 100:.1f} to "
                f"{np.nanpercentile(result['dfof'], 99.9) * 100:.1f} % · z peak {np.nanmax(result['z']):.1f}"
            )
        else:
            imgui.text_disabled("")
        if self._preview.error:
            imgui.text_colored(_MISSING_COLOR, self._preview.error)
            return
        if result is None:
            return
        self._draw_preview_image(result)
        imgui.same_line()
        self._draw_preview_traces(result)
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
                [f"{result['name']}: ROI {r}" for r in rois],
            )
            if changed:
                self._preview_roi = index
        else:
            imgui.text(f"{result['name']}: ROI {roi}")
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
                f"ROI {roi}'s mean image ({h} x {w}); the red pixels are {result['name']}'s. "
                "The traces average every ROI of the domain."
            )

    def _draw_preview_traces(self, result: dict) -> None:
        """F (with pixel mean, background and F0 when present), dF/F and z stacked on one time axis, refitted when a new result lands."""
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

    def _value(self, group: str, name: str):
        if (group, name) == POLARITY:
            return self.settings.polarity
        return getattr(getattr(self.settings, group), name)

    def _set_value(self, group: str, name: str, value) -> None:
        if (group, name) == POLARITY:
            self.settings.polarity = value
        else:
            setattr(getattr(self.settings, group), name, value)

    def _default_of(self, group: str, name: str):
        """The archive's value of a field."""
        if (group, name) == POLARITY:
            return POLARITY_DEFAULT
        return _field_default(getattr(self.settings, group), name)

    def _modified(self, group: str, name: str) -> bool:
        if (group, name) == POLARITY:
            return self.settings.polarity != POLARITY_DEFAULT
        return not _is_default(getattr(self.settings, group), name)

    def _text_of(self, group: str, name: str, value) -> str:
        """A field's value as the popup writes it beside ``default``."""
        hint = HINTS[group][name]
        if hint.choices is not None and value in hint.choices:
            return str(hint.choices[value])
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, (tuple, list)):
            return ", ".join(f"{float(v):g}" for v in value)
        if isinstance(value, float):
            fs = float(self.settings.runtime.reference_fs or 1.0)
            return (hint.fmt % (value / fs if hint.seconds else value)).strip()
        return str(value)

    def _fields_of(self, sections, tiers: tuple) -> list[tuple[str, str]]:
        """The ``(group, name)`` of a column's fields of these tiers that apply now, in order."""
        return [
            (group, name)
            for _title, group, names in sections
            for name in names
            if HINTS[group][name].tier in tiers and applies(self.settings, group, name)
        ]

    def _draw_parameters_block(self) -> None:
        """The Parameters button, then every value that differs from the archive's."""
        imgui.spacing()
        imgui.separator_text("Parameters")
        if imgui.button(f"{fa.ICON_FA_SLIDERS}  Parameters##vnoiser_settings"):
            self._show_settings_popup = True
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The trace, dF/F, wavelet denoiser and peak settings, with a preview."
            )
        imgui.same_line()
        self._draw_defaults()
        self._draw_modified_table()

    def _draw_defaults(self) -> None:
        """An orange Defaults button back to the archive's settings, disabled while nothing differs, then Recommended."""
        from mbo_utilities.vnoiser.params import VnoiserSettings

        changed = len(self._modified_rows()) > 0
        with button_colors(*COLORS_DEFAULTS):
            imgui.begin_disabled(not changed)
            if imgui.button(
                f"{fa.ICON_FA_ARROW_ROTATE_LEFT}  Defaults##vnoiser_defaults"
            ):
                self.settings = VnoiserSettings()
                self._outdir = self._default_outdir()
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "Reset every parameter to the archive's (Noguchi & Terada): raw counts, "
                "Gaussian dF/F, SD z-score, z flipped."
                if changed
                else "Every value is the archive's."
            )
        imgui.same_line()
        if imgui.button("Recommended##vnoiser_recommended"):
            self.settings = VnoiserSettings.recommended()
            self._outdir = self._default_outdir()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "For new recordings: the archive's settings on photon counts, so dF/F has "
                "its real size, with a robust z-score."
            )

    def _draw_parameters(self) -> None:
        """The Parameters popup: a column per stage, the preview and the runtime under them, then Defaults and Close."""
        if self._show_settings_popup:
            imgui.open_popup(POPUP_PARAMETERS)
            self._show_settings_popup = False
        em = imgui.get_font_size()
        style = imgui.get_style()
        viewport = imgui.get_main_viewport()
        room = (
            viewport.size.x * 0.96
            - 2 * style.window_padding.x
            - (len(COLUMNS) - 1) * style.item_spacing.x
        )
        width_column = min(
            max(room / len(COLUMNS), WIDTH_COLUMN_EM[0] * em), WIDTH_COLUMN_EM[1] * em
        )
        height_column = viewport.size.y * 0.94 - self._parameters_chrome
        ## NOTE: centered every frame, since an auto-resized popup is 0 wide on the frame it appears
        imgui.set_next_window_pos(
            viewport.get_center(), imgui.Cond_.always, imgui.ImVec2(0.5, 0.5)
        )
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(0, 0),
            imgui.ImVec2(viewport.size.x * 0.96, viewport.size.y * 0.94),
        )
        flags = (
            imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_move
        )
        opened, visible = imgui.begin_popup_modal(POPUP_PARAMETERS, True, flags)
        if not opened:
            return
        if not visible or (
            imgui.is_key_pressed(imgui.Key.escape) and not imgui.is_any_item_active()
        ):
            imgui.close_current_popup()
        rt = self.settings.runtime
        tr = self.settings.traces
        if self._dims()[2] < 2:
            tr.reference = "none"
        if tr.reference == "none":
            tr.reference_channel = None
        y_columns = imgui.get_cursor_pos_y()
        for i, (stage, sections) in enumerate(COLUMNS):
            if i > 0:
                imgui.same_line()
            self._draw_column(stage, sections, width_column, height_column)
        height_columns = imgui.get_cursor_pos_y() - y_columns
        draw_divider()
        imgui.set_next_item_open(self._preview_open, imgui.Cond_.once)
        self._preview_open = imgui.collapsing_header("Preview##vnoiser_preview")
        if self._preview_open:
            self._draw_preview()
        draw_divider()
        folder, fmt = self._default_outdir(), rt.output_format
        imgui.text_colored(_TITLE_COLOR, "Runtime")
        if imgui.begin_table(
            "##vnoiser_runtime", len(COLUMNS), imgui.TableFlags_.sizing_stretch_same
        ):
            for name in RUNTIME:
                if applies(self.settings, "runtime", name):
                    imgui.table_next_column()
                    self._draw_param("runtime", name)
            imgui.end_table()
        # an untouched folder follows the format; a chosen one stays
        if rt.output_format != fmt and self._outdir in ("", folder):
            self._outdir = self._default_outdir()
        draw_divider()
        self._draw_defaults()
        imgui.same_line()
        imgui.set_cursor_pos_x(
            imgui.get_window_width() - style.window_padding.x - button_width("Close")
        )
        with button_colors(*COLORS_CLOSE):
            if imgui.button("Close##vnoiser_settings_close"):
                imgui.close_current_popup()
        self._parameters_chrome = (
            imgui.get_cursor_pos_y()
            + style.window_padding.y
            + style.popup_border_size
            - height_columns
        )
        imgui.end_popup()

    def _draw_column(
        self, stage: str, sections, width: float, height_max: float
    ) -> None:
        """One stage's column: its name, each section's fields, then folds holding its advanced and its stable fields; it scrolls once taller than ``height_max``."""
        flags = (
            imgui.ChildFlags_.borders
            | imgui.ChildFlags_.auto_resize_y
            | imgui.ChildFlags_.always_use_window_padding
        )
        padding = imgui.get_style().window_padding
        rt, ev = self.settings.runtime, self.settings.events
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(width, 0), imgui.ImVec2(width, height_max)
        )
        ## NOTE: whole-pixel padding, since imgui truncates an auto-fit height and a scaled padding such as 8.4 px
        ## leaves every column 0.8 px short of its content and so scrolling
        with (
            imgui_ctx.push_style_var(
                imgui.StyleVar_.window_padding,
                imgui.ImVec2(round(padding.x), round(padding.y)),
            ),
            imgui_ctx.begin_child(
                f"##vnoiser_column_{stage}", imgui.ImVec2(width, 0), flags
            ),
        ):
            imgui.text_colored(_TITLE_COLOR, stage)
            if stage == STAGE_DENOISER:
                self._draw_denoise_toggle()
            imgui.begin_disabled(
                stage in (STAGE_DENOISER, STAGE_PEAKS) and not rt.denoise
            )
            for title, group, names in sections:
                fields = self._fields_of(
                    ((title, group, names),), ("important", "normal")
                )
                if not fields:
                    continue
                imgui.spacing()
                imgui.separator_text(title)
                for _group, name in fields:
                    imgui.begin_disabled(
                        group == "events" and name != "detect" and not ev.detect
                    )
                    self._draw_field(group, name)
                    imgui.end_disabled()
            for tier, title in (("advanced", "Advanced"), ("stable", "Stable")):
                folded = self._fields_of(sections, (tier,))
                if not folded:
                    continue
                changed = sum(self._modified(g, n) for g, n in folded)
                color = (
                    _MODIFIED_COLOR
                    if changed
                    else _DIM_COLOR
                    if tier == "stable"
                    else None
                )
                imgui.spacing()
                if color is not None:
                    imgui.push_style_color(imgui.Col_.text, color)
                opened = imgui.tree_node(
                    f"{title}{f' ({changed} changed)' if changed else ''}###vnoiser_{tier}_{stage}"
                )
                if color is not None:
                    imgui.pop_style_color()
                if tier == "stable" and imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Choices the archive made; change one only to depart from it on purpose."
                    )
                if opened:
                    for group, name in folded:
                        self._draw_field(group, name)
                    imgui.tree_pop()
            imgui.end_disabled()

    def _draw_denoise_toggle(self) -> None:
        """Whether the run goes past dF/F and the z-score; off greys the denoiser and the peaks."""
        rt = self.settings.runtime
        pushed = not rt.denoise
        if pushed:
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
        changed, rt.denoise = imgui.checkbox("denoise and find peaks", rt.denoise)
        if pushed:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Off stops after dF/F and the z-score: no wavelet denoising, no peaks. "
                "The run then writes the results zarr."
            )
        if changed and not rt.denoise:
            rt.output_format = "zarr"

    def _draw_field(self, group: str, name: str) -> None:
        """A column's field, with the choices only the tab knows: the table's domains, the other channels."""
        tr = self.settings.traces
        _, _, n_channels = self._dims()
        if (group, name) == ("traces", "reference") and n_channels < 2:
            imgui.text_disabled("one channel")
        elif (group, name) == ("traces", "background_domain"):
            names = [r.name for r in self._domains] or [tr.background_domain]
            self._draw_param(group, name, {n: n for n in names})
        elif (group, name) == ("traces", "reference_channel"):
            others = [c for c in range(n_channels) if c != self._channel_index]
            self._draw_param(group, name, {c: f"ch {c + 1}" for c in others})
        else:
            self._draw_param(group, name)
        if (group, name) == POLARITY and self.parent.invert_deflection:
            imgui.text_disabled(
                f"{fa.ICON_FA_CIRCLE_INFO} the viewer's Invert Deflection is display only"
            )

    def _draw_param(self, group: str, name: str, choices: dict | None = None) -> None:
        """One field: a widget chosen by the value it holds, its label to the right (boxed when important), and once it differs from the archive's value an orange tint, a reset button and the default beside it."""
        hint = HINTS[group][name]
        choices = hint.choices if choices is None else choices
        value, default = self._value(group, name), self._default_of(group, name)
        modified = self._modified(group, name)
        style = imgui.get_style()
        width = hello_imgui.em_size(WIDTH_INPUT_EM)
        path = f"##vnoiser_{group}_{name}"
        fs = float(self.settings.runtime.reference_fs or 1.0) if hint.seconds else 1.0
        if modified:
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
        if choices is not None:
            keys = list(choices)
            labels = [str(choices[k]) for k in keys]
            widest = max(imgui.calc_text_size(text).x for text in labels)
            imgui.set_next_item_width(
                max(
                    width, widest + 2 * style.frame_padding.x + imgui.get_frame_height()
                )
            )
            changed, index = imgui.combo(
                path, keys.index(value) if value in keys else 0, labels
            )
            new = keys[index]
            changed = changed or value not in keys
        elif isinstance(value, bool):
            changed, new = imgui.checkbox(path, value)
        elif isinstance(value, int):
            imgui.set_next_item_width(width)
            changed, new = imgui.input_int(path, value, 0, 0)
        elif isinstance(value, (tuple, list)):
            imgui.set_next_item_width(width * len(value) * 0.55)
            changed, levels = imgui.input_float4(
                path, [float(v) for v in value], hint.fmt
            )
            new = tuple(float(v) for v in levels)
        else:
            imgui.set_next_item_width(width)
            changed, seconds = imgui.input_float(
                path, float(value) / fs, 0.0, 0.0, hint.fmt
            )
            new = seconds * fs
        if modified:
            imgui.pop_style_color()
        if hint.lo is not None and not isinstance(new, (bool, tuple, str)):
            new = max(type(new)(hint.lo * fs), new)
        if changed:
            self._set_value(group, name, new)
        if hint.tier == "important":
            imgui.same_line(0, style.item_inner_spacing.x + 0.3 * imgui.get_font_size())
        else:
            imgui.same_line(0, style.item_inner_spacing.x)
        if modified:
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
        if hint.tier == "important":
            draw_boxed_label(hint.label, font=self.parent._bold_font)
        else:
            imgui.text(hint.label)
        if modified:
            imgui.pop_style_color()
        text_default = self._text_of(group, name, default)
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 32.0)
            if hint.tooltip:
                imgui.text_unformatted(hint.tooltip)
            imgui.text_disabled(f"{group}.{name}, archive default {text_default}")
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        resettable = choices is None or default in choices
        if modified and resettable and self._draw_reset(path, text_default):
            self._set_value(group, name, default)

    def _draw_reset(self, path: str, text_default: str) -> bool:
        """A reset button after a changed value, then its default in dim text; True when the button was clicked."""
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        clicked = imgui.small_button(f"{fa.ICON_FA_ARROW_ROTATE_LEFT}##reset_{path}")
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Reset to {text_default}")
        same_line_if_fits(imgui.calc_text_size(f"default {text_default}").x)
        imgui.text_disabled(f"default {text_default}")
        return clicked

    def _modified_rows(self) -> list[tuple[str, str, str]]:
        """``(field, current, default)`` of every setting that differs from the archive's, as the popup words it."""
        rows = []
        if self._modified(*POLARITY):
            rows.append(
                (
                    "traces.polarity",
                    self._text_of(*POLARITY, self.settings.polarity),
                    self._text_of(*POLARITY, POLARITY_DEFAULT),
                )
            )
        for group in ("traces", "dfof", "denoiser", "events", "runtime"):
            obj = getattr(self.settings, group)
            for f in dataclasses.fields(obj):
                if (group, f.name) in HIDDEN and group != "runtime":
                    continue
                if _is_default(obj, f.name):
                    continue
                current, default = getattr(obj, f.name), _field_default(obj, f.name)
                if f.name in HINTS[group]:
                    current = self._text_of(group, f.name, current)
                    default = self._text_of(group, f.name, default)
                rows.append((f"{group}.{f.name}", str(current), str(default)))
        return rows

    def _draw_modified_table(self) -> None:
        """Every changed value beside its default."""
        rows = self._modified_rows()
        if not rows:
            imgui.text_disabled("All parameters at the archive's defaults")
            return
        imgui.text(f"Modified parameters ({len(rows)})")
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.borders_outer
            | imgui.TableFlags_.sizing_stretch_prop
        )
        if not imgui.begin_table("##vnoiser_mod_tbl", 3, flags):
            return
        for heading, weight in (("Parameter", 4.0), ("Current", 2.5), ("Default", 2.5)):
            imgui.table_setup_column(
                heading, imgui.TableColumnFlags_.width_stretch, weight
            )
        imgui.table_headers_row()
        for name, current, default in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_colored(_MODIFIED_COLOR, name)
            imgui.table_next_column()
            imgui.text_wrapped(current)
            imgui.table_next_column()
            imgui.push_style_color(imgui.Col_.text, _DIM_COLOR)
            imgui.text_wrapped(default)
            imgui.pop_style_color()
        imgui.end_table()

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
        # curation opens the run once written, else the file itself with every line raw
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

        # the table a run used is the one the dashboard and curation window show
        self._save_domains_file()
        args = {
            "input_path": str(mesc),
            "reader_kwargs": widget_reader_kwargs(self.parent.image_widget),
            "output_dir": self._outdir,
            "units": scans,
            **self._block_args(scans),
            "domains": domains,
            "channel": self._channel(),
            "frames": None if window is None else [window[0], window[1]],
            "invert_deflection": bool(self.parent.invert_deflection),
            "settings": self.settings.to_dict(),
            "custom_metadata": dict(self.parent._custom_metadata),
        }
        description = f"vnoiser: {len(scans)} scan(s), {len(domains)} domain(s)"
        pid = get_process_manager().spawn(
            task_type="vnoiser",
            args=args,
            description=description,
            output_path=self._outdir,
        )
        if pid:
            self._spawned.add(int(pid))
            self._set_status(f"Started (PID {pid}); progress in the process console.")
        else:
            self._set_status("Failed to start the worker.", error=True)

    def poll(self) -> None:
        """Once a second, whatever tab is showing: a finished worker this tab started lands in the Traces tab (:meth:`_load_traces`), a failed one in the status row."""
        if time.monotonic() - self._polled < 1.0:
            return
        self._polled = time.monotonic()
        self._adopt_domain_file()
        if not self._spawned:
            return
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        for info in get_process_manager().get_running():
            if info.pid not in self._spawned or info.status == "running":
                continue
            self._spawned.discard(info.pid)
            if info.status != "completed":
                self._set_status(
                    f"vnoiser failed: {info.status_message or info.status}", error=True
                )
                continue
            out = (info.args or {}).get("output_dir") or self._outdir
            results = newest_results(out, "vnoiser", source=self._mesc_path())
            if results is None:
                self._set_status(
                    f"vnoiser finished but left no results file in {out}", error=True
                )
                continue
            self._load_traces(results)

    def _load_traces(self, results: Path) -> None:
        """Switch Manual ROI Labeling on and load a results file into its Traces tab."""
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        set_widget_enabled("manual_roi", True)
        self.parent.sync_manual_roi(True)
        widget = self.parent.manual_roi
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
        """Open the spike-triggered average on the results unit that processed the recording on screen, at the ROI whose lines hold the line on the ROI slider."""
        from mbo_utilities.gui.launch import LaunchedWindow
        from mbo_utilities.gui.spike_average_viewer import launch_spike_average
        from mbo_utilities.results import Results, unit_for_source

        found = Results.read(results)
        iw = self.parent.image_widget
        shown = base_array(iw.data[0]) if iw is not None else None
        unit = unit_for_source(found, str(getattr(shown, "unit_key", "") or ""))
        roi = None
        view = self.parent.slice
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
