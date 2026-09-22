"""The ``PF`` folder the voltage pipeline writes, opened as an array.

``imread`` returns a :class:`PfArray` for a ``PF`` folder, its traces pickle
or the experiment folder holding it, the way a suite2p output folder opens
as a ``Suite2pArray``. The image is the line scan the folder was made from
when that ``.mesc`` is reachable (named in ``pipeline.json``, or laid out the
archive's way beside the folder), else a raster of the denoised traces: one
row per domain, one column per ``raster_bin`` samples. The traces, domains,
events and settings are attributes either way.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import ReductionMixin, _imwrite_base
from mbo_utilities.lazy_array import LazyArray
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.pf")

TRACES_FILE = "denoised_trace_scans.pkl"
FS_FILE = "fs_scans.pkl"
ROIS_FILE = "scanIDs_ROIs.pkl"
PEAKS_FILE = "detected_events_peaks.pkl"
PARAMS_FILE = "param_spike_detect.pkl"
PROVENANCE_FILE = "pipeline.json"
EXCLUDED_DOMAINS = ("All_domains", "bg")
# the archive renamed one domain between the ROI table and the final traces
FINAL_DOMAIN_NAMES = {"soma1": "soma"}
RASTER_WIDTH = 4096

_PF_INFO = PipelineInfo(
    name="voltage",
    description="Spatial JEDI voltage pipeline: AOD ROI traces, dF/F, wavelet denoising, peaks (a PF folder)",
    input_patterns=["**/*.mesc"],
    output_patterns=["**/*.voltage.zarr", f"**/PF/{TRACES_FILE}", f"**/PF/{PROVENANCE_FILE}", "**/PF/test.h5"],
    input_extensions=["mesc"],
    output_extensions=["pkl", "h5", "json", "zarr"],
    marker_files=[TRACES_FILE],
    category="processor",
)
register_pipeline(_PF_INFO)


def pf_results_in(folder) -> Path | None:
    """The newest voltage results zarr directly in ``folder``, or None."""
    from mbo_utilities.results import newest_results

    return newest_results(folder, "voltage")


def pf_files(pf_dir) -> Path:
    """Where a run's own files (``pipeline.json``, ``timings.json``, its h5 and
    ``traces/``) sit: ``_sidecar`` inside a results zarr, the folder itself for
    a ``PF`` folder of pickles."""
    from mbo_utilities.results import SIDECAR

    pf_dir = Path(pf_dir)
    return pf_dir / SIDECAR if pf_dir.suffix == ".zarr" else pf_dir


def pf_dir_of(path) -> Path | None:
    """What a voltage run left, from anything naming it: a results zarr
    itself, a ``PF`` folder of pickles, or a folder holding either (its own
    ``*.zarr``, or a ``PF`` beside it). None for anything else."""
    from mbo_utilities.results import results_pipeline

    p = Path(path)
    if p.is_file():
        return p.parent if p.name == TRACES_FILE else None
    if not p.is_dir():
        return None
    if p.suffix == ".zarr":
        return p if results_pipeline(p) == "voltage" else None
    if (p / TRACES_FILE).is_file():
        return p
    if (p / "PF" / TRACES_FILE).is_file():
        return p / "PF"
    found = pf_results_in(p) or pf_results_in(p / "PF")
    return found


def pf_source(pf_dir, provenance: dict | None = None) -> tuple[Path | None, dict[str, str]]:
    """The line scan a ``PF`` folder was made from and its ``{scan id: unit
    key}``: the pipeline's provenance (``pipeline.json`` unless given) names
    both; without it the ``.mesc`` laid out the archive's way beside the
    folder is the source and the units are ``MUnit_<scan>``. ``(None, {})``
    when no source is reachable."""
    pf_dir = Path(pf_dir)
    if provenance is None:
        provenance = {}
        prov_file = pf_files(pf_dir) / PROVENANCE_FILE
        if prov_file.is_file():
            provenance = json.loads(prov_file.read_text())
    block = provenance.get("source") or {}
    units = {str(s): str(u) for s, u in (block.get("units") or {}).items()}
    named = block.get("mesc")
    if named and Path(named).is_file():
        return Path(named), units
    from mbo_utilities.analysis.linescan import experiment_linescan_mesc

    return experiment_linescan_mesc(pf_dir), units


class PfArray(ReductionMixin, LazyArray):
    """A ``PF`` folder: the voltage pipeline's traces, domains and events of
    one experiment, with the line scan they came from as the image when that
    file is reachable.

    Parameters
    ----------
    filenames : path
        The ``PF`` folder, its ``denoised_trace_scans.pkl`` or its
        ``<date>_<tags>.zarr`` results file (``mbo_utilities.results``, the
        pipeline's ``output_format="zarr"``), or the experiment folder
        holding ``PF``.
    scan : str or int, optional
        The scan the image and ``fs`` follow; the first by default.
    unit : str, optional
        The same choice given as the line scan's unit (``MUnit_35``).
    source : bool
        Open the source ``.mesc`` unit as the image when it is reachable;
        False always shows the trace raster.
    """

    PRIORITY = 70

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        return isinstance(file, (str, Path)) and pf_dir_of(file) is not None

    def __init__(
        self,
        filenames: Path | str,
        scan: str | int | None = None,
        unit: str | None = None,
        source: bool = True,
    ):
        pf_dir = pf_dir_of(filenames)
        if pf_dir is None:
            raise FileNotFoundError(f"no {TRACES_FILE} or voltage results zarr at {filenames}")
        self.pf_dir = pf_dir
        self.filenames = [pf_dir]
        self._metadata: dict = {}
        self.results_path = (
            pf_dir if pf_dir.suffix == ".zarr"
            else None if (pf_dir / TRACES_FILE).is_file()
            else pf_results_in(pf_dir)
        )
        try:
            from vnoiser import read_pf
        except ImportError:
            read_pf = None
        if self.results_path is not None:
            from mbo_utilities.results import read_results

            results = read_results(self.results_path)
            scans = [u for u in results.units.values() if u.kind == "scan"]
            traces = {u.attrs["scan_id"]: dict(zip(u.roi_names, u.traces["denoised"], strict=True)) for u in scans}
            fs = {u.attrs["scan_id"]: u.fs for u in scans}
            rois = {
                "scanID_spatial": [u.attrs["scan_id"] for u in scans],
                "scanID_1st_env": [u.attrs["scan_id"] for u in scans if u.attrs.get("first_env")],
                "domain_ROInumber": {n: m.tolist() for u in scans[:1] for n, m in zip(u.roi_names, u.members, strict=True)},
                "roi_list": {u.attrs["scan_id"]: list(u.attrs.get("member_ids") or []) for u in scans},
            }
            peaks = {u.attrs["scan_id"]: u.events for u in scans}
            provenance = results.provenance or {"source": results.source, "settings": results.settings}
            params = dict(provenance.get("events") or {})
        elif read_pf is not None:
            files = read_pf(pf_dir)
            traces, fs, rois = files.traces, files.fs, files.rois
            peaks, params, provenance = files.peaks or {}, files.params or {}, files.provenance or {}
        else:
            with (pf_dir / TRACES_FILE).open("rb") as handle:
                traces = pickle.load(handle)
            with (pf_dir / FS_FILE).open("rb") as handle:
                fs = pickle.load(handle)
            with (pf_dir / ROIS_FILE).open("rb") as handle:
                rois = pickle.load(handle)
            peaks, params, provenance = {}, {}, {}
            if (pf_dir / PEAKS_FILE).is_file():
                with (pf_dir / PEAKS_FILE).open("rb") as handle:
                    peaks = pickle.load(handle)
            if (pf_dir / PARAMS_FILE).is_file():
                with (pf_dir / PARAMS_FILE).open("rb") as handle:
                    params = pickle.load(handle)
            if (pf_dir / PROVENANCE_FILE).is_file():
                provenance = json.loads((pf_dir / PROVENANCE_FILE).read_text())
        self.traces = {
            str(s): {str(d): np.asarray(t, dtype=np.float64) for d, t in v.items()}
            for s, v in traces.items()
        }
        self.fs_by_scan = {str(s): float(v) for s, v in fs.items() if v is not None}
        self.scan_ids = [str(s) for s in rois.get("scanID_spatial", list(self.traces))]
        self.first_env = [str(s) for s in rois.get("scanID_1st_env", [])]
        self.roi_list = {str(s): [int(r) for r in v] for s, v in rois.get("roi_list", {}).items()}
        self.domains = {
            FINAL_DOMAIN_NAMES.get(str(k), str(k)): [int(r) for r in v]
            for k, v in rois.get("domain_ROInumber", {}).items()
            if str(k) not in EXCLUDED_DOMAINS
        }
        self.peaks = {
            str(s): {str(d): np.asarray(p, dtype=int) for d, p in v.items()}
            for s, v in peaks.items()
        }
        self.params = dict(params)
        self.provenance = dict(provenance)
        self.settings = dict(self.provenance.get("settings") or {})
        traced = list(dict.fromkeys(d for v in self.traces.values() for d in v))
        self.domain_names = [d for d in self.domains if d in traced] + [d for d in traced if d not in self.domains]
        if scan is None and unit is not None:
            scan = str(unit).rsplit("/", 1)[-1].rsplit("_", 1)[-1]
        if scan is not None and str(scan) not in self.scan_ids:
            raise ValueError(f"{pf_dir} has no scan {scan!r}; its scans are {self.scan_ids}")
        self.scan = str(scan) if scan is not None else (self.scan_ids[0] if self.scan_ids else "")
        self.source_mesc, self.source_units = pf_source(pf_dir, self.provenance)
        self._source = None
        if source and self.source_mesc is not None and self.scan:
            from mbo_utilities.arrays.mesc import MescArray

            unit_key = self.source_units.get(self.scan, f"MUnit_{self.scan}")
            try:
                self._source = MescArray(self.source_mesc, unit=unit_key)
            except Exception as error:
                logger.warning(
                    f"{self.source_mesc.name} unit {unit_key} does not open ({error}); showing the trace raster"
                )
        self._raster = None

    @property
    def raster_bin(self) -> int:
        """Samples per raster column, so the longest trace spans at most ``RASTER_WIDTH`` columns."""
        longest = max((len(t) for v in self.traces.values() for t in v.values()), default=1)
        return max(1, -(-longest // RASTER_WIDTH))

    @property
    def raster(self) -> np.ndarray:
        """``(scans, domains, columns)`` float32: every scan's traces binned by
        ``raster_bin``, NaN past a trace's end and for a domain a scan lacks."""
        if self._raster is None:
            b = self.raster_bin
            longest = max((len(t) for v in self.traces.values() for t in v.values()), default=1)
            width = max(1, -(-longest // b))
            out = np.full((len(self.scan_ids), len(self.domain_names), width), np.nan, dtype=np.float32)
            for i, scan in enumerate(self.scan_ids):
                for j, domain in enumerate(self.domain_names):
                    trace = self.traces.get(scan, {}).get(domain)
                    if trace is None or not len(trace):
                        continue
                    n = len(trace) // b
                    out[i, j, :n] = trace[: n * b].reshape(n, b).mean(axis=1)
                    if n * b < len(trace):
                        out[i, j, n] = trace[n * b :].mean()
            self._raster = out
        return self._raster

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        if self._source is not None:
            return self._source._shape5d()
        scans, domains, width = self.raster.shape
        return (1, 1, scans, domains, width)

    @property
    def dtype(self):
        return self._source.dtype if self._source is not None else np.dtype(np.float32)

    def __getitem__(self, key):
        if self._source is not None:
            return self._source[key]
        return self.raster[None, None][key]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def source_path(self) -> Path:
        return self.pf_dir

    @property
    def reader_kwargs(self) -> dict:
        """Kwargs `imread` needs to re-open this scan in another process."""
        return {"scan": self.scan}

    @property
    def unit_key(self) -> str:
        """The source unit shown, ``MSession_0/MUnit_35``, or "" without a source."""
        if self._source is not None:
            return str(self._source.unit_key)
        return self.source_units.get(self.scan, "")

    @property
    def slider_dim_labels(self) -> tuple[str, ...]:
        if self._source is not None:
            return self._source.slider_dim_labels
        return ("Scan",) if len(self.scan_ids) > 1 else ()

    @property
    def metadata(self) -> dict:
        md = dict(self._source.metadata) if self._source is not None else {}
        md.update(
            {
                "pf_dir": str(self.pf_dir),
                "pf_scan": self.scan,
                "pf_scans": list(self.scan_ids),
                "pf_first_env": list(self.first_env),
                "pf_domains": {k: list(v) for k, v in self.domains.items()},
                "voltage_settings": dict(self.settings),
                "source_mesc": None if self.source_mesc is None else str(self.source_mesc),
                "source_unit": self.unit_key,
            }
        )
        md.setdefault("fs", self.fs_by_scan.get(self.scan))
        # the run's per-step wall, cpu and memory record, as the pipeline wrote it
        for key in ("timing", "processing_history"):
            if key in self.provenance:
                md[key] = self.provenance[key]
        md.update(self._metadata)
        return md

    @metadata.setter
    def metadata(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = dict(value)

    def trace(self, domain: str, scan: str | None = None) -> np.ndarray:
        """The denoised trace of ``domain`` in ``scan`` (the current scan by default)."""
        return self.traces[str(scan or self.scan)][str(domain)]

    def events(self, domain: str, scan: str | None = None) -> np.ndarray:
        """Sample indices of the detected events; empty when none were detected."""
        return self.peaks.get(str(scan or self.scan), {}).get(str(domain), np.zeros(0, dtype=int))

    def recording_id(self, domain: str, scan: str | None = None) -> str:
        """The curation's id of a domain trace, ``scan=<id>/domain=<name>``:
        what its labels are keyed by in ``PF/.curation``."""
        return f"scan={scan or self.scan}/domain={domain}"

    def domain_of_line(self, line: int) -> str | None:
        """The domain that averages line ROI ``line``, or None."""
        return next((d for d, rois in self.domains.items() if int(line) in rois), None)

    def _imwrite(self, outpath, **kwargs):
        if self._source is not None:
            return self._source._imwrite(outpath, **kwargs)
        return _imwrite_base(self, outpath, **kwargs)

    def save(self, outpath, **kwargs):
        return self._imwrite(outpath, **kwargs)
