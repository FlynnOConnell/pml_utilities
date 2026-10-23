"""The vnoiser pipeline on a raw ``.mesc``.

Each recording unit traces come from (:func:`~mbo_utilities.arrays.mesc.is_trace_unit`:
the lines of a line scan, the patches of a chessboard, the boxes of a ribbon
scan, or a single patch or raster whose whole frame is the ROI) is one scan.
Each domain (the lines of a soma, a patch, the part of a patch the cell
covers) is read with :func:`~mbo_utilities.analysis.roi_traces.extract_traces`:
its pixels (every pixel, a scored top fraction or a drawn mask), the file's
conversion offset, a background, a reference channel and the inversion, as
``settings.traces`` says. dF/F and the z-score follow ``settings.dfof``
(:func:`~mbo_utilities.analysis.dff.dfof_zscore`; the archive's Gaussian by
default), then vnoiser's wavelet denoising and peaks
(:func:`vnoiser.pipeline.process_domain`) unless ``settings.runtime.denoise``
is off, and the results zarr (or the archive's ``PF`` folder,
:class:`vnoiser.pf.PfWriter`) that ``mbo curate`` opens. The runner owns
that loop so every step (each scan's read, its dF/F, each domain's
denoising, the writes) is logged with its wall time, CPU time and memory,
and recorded per step in ``pipeline.json`` (``timing`` and
``processing_history``) and ``timings.json`` beside the results
(:class:`_RunUsage`). Domains come from a ``domains.json`` beside the file,
an archive's ``scanIDs_ROIs.pkl``, or the Run tab. Settings are written for
the archive's frame rate and scaled to the scans'
(:meth:`~mbo_utilities.vnoiser.params.VnoiserSettings.at_fs`).

Reproduces the archive: ``stan112_expt12/stan112_expt12/stan112_expt12.mesc``
units 35 and 38, read this way with the default settings, give the per-ROI
traces the lab's ``VI_2025-07-24.pkl`` holds to float precision, and the
traces of its ``PF`` folder follow (see ``tests/test_vnoiser_pipeline.py``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
import psutil
import zarr
from vnoiser.pf import DFOF_FILE, PROVENANCE_FILE, PfWriter, final_domain_name

from mbo_utilities import log
from mbo_utilities._sysmem import MemoryMonitor, mem_snapshot
from mbo_utilities._writers import add_processing_step
from mbo_utilities.analysis.dff import baseline
from mbo_utilities.analysis.roi_traces import (
    Domain,
    TraceSettings,
    domains_from_json,
    extract_traces,
    roi_geometry,
)
from mbo_utilities.vnoiser.params import (
    OUTPUT_FORMATS,
    VnoiserDfofSettings,
    VnoiserSettings,
)
from vnoiser import DfofConfig, ScanTraces, SpikeDetectConfig
from vnoiser.pipeline import load_scan_rois, process_domain

__all__ = [
    "DOMAINS_FILE",
    "TIMINGS_FILE",
    "preview_domain",
    "scan_traces_from_mesc",
    "scan_units",
    "read_domains",
    "write_domains_template",
    "default_pf_dir",
    "run_vnoiser_pipeline",
    "TRACES_DIR",
]

DOMAINS_FILE = "domains.json"
TRACES_DIR = "traces"
TIMINGS_FILE = "timings.json"


def _munit_number(unit_key: str) -> str:
    return str(unit_key).rsplit("/", 1)[-1].rsplit("_", 1)[-1]


class _RunUsage:
    """Wall time, CPU time and memory of every step of one run.

    ``begin(step, **fields)`` then ``end(message, **fields)`` around each
    step: one INFO line, one ``processing_history`` entry
    (:func:`~mbo_utilities._writers.add_processing_step`, ``vnoiser_<step>``)
    and one flat row in ``steps``. A :class:`~mbo_utilities._sysmem.MemoryMonitor`
    samples the process tree every two seconds while the run lasts, so a
    step's ``peak_rss_gb`` is what it really used (the wavelet transform
    allocates and frees inside a step), and warns above 90 % of system
    memory. ``summary()`` is the ``timing`` block of ``pipeline.json``.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.proc = psutil.Process()
        self.started = datetime.now()
        self.t0 = time.perf_counter()
        self.cpu0 = self._cpu()
        self.steps: list[dict] = []
        self.history: dict = {}
        self.peak_gb = 0.0
        self._step = None
        self._step_peak = 0.0
        self.monitor = MemoryMonitor(
            tick_s=2.0,
            log_s=0,
            warn_pct=90.0,
            logger=logger,
            log_peaks=False,
            on_sample=self._sample,
        ).start()

    def _cpu(self) -> float:
        t = self.proc.cpu_times()
        return float(t.user + t.system)

    def _sample(self, snap: dict) -> None:
        self._step_peak = max(self._step_peak, snap["proc_gb"])
        self.peak_gb = max(self.peak_gb, snap["proc_gb"])

    def begin(self, step: str, **fields) -> None:
        self._step = (step, fields, time.perf_counter(), self._cpu())
        self._step_peak = 0.0

    def end(self, message: str, **fields) -> dict:
        step, begun, t0, cpu0 = self._step
        wall = time.perf_counter() - t0
        cpu = self._cpu() - cpu0
        snap = mem_snapshot(self.proc)
        self._sample(snap)
        row = {
            "step": step,
            **begun,
            **fields,
            "seconds": round(wall, 3),
            "cpu_seconds": round(cpu, 3),
            "rss_gb": round(snap["proc_gb"], 3),
            "peak_rss_gb": round(self._step_peak, 3),
        }
        self.steps.append(row)
        add_processing_step(
            self.history,
            f"vnoiser_{step}",
            duration_seconds=wall,
            extra={k: v for k, v in row.items() if k not in ("step", "seconds")},
        )
        self.logger.info(
            f"{message} in {wall:.1f} s (cpu {cpu:.1f} s) | process {snap['proc_gb']:.2f} GB, "
            f"peak {self._step_peak:.2f} GB, system {snap['sys_pct']:.0f}% of {snap['total_gb']:.0f} GB"
        )
        self._step = None
        return row

    def summary(self) -> dict:
        """Totals per step, per scan (and per domain), and the denoiser's stages summed over domains."""
        totals: dict[str, float] = {}
        stages: dict[str, float] = {}
        scans: dict[str, dict] = {}
        for row in self.steps:
            step, seconds = row["step"], row["seconds"]
            totals[step] = round(totals.get(step, 0.0) + seconds, 3)
            if step == "denoise":
                for key, value in row.items():
                    if key.endswith("_s"):
                        stages[key[:-2]] = round(stages.get(key[:-2], 0.0) + value, 3)
            if "scan" not in row:
                continue
            scan = scans.setdefault(row["scan"], {})
            scan[step] = round(scan.get(step, 0.0) + seconds, 3)
            if "domain" in row:
                scan.setdefault("domains", {}).setdefault(row["domain"], {})[step] = (
                    seconds
                )
        return {
            "started": self.started.isoformat(timespec="seconds"),
            "finished": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": round(time.perf_counter() - self.t0, 3),
            "cpu_seconds": round(self._cpu() - self.cpu0, 3),
            "cpu_count": os.cpu_count(),
            "peak_rss_gb": round(self.peak_gb, 3),
            "totals": totals,
            "denoise_stages": stages,
            "scans": scans,
        }

    def close(self) -> None:
        self.monitor.stop()


def _write_timing(pf_dir: Path, usage: _RunUsage) -> dict:
    """Put the run's timing into ``pipeline.json`` and ``timings.json``; returns the provenance."""
    timing = usage.summary()
    prov = json.loads((pf_dir / PROVENANCE_FILE).read_text())
    prov["timing"] = timing
    prov["processing_history"] = list(usage.history.get("processing_history", []))
    (pf_dir / PROVENANCE_FILE).write_text(json.dumps(prov, indent=2))
    (pf_dir / TIMINGS_FILE).write_text(
        json.dumps({**timing, "steps": usage.steps}, indent=2)
    )
    return prov


def _read_progress(
    logger, progress_callback, scan_id, k, n_scans, i, n, _seconds
) -> None:
    """``linescan_roi_read``'s per-ROI callback: one log line, one progress tick.

    The line carries no elapsed time: the log's timestamps say when each ROI
    landed. The reads are the first 20 % of the run, scan ``k`` of
    ``n_scans``; the last ROI of the last scan lands on 0.2 exactly.
    """
    logger.info(f"scan {scan_id}: read ROI {i + 1}/{n}")
    if progress_callback is not None:
        progress_callback(
            0.2 * min(1.0, (k + (i + 1) / n) / n_scans),
            f"scan {scan_id}: read ROI {i + 1}/{n}",
        )


def scan_units(mesc_path, units=None) -> list[dict]:
    """The recordings of ``mesc_path`` traces come from
    (:func:`~mbo_utilities.arrays.mesc.is_trace_unit`), as
    :func:`~mbo_utilities.arrays.mesc.list_mesc_units` rows in file order, or
    the ``units`` named (``MUnit_n`` or ``MSession_s/MUnit_n``) in that
    order. Naming a unit that holds no trace (a picture, a Z-stack) raises.
    """
    from mbo_utilities.arrays.mesc import is_trace_unit, list_mesc_units

    mesc_path = Path(mesc_path)
    every = list_mesc_units(mesc_path)
    traced = [u for u in every if is_trace_unit(u)]
    if not units:
        return traced
    traced_munits = {u["munit"] for u in traced}
    chosen = []
    for name in units:
        match = [u for u in traced if u["key"] == name or u["munit"] == name]
        if not match:
            other = [
                u
                for u in every
                if u["key"] == name
                or (u["munit"] == name and name not in traced_munits)
            ]
            if other:
                raise ValueError(
                    f"{name} is a {other[0]['modality_name']} unit ({other[0]['role'] or 'no role'}, "
                    f"{other[0]['nframes']} frame(s)) with no trace to take"
                )
            raise ValueError(f"{name} is not in {mesc_path.name}")
        chosen.append(match[0])
    return chosen


def scan_traces_from_mesc(
    mesc_path,
    unit_key: str,
    *,
    channel: int = 0,
    convert: bool = False,
    frames=None,
    rois=None,
    batch_size: int = 20000,
    progress=None,
) -> ScanTraces:
    """One recording unit's whole-ROI means as vnoiser's :class:`ScanTraces`.

    The ROIs are the unit's lines, patches or boxes, or its one frame for a
    single-patch scan or a raster. ``convert=False`` keeps MESc's raw counts,
    as the archive's converter did; ``True`` applies the file's linear
    conversion so zero means no photons (the dF/F then differs from the
    archive's by a slowly varying factor; the z-score is nearly unaffected).
    ``frames=(start, stop)`` keeps that half-open window of the recording.
    ``rois`` (0-based ROI indices: the unit's Z axis) reads only those ROIs;
    the traces stay keyed by the ROI's own index. ``progress(i, K, seconds)``
    is called after each ROI when given.
    """
    from mbo_utilities.arrays.mesc import MescArray

    arr = MescArray(mesc_path, unit=unit_key)
    md = arr.metadata
    geometry = roi_geometry(arr)
    if rois is not None and md.get("mesc_z_axis_meaning") != "roi_index":
        raise ValueError(
            f"{unit_key}: Z means {md.get('mesc_z_axis_meaning')!r} here, not ROIs"
        )
    rois = sorted(geometry) if rois is None else [int(r) for r in rois]
    found = extract_traces(
        arr,
        [Domain(f"roi{r}", [r]) for r in rois],
        TraceSettings(convert=convert),
        channel=channel,
        frames=frames,
        batch_size=batch_size,
        progress=progress,
    )
    return ScanTraces(
        _munit_number(md.get("mesc_unit", unit_key)),
        float(md["fs"]),
        {r: found.roi_mean[r] for r in rois},
        {r: float(geometry[r][1] * geometry[r][2]) for r in rois},
        comment=str(md.get("comment", "") or ""),
    )


def read_domains(path) -> dict:
    """``{"domains", "scan_ids", "first_env"}`` from a ``domains.json`` or an
    archive ``scanIDs_ROIs.pkl``.

    The JSON holds ``{"domains": {"soma1": [0, 1, 2], "cell": {"rois": [3],
    "pixels": "correlated", "fraction": 0.3}, ...}, "scans": ["35", "38"],
    "first_env": ["35"]}``: a domain is an ROI list (every pixel of those
    ROIs) or a :meth:`~mbo_utilities.analysis.roi_traces.Domain.from_json`
    entry; ``scans`` and ``first_env`` are optional. ``domains`` comes back
    as given, checked.
    """
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        rois = load_scan_rois(path)
        return {
            "domains": rois["domains"],
            "scan_ids": rois["scan_ids"],
            "first_env": rois["first_env"],
        }
    doc = json.loads(path.read_text())
    found = domains_from_json(doc.get("domains") or {})
    if not found:
        raise ValueError(f"{path} defines no domains")
    return {
        "domains": {d.name: d.to_json() for d in found},
        "scan_ids": [str(s) for s in doc.get("scans", [])],
        "first_env": [str(s) for s in doc.get("first_env", [])],
    }


def comment_names(comment: str, n_rois: int) -> list[str] | None:
    """One name per ROI from a unit comment that lists as many
    (``"soma, bas1, api1"``), else None: ``"soma,bas1-3,api1-5"`` names
    groups, not ROIs, and a free comment names nothing.
    """
    names = [n.strip() for n in str(comment or "").split(",") if n.strip()]
    if len(names) != n_rois or len(set(names)) != len(names):
        return None
    if any(not n.replace("_", "").replace("-", "").isalnum() for n in names):
        return None
    return names


def write_domains_template(
    mesc_path, path=None, *, units=None, per_domain: int = 1
) -> Path:
    """Write a ``domains.json`` to edit: the file's recordings as
    ``scans`` and one domain per ROI (``roi0: [0]``, ...; ``per_domain``
    groups consecutive ROIs instead, the archive's three lines per
    domain), named from the unit comment when it lists as many names
    (``'soma,bas1-3,api1-5'`` does not). Returns the path.
    """
    mesc_path = Path(mesc_path)
    path = Path(path) if path is not None else mesc_path.parent / DOMAINS_FILE
    scans = scan_units(mesc_path, units)
    if not scans:
        raise ValueError(f"no recording with traces in {mesc_path}")
    n_rois = int(scans[0]["nrois"])
    groups = [
        list(range(i, min(i + per_domain, n_rois)))
        for i in range(0, n_rois, per_domain)
    ]
    names = comment_names(scans[0].get("comment"), len(groups)) or [
        f"roi{g[0]}" if len(g) == 1 else f"domain{i + 1}" for i, g in enumerate(groups)
    ]
    doc = {
        "mesc": mesc_path.name,
        "scans": [_munit_number(u["key"]) for u in scans],
        "first_env": [_munit_number(scans[0]["key"])],
        "domains": {name: rois for name, rois in zip(names, groups, strict=True)},
        "note": "domains: name -> ROI indices (0-based, the order the lines or patches were drawn), "
        'or {"rois": [...], "pixels": "all|bright|active|correlated|mask", "fraction": 0.3, '
        '"mask": {"<roi>": [flat pixel indices]}}; scans: MUnit numbers in order; '
        "first_env: the first scan of each environment",
    }
    path.write_text(json.dumps(doc, indent=2))
    return path


def default_pf_dir(mesc_path) -> Path:
    """Where a ``pkl`` run's PF folder goes: ``<animal>/<expt>/PF`` for the
    archive's ``<animal>/<expt>/<expt>/<expt>.mesc`` layout, else ``PF``
    beside the file.
    """
    mesc_path = Path(mesc_path)
    if (
        mesc_path.parent.name == mesc_path.stem
        and mesc_path.parent.parent != mesc_path.parent
    ):
        return mesc_path.parent.parent / "PF"
    return mesc_path.parent / "PF"


def default_results_path(mesc_path, out=None) -> Path:
    """The results zarr a ``zarr`` run writes: ``out`` when it names one, a
    fresh name inside ``out`` when it names a folder, else beside the file.

    Named after its input (:func:`mbo_utilities.results.results_name`), so a
    run lands as ``session1.2026-09-21-14-30-22.vnoiser.zarr`` next to
    ``session1.mesc`` and never overwrites an earlier one.
    """
    from mbo_utilities.results import results_name

    mesc_path = Path(mesc_path)
    out = mesc_path.parent if out is None else Path(out)
    if out.suffix == ".zarr":
        return out
    return out / results_name(mesc_path, pipeline="vnoiser")


def _cut_domains(
    domains: list[Domain], rois: set[int]
) -> tuple[list[Domain], list[str]]:
    """Every domain cut down to ``rois``; one left with none is dropped (named in the second list)."""
    kept, dropped = [], []
    for d in domains:
        mine = [r for r in d.rois if r in rois]
        if not mine:
            dropped.append(d.name)
            continue
        mask = None if d.mask is None else {r: d.mask[r] for r in mine}
        kept.append(Domain(d.name, mine, d.pixels, d.fraction, mask))
    return kept, dropped


def _versions() -> tuple[str | None, dict]:
    mbo_version = None
    for dist in ("pml_utilities", "mbo_utilities"):
        try:
            mbo_version = version(dist)
            break
        except PackageNotFoundError:
            continue
    versions = {"python": platform.python_version()}
    for name in ("vnoiser", "numpy", "scipy", "scikit-learn", "PyWavelets", "h5py"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return mbo_version, versions


def run_vnoiser_pipeline(
    mesc_path,
    *,
    domains,
    units=None,
    first_env=(),
    out=None,
    channel: int = 0,
    convert: bool | None = None,
    frames=None,
    planes=None,
    save_cwt: bool = False,
    spike_cfg: SpikeDetectConfig | None = None,
    detect: bool = True,
    dfof_cfg: DfofConfig | None = None,
    denoiser_factory=None,
    settings: VnoiserSettings | None = None,
    overwrite: bool = False,
    provenance: dict | None = None,
    progress_callback=None,
    logger: logging.Logger | None = None,
) -> dict:
    """Every recording of ``mesc_path`` (or ``units``) into a results file or PF folder.

    Parameters
    ----------
    mesc_path : path
    domains : mapping of domain -> entry, or sequence of Domain
        An entry is an ROI list or a
        :meth:`~mbo_utilities.analysis.roi_traces.Domain.from_json` dict
        (pixels, fraction, mask). ``All_domains`` is ignored, and ``bg``
        gets no trace unless it is the background domain.
    units : sequence of str, optional
        ``MUnit_n`` or ``MSession_s/MUnit_n`` keys, in scan order; default
        every recording traces come from (:func:`scan_units`). One run takes
        scans of one frame rate.
    first_env : sequence of scan ids
    out : path, optional
        The PF folder; default :func:`default_pf_dir`.
    channel : int
        0-based channel the indicator is on.
    convert : bool, optional
        Overrides ``settings.traces.convert`` when given.
    frames : (start, stop), optional
        The half-open window of frames read.
    planes : sequence of int, optional
        The pipeline input contract's 1-based Z selection. On an AOD ROI
        unit Z is the ROI index, so these are the ROIs (lines, patches) to
        process: only they are read, every domain is cut down to them and a
        domain left with no ROI is dropped. Default every ROI the domains use.
    settings : VnoiserSettings, optional
        The Run tab's settings (default :class:`VnoiserSettings`), scaled to
        the scans' frame rate with ``at_fs``: ``traces`` (pixels in,
        background, reference, inversion), ``dfof``, the denoiser, peaks and
        runtime. A ``dfof_cfg``, ``denoiser_factory`` or ``spike_cfg`` given
        explicitly is used as it is instead.
    save_cwt, spike_cfg, detect, dfof_cfg, denoiser_factory, overwrite :
        see :func:`vnoiser.run_pipeline`
    provenance : dict, optional
        Extra keys for ``pipeline.json`` beside the source block.
    progress_callback : callable(fraction, message), optional
        Called after every ROI read, before every domain's denoising and
        at each write; the fraction runs 0 to 1.
    logger : logging.Logger, optional
        Where every step is reported (default the ``mbo`` logger): one line
        when a step starts and one when it ends with its wall time, CPU
        time and memory, then a summary line.

    Returns ``{file name: path}`` of the written folder. Beside the archive
    format a ``traces`` subfolder holds plain files: ``scans.csv``,
    ``domains.csv``, and per scan ``scan<id>_rois.npy`` (ROI, frame: each
    whole ROI's mean), ``scan<id>_f.npy`` (domain, frame: the trace dF/F was
    taken of, before any inversion), ``scan<id>_background.npy`` /
    ``_reference.npy`` when there was one, ``scan<id>_masks.json`` (each
    domain's pixels per ROI) and ``scan<id>_images.npz`` (each ROI's mean,
    SD and correlation images when a pixel rule scored them),
    ``scan<id>_dfof.npy`` / ``_zscore.npy`` / ``_denoised.npy`` (domain,
    frame) in ``domains.csv`` row order, ``scan<id>_peaks.csv``, and
    ``scan<id>_denoised.png`` / ``scan<id>_rois.png`` figures.
    ``pipeline.json`` carries the run's ``timing`` (wall, CPU and peak
    memory, totals per step, per scan and per domain, and the denoiser's
    stages ``cwt`` / ``cluster`` / ``reduce`` / ``mask`` / ``baseline`` /
    ``baseline_100hz`` / ``peaks`` summed over domains) and its
    ``processing_history`` (one ``vnoiser_<step>`` entry per step, the shape
    suite2p's ``ops.npy`` uses); ``timings.json`` beside it is the same with
    one flat row per step, a denoise row carrying its stages as ``<stage>_s``.
    With ``settings.runtime.output_format == "zarr"`` the pickles are
    replaced by one ``<stem>.<stamp>.vnoiser.zarr`` results file
    (:mod:`mbo_utilities.results`; ``test.h5``, ``traces``, ``pipeline.json``
    and ``timings.json`` move into its ``vnoiser/`` folder) whose
    ``provenance`` attr holds the same timing. With ``settings.runtime.denoise``
    off the run stops after the z-score: no denoised trace, no peaks, zarr only.
    """
    from mbo_utilities.arrays.mesc import MescArray

    logger = logger or log.get()
    mesc_path = Path(mesc_path)
    chosen = scan_units(mesc_path, units)
    if not chosen:
        raise ValueError(f"no recording with traces in {mesc_path}")
    rates = sorted({round(float(u["fs"] or 0), 3) for u in chosen})
    if len(rates) > 1:
        listed = ", ".join(f"{r:g}" for r in rates)
        raise ValueError(
            f"the chosen scans differ in frame rate ({listed} Hz); run them separately"
        )
    settings = (settings or VnoiserSettings()).at_fs(float(chosen[0]["fs"]))
    if convert is not None:
        settings.traces.convert = bool(convert)
    if settings.runtime.output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {OUTPUT_FORMATS}, got {settings.runtime.output_format!r}"
        )
    denoise = bool(settings.runtime.denoise)
    if not denoise and settings.runtime.output_format != "zarr":
        raise ValueError(
            "a run without denoising writes the results zarr, not a PF folder"
        )
    if dfof_cfg is not None:
        settings.dfof = VnoiserDfofSettings(
            sigma_dfof=float(dfof_cfg.sigma_dfof),
            sigma_baseline=float(dfof_cfg.sigma_baseline),
            n_startup=int(dfof_cfg.n_startup),
            negative=bool(dfof_cfg.negative),
        )
    polarity = settings.polarity
    detect = bool(detect) and denoise
    spike_cfg = spike_cfg or settings.events.config()
    denoiser_factory = denoiser_factory or settings.denoiser.factory
    if isinstance(domains, dict):
        given = {str(k): v for k, v in domains.items()}
        found = domains_from_json(given)
    else:
        found = list(domains)
        given = {d.name: d.to_json() for d in found}
    background = (
        settings.traces.background_domain
        if settings.traces.background == "domain"
        else None
    )
    found = [
        d
        for d in found
        if d.name != "All_domains" and (d.name != "bg" or d.name == background)
    ]
    rois = None
    if planes is not None:
        rois = sorted({int(p) - 1 for p in planes})
        n_rois = min(int(u["nrois"]) for u in chosen)
        bad = [r + 1 for r in rois if not 0 <= r < n_rois]
        if bad:
            raise ValueError(f"planes {bad} are outside 1..{n_rois}, the scans' ROIs")
        found, dropped = _cut_domains(found, set(rois))
        if not [d for d in found if d.name != background]:
            raise ValueError(
                f"no domain has an ROI among planes {[r + 1 for r in rois]}"
            )
        if dropped:
            logger.info(
                f"vnoiser: planes {[r + 1 for r in rois]} leave no ROI in domain(s) {dropped}; dropped"
            )
    if background is not None and not any(d.name == background for d in found):
        raise ValueError(
            f"the background domain {background!r} is not among the domains"
        )
    traced = [d for d in found if d.name != background]
    if not traced:
        raise ValueError("no domains to process")
    names = [d.name for d in traced]
    domain_rois = {d.name: list(d.rois) for d in found}
    # a zarr run works in a scratch folder the pipeline's own writers need and
    # folds it into the results file at the end; a pkl run works in PF itself
    as_zarr = settings.runtime.output_format == "zarr"
    results_path = default_results_path(mesc_path, out) if as_zarr else None
    pf_dir = (
        results_path.with_suffix(".work")
        if as_zarr
        else (Path(out) if out is not None else default_pf_dir(mesc_path))
    )
    mbo_version, versions = _versions()

    usage = _RunUsage(logger)
    system = mem_snapshot(usage.proc)
    logger.info(
        f"vnoiser: {len(chosen)} scan(s) x {len(names)} domain(s) -> {results_path or pf_dir} "
        f"({settings.runtime.output_format}); {os.cpu_count()} cpus, {system['total_gb']:.0f} GB RAM "
        f"({system['sys_pct']:.0f}% in use); vnoiser {versions['vnoiser']}, mbo_utilities {mbo_version}"
    )
    logger.info(
        f"vnoiser: polarity {polarity}, pixels "
        + ", ".join(f"{d.name}={d.pixels}" for d in traced)
        + f", background {settings.traces.background}, reference {settings.traces.reference}, "
        f"conversion {'on' if settings.traces.convert else 'off'}, dF/F {settings.dfof.method}, "
        f"z noise {settings.dfof.noise}{'' if denoise else ', no denoising'}"
    )
    try:
        scans, extracted = [], {}
        for k, u in enumerate(chosen):
            sid = _munit_number(u["key"])
            logger.info(
                f"scan {sid}: reading {u['key']} ({u['nrois']} ROIs x {u['nframes']} frames at {u['fs']:.2f} Hz)"
            )
            usage.begin("read", scan=sid, unit=u["key"])
            arr = MescArray(mesc_path, unit=u["key"])
            ex = extract_traces(
                arr,
                found,
                settings.traces,
                channel=channel,
                frames=frames,
                progress=partial(
                    _read_progress, logger, progress_callback, sid, k, len(chosen)
                ),
            )
            geometry = roi_geometry(arr)
            scan = ScanTraces(
                sid,
                float(arr.metadata["fs"]),
                ex.roi_mean,
                {r: float(geometry[r][1] * geometry[r][2]) for r in ex.roi_mean},
                comment=str(arr.metadata.get("comment", "") or u.get("comment") or ""),
            )
            scans.append(scan)
            extracted[sid] = ex
            usage.end(
                f"scan {sid}: read {len(scan.traces)} ROIs x {scan.n_frames} frames",
                n_rois=len(scan.traces),
                n_frames=scan.n_frames,
            )

        example = denoiser_factory(scans[0].fs_hz)
        info = {
            "pipeline": "mbo_utilities.vnoiser.pipeline.run_vnoiser_pipeline",
            "versions": versions,
            "traces": settings.traces.to_dict(),
            "polarity": polarity,
            "dfof": asdict(settings.dfof),
            "denoiser": example.describe() if denoise else None,
            "events": asdict(spike_cfg) if detect else None,
            "comments": {s.scan_id: s.comment for s in scans if s.comment},
            "roi_weights": {
                s.scan_id: {int(k): float(v) for k, v in s.weights.items()}
                for s in scans
            },
            "source": {
                "mesc": str(mesc_path),
                "units": {
                    s.scan_id: u["key"] for s, u in zip(scans, chosen, strict=True)
                },
                "channel": int(channel),
                "convert": bool(settings.traces.convert),
                "frames": None if frames is None else [int(frames[0]), int(frames[1])],
                "planes": None if planes is None else [int(p) for p in planes],
                # the table as given; `planes` cuts the one the run works on
                "domains": given,
                "mbo_utilities": mbo_version,
            },
        }
        info.update(provenance or {})
        processed = {s.scan_id for s in scans}
        first_env = [str(s) for s in first_env if str(s) in processed]
        writer = None
        if denoise:
            writer = PfWriter(
                pf_dir,
                domains={n: domain_rois[n] for n in names}
                | ({"bg": domain_rois["bg"]} if background == "bg" else {}),
                scan_ids=[s.scan_id for s in scans],
                first_env=first_env,
                roi_list={s.scan_id: s.rois for s in scans},
                spike_params=spike_cfg.to_param_pickle() if detect else None,
                save_cwt=save_cwt,
                freq_scales=example.freq_scales,
                provenance=info,
                overwrite=overwrite,
            )
        else:
            pf_dir.mkdir(parents=True, exist_ok=True)
        n_domains = max(len(names), 1)
        normalized, denoised_rows, peaks = {}, {}, {}
        for k, scan in enumerate(scans):
            sid = scan.scan_id
            ex = extracted[sid]
            # one domain at a time: each row of dF/F and z is independent of the others, and the
            # baseline filters take about ten seconds per domain on a five-minute scan
            dfof_rows, z_rows = [], []
            for row, name in enumerate(names):
                if progress_callback is not None:
                    # dF/F is the next 30 % of the run, scan k of the run, domain row of the scan
                    fraction = 0.2 + 0.3 * min(1.0, (k + row / n_domains) / len(scans))
                    progress_callback(
                        fraction, f"scan {sid}: dF/F of {name} ({row + 1}/{len(names)})"
                    )
                usage.begin("dfof", scan=sid, domain=name)
                dfof, z = settings.dfof.apply(ex.F[[ex.row(name)]], scan.fs_hz)
                dfof_rows.append(dfof[0])
                z_rows.append(z[0])
                usage.end(
                    f"scan {sid}: dF/F and z-score of {name} ({row + 1}/{len(names)})"
                )
            normalized[sid] = (np.stack(dfof_rows), np.stack(z_rows))
            if writer is not None:
                writer.add_scan(sid, scan.fs_hz, *normalized[sid])
            if not denoise:
                continue
            rows = []
            for row, name in enumerate(names):
                if progress_callback is not None:
                    # the denoising is the next 40 %
                    fraction = 0.5 + 0.4 * min(1.0, (k + row / n_domains) / len(scans))
                    progress_callback(
                        fraction,
                        f"scan {sid}: denoising {name} ({row + 1}/{len(names)})",
                    )
                logger.info(f"scan {sid}: denoising {name} ({row + 1}/{len(names)})")
                usage.begin("denoise", scan=sid, domain=name)
                result = process_domain(
                    normalized[sid][1][row],
                    scan.fs_hz,
                    denoiser=denoiser_factory(scan.fs_hz),
                    spike_cfg=spike_cfg,
                    detect=detect,
                    keep_cwt=save_cwt,
                )
                writer.add_domain(sid, name, result)
                rows.append(result.trace)
                if result.peaks is not None:
                    peaks.setdefault(sid, {})[final_domain_name(name)] = np.asarray(
                        result.peaks, dtype=np.int64
                    )
                result.cwt = None
                n_events = None if result.peaks is None else int(len(result.peaks))
                events = "" if n_events is None else f", {n_events} events"
                stages = {
                    stage: float(seconds)
                    for stage, seconds in (result.timing or {}).items()
                }
                inside = ", ".join(
                    f"{stage} {seconds:.1f}" for stage, seconds in stages.items()
                )
                usage.end(
                    f"scan {sid}: denoised {name} ({row + 1}/{len(names)}){events} [{inside} s]",
                    n_events=n_events,
                    **{
                        f"{stage}_s": round(seconds, 3)
                        for stage, seconds in stages.items()
                    },
                )
            denoised_rows[sid] = np.stack(rows)

        if progress_callback is not None:
            progress_callback(0.9, f"writing {pf_dir.name}")
        if writer is not None:
            usage.begin("write_pf")
            paths = writer.finish()
            usage.end(f"wrote {len(paths)} PF files to {pf_dir}")
        else:
            usage.begin("write_pf")
            paths = _write_traces_only(
                pf_dir, info, scans, names, domain_rois, first_env, normalized
            )
            usage.end(f"wrote dF/F and z of {len(scans)} scan(s) to {pf_dir}")
        traces_dir = pf_dir / TRACES_DIR
        traces_dir.mkdir(exist_ok=True)
        usage.begin("traces")
        with (traces_dir / "scans.csv").open("w", newline="") as fh:
            rows = csv.writer(fh)
            rows.writerow(["scan", "unit", "fs_hz", "n_frames", "n_rois"])
            for s, u in zip(scans, chosen, strict=True):
                rows.writerow([s.scan_id, u["key"], s.fs_hz, s.n_frames, len(s.traces)])
        with (traces_dir / "domains.csv").open("w", newline="") as fh:
            rows = csv.writer(fh)
            rows.writerow(["row", "domain", "rois"])
            for i, name in enumerate(names):
                rows.writerow(
                    [
                        i,
                        final_domain_name(name),
                        " ".join(str(r) for r in domain_rois[name]),
                    ]
                )
        for s in scans:
            sid = s.scan_id
            ex = extracted[sid]
            order = [ex.row(n) for n in names]
            rois = np.stack([np.asarray(s.traces[r]) for r in sorted(s.traces)])
            np.save(traces_dir / f"scan{sid}_rois.npy", rois.astype(np.float32))
            np.save(
                traces_dir / f"scan{sid}_f.npy", ex.measured[order].astype(np.float32)
            )
            if ex.background is not None:
                np.save(
                    traces_dir / f"scan{sid}_background.npy",
                    ex.background[order].astype(np.float32),
                )
            if ex.reference is not None:
                np.save(
                    traces_dir / f"scan{sid}_reference.npy",
                    ex.reference[order].astype(np.float32),
                )
            (traces_dir / f"scan{sid}_masks.json").write_text(
                json.dumps(
                    {
                        "shapes": {str(r): list(hw) for r, hw in ex.shapes.items()},
                        "domains": {
                            d: {str(r): v.tolist() for r, v in m.items()}
                            for d, m in ex.masks.items()
                        },
                    }
                )
            )
            if ex.images:
                np.savez_compressed(
                    traces_dir / f"scan{sid}_images.npz",
                    **{
                        f"roi{r}_{kind}": img.astype(np.float32)
                        for r, imgs in ex.images.items()
                        for kind, img in imgs.items()
                    },
                )
            np.save(
                traces_dir / f"scan{sid}_dfof.npy",
                normalized[sid][0].astype(np.float32),
            )
            np.save(
                traces_dir / f"scan{sid}_zscore.npy",
                normalized[sid][1].astype(np.float32),
            )
            if denoise:
                np.save(
                    traces_dir / f"scan{sid}_denoised.npy",
                    denoised_rows[sid].astype(np.float32),
                )
                with (traces_dir / f"scan{sid}_peaks.csv").open("w", newline="") as fh:
                    rows = csv.writer(fh)
                    rows.writerow(["domain", "frame", "time_s"])
                    for n in names:
                        for frame in peaks.get(sid, {}).get(final_domain_name(n), ()):
                            rows.writerow(
                                [
                                    final_domain_name(n),
                                    int(frame),
                                    f"{int(frame) / s.fs_hz:.6f}",
                                ]
                            )
        usage.end(f"wrote {TRACES_DIR}/ for {len(scans)} scan(s)")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        usage.begin("figures")
        for s in scans:
            sid = s.scan_id
            t = np.arange(s.n_frames) / s.fs_hz
            shown = denoised_rows[sid] if denoise else normalized[sid][1]
            fig, axes = plt.subplots(
                len(names),
                1,
                figsize=(12, 1.6 * len(names) + 0.8),
                sharex=True,
                squeeze=False,
            )
            for ax, n, trace in zip(axes[:, 0], names, shown, strict=True):
                ax.plot(t, trace, lw=0.5, color="k")
                found_peaks = peaks.get(sid, {}).get(final_domain_name(n), ())
                if len(found_peaks):
                    ax.plot(t[found_peaks], trace[found_peaks], ".", color="r", ms=4)
                ax.set_ylabel(final_domain_name(n), rotation=0, ha="right", va="center")
                ax.spines[["top", "right"]].set_visible(False)
            axes[-1, 0].set_xlabel("s")
            axes[0, 0].set_title(f"scan {sid} {'denoised' if denoise else 'z-score'}")
            fig.tight_layout()
            fig.savefig(
                traces_dir / f"scan{sid}_{'denoised' if denoise else 'zscore'}.png",
                dpi=120,
            )
            plt.close(fig)
            rois = np.load(traces_dir / f"scan{sid}_rois.npy")
            fig, ax = plt.subplots(figsize=(12, 0.5 * len(rois) + 1.2))
            for i, trace in enumerate(rois):
                span = np.ptp(trace) or 1.0
                ax.plot(t, (trace - trace.mean()) / span - i, lw=0.4, color="k")
            ax.set_yticks(-np.arange(len(rois)), [str(r) for r in sorted(s.traces)])
            ax.set_ylabel("ROI")
            ax.set_xlabel("s")
            ax.set_title(f"scan {sid} raw ROIs")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(traces_dir / f"scan{sid}_rois.png", dpi=120)
            plt.close(fig)
        usage.end(f"wrote {2 * len(scans)} figures")
        paths.update(
            {f"{TRACES_DIR}/{p.name}": p for p in sorted(traces_dir.iterdir())}
        )
        prov = _write_timing(pf_dir, usage)
        if as_zarr:
            import shutil

            from mbo_utilities.results import Results, pipeline_files

            if progress_callback is not None:
                progress_callback(0.97, "writing the results zarr")
            usage.begin("results")
            results_path = Results.from_pf(pf_dir).write(results_path, overwrite=True)
            usage.end(f"wrote {results_path.name}")
            prov = _write_timing(pf_dir, usage)
            # the results file replaces the pickles; everything else the run
            # made moves into its own folder inside it, named after the
            # pipeline, so one path is the whole output
            own = pipeline_files(results_path)
            own.mkdir()
            paths = {}
            for name in (DFOF_FILE, PROVENANCE_FILE, TIMINGS_FILE, TRACES_DIR):
                source = pf_dir / name
                if source.exists():
                    shutil.move(str(source), str(own / name))
            shutil.rmtree(pf_dir, ignore_errors=True)
            for item in sorted(own.rglob("*")):
                if item.is_file():
                    paths[f"{own.name}/{item.relative_to(own).as_posix()}"] = item
            paths[results_path.name] = results_path
            zarr.open_group(str(results_path), mode="r+").attrs["provenance"] = prov
            timings_at = own / TIMINGS_FILE
        else:
            timings_at = pf_dir / TIMINGS_FILE
            paths[TIMINGS_FILE] = timings_at
        timing = prov["timing"]
        wall, cpu = timing["wall_seconds"], timing["cpu_seconds"]
        totals = ", ".join(
            f"{step} {seconds:.1f} s" for step, seconds in timing["totals"].items()
        )
        stages = ", ".join(
            f"{stage} {seconds:.1f} s"
            for stage, seconds in timing["denoise_stages"].items()
        )
        logger.info(
            f"vnoiser done in {int(wall // 60)}m {wall % 60:04.1f}s (cpu {cpu:.0f} s, {cpu / max(wall, 1e-9):.1f} cores); "
            f"peak process memory {timing['peak_rss_gb']:.2f} GB; {totals}; denoise stages: {stages or 'none'}; "
            f"timings in {timings_at}"
        )
        if progress_callback is not None:
            progress_callback(
                1.0, f"wrote {len(paths)} files to {results_path or pf_dir}"
            )
        return paths
    finally:
        usage.close()


def _write_traces_only(
    pf_dir, info, scans, names, domain_rois, first_env, normalized
) -> dict:
    """What a run without denoising leaves for ``Results.from_pf``: ``test.h5``
    (dF/F and z per scan) and the ``pipeline.json`` PfWriter would write,
    with no pickles.
    """
    with h5py.File(pf_dir / DFOF_FILE, "w") as f:
        for s in scans:
            grp = f.create_group(s.scan_id)
            grp.create_dataset("dfof_raw", data=normalized[s.scan_id][0])
            grp.create_dataset("dfof_zscore", data=normalized[s.scan_id][1])
    every = sorted({r for s in scans for r in s.rois})
    provenance = {
        "created_utc": datetime.now().astimezone().isoformat(),
        "scan_ids": [s.scan_id for s in scans],
        "first_env": list(first_env),
        "domains": {"All_domains": list(range(max(every) + 1)) if every else []}
        | {n: domain_rois[n] for n in names},
        "roi_list": {s.scan_id: s.rois for s in scans},
        "fs_hz": {s.scan_id: s.fs_hz for s in scans},
        "n_frames": {s.scan_id: s.n_frames for s in scans},
        "files": [DFOF_FILE],
        **info,
    }
    (pf_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2))
    return {DFOF_FILE: pf_dir / DFOF_FILE, PROVENANCE_FILE: pf_dir / PROVENANCE_FILE}


def preview_domain(
    mesc_path,
    unit: str,
    domains,
    name: str,
    settings: VnoiserSettings | None = None,
    *,
    channel: int = 0,
    frames=None,
) -> dict:
    """What a run would make of one domain on one scan, over ``frames``.

    ``domains`` is the run's table (``{name: entry}``, so a background
    domain is there too); only ``name`` and the background are read. The
    settings are scaled to the scan's rate as the run does. Returns ``t``
    (seconds), ``pixel_mean``, ``measured``, ``F`` (what dF/F is taken
    of), ``f0`` (its baseline), ``background``, ``reference``, ``dfof``,
    ``z`` (each ``(T,)`` or None), ``images`` / ``mask`` /
    ``shape`` per ROI of the domain, ``rois``, ``fs`` and ``info``. The Run
    tab draws it; it holds no pipeline rule of its own.
    """
    from mbo_utilities.arrays.mesc import MescArray

    found = {d.name: d for d in domains_from_json(domains)}
    if name not in found:
        raise KeyError(f"no domain {name!r}")
    arr = MescArray(mesc_path, unit=unit)
    fs = float(arr.metadata["fs"])
    settings = (settings or VnoiserSettings()).at_fs(fs)
    wanted = [found[name]]
    if settings.traces.background == "domain":
        bg = found.get(settings.traces.background_domain)
        if bg is None:
            raise ValueError(
                f"the background domain {settings.traces.background_domain!r} is not in the table"
            )
        if bg.name != name:
            wanted.append(bg)
    ex = extract_traces(
        arr, wanted, settings.traces, channel=channel, frames=frames, images=True
    )
    row = ex.row(name)
    dfof, z = settings.dfof.apply(ex.F[[row]], fs)
    d = settings.dfof
    f0 = baseline(
        ex.F[[row]],
        fs,
        d.method,
        sigma=float(d.sigma_dfof),
        window_s=float(d.window_s),
        percentile=float(d.percentile),
        f0_window_s=(float(d.f0_start_s), float(d.f0_stop_s)),
    )
    t0 = ex.frames[0]
    return {
        "t": (np.arange(ex.F.shape[1]) + t0) / fs,
        "fs": fs,
        "pixel_mean": ex.pixel_mean[row],
        "measured": ex.measured[row],
        "F": ex.F[row],
        "f0": None if f0 is None else f0[0],
        "background": None if ex.background is None else ex.background[row],
        "reference": None if ex.reference is None else ex.reference[row],
        "dfof": dfof[0],
        "z": z[0],
        "rois": list(found[name].rois),
        "images": {r: ex.images[r] for r in found[name].rois},
        "mask": {r: ex.masks[name][r] for r in found[name].rois},
        "shape": {r: ex.shapes[r] for r in found[name].rois},
        "polarity": settings.polarity,
        "info": ex.info,
    }
