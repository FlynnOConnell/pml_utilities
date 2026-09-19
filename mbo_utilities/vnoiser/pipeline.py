"""The spatial JEDI voltage pipeline on a raw AOD ``.mesc``.

Each unit with AOD ROIs (:data:`mbo_utilities.arrays.mesc.ROI_LAYOUTS`: the
lines of a line scan, the patches of a chessboard, the boxes of a ribbon
scan) is one scan; each ROI's mean fluorescence per frame
(``roi_workflow.linescan_roi_read``, the same read ``mbo linescan`` does)
goes through vnoiser's stages 0-5: domains are pixel-weighted means of
their ROIs, dF/F and z-score (:func:`vnoiser.preprocess.domain_zscore`),
wavelet denoising and peaks (:func:`vnoiser.pipeline.process_domain`), and
a ``PF`` folder that ``mbo curate`` opens (:class:`vnoiser.pf.PfWriter`).
The runner owns that loop so every step (each scan's read, its dF/F, each
domain's denoising, the writes) is logged with its wall time, CPU time and
memory, and recorded per step in ``pipeline.json`` (``timing`` and
``processing_history``) and ``timings.json`` beside the results
(:class:`_RunUsage`). Domains (which ROIs make the soma, each branch; which
patch is which cell) come from a ``domains.json`` beside the file, or an
archive's ``scanIDs_ROIs.pkl``. Settings are written for the archive's
frame rate and scaled to the scans'
(:meth:`~mbo_utilities.vnoiser.params.VoltageSettings.at_fs`).

Reproduces the archive: ``stan112_expt12/stan112_expt12/stan112_expt12.mesc``
units 35 and 38, read this way with ``convert=False``, give the per-ROI
traces the lab's ``VI_2025-07-24.pkl`` holds to float precision, and the
traces of its ``PF`` folder follow (see ``tests/test_voltage_pipeline.py``).
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
from vnoiser import DfofConfig, ScanTraces, SpikeDetectConfig, read_pf
from vnoiser.pf import DFOF_FILE, PIPELINE_FILES, PROVENANCE_FILE, PfWriter, final_domain_name
from vnoiser.pipeline import load_scan_rois, process_domain
from vnoiser.preprocess import DomainTraces, domain_names, domain_zscore

from mbo_utilities import log
from mbo_utilities._sysmem import MemoryMonitor, mem_snapshot
from mbo_utilities._writers import add_processing_step
from mbo_utilities.vnoiser.params import OUTPUT_FORMATS, VoltageSettings

__all__ = [
    "DOMAINS_FILE",
    "TIMINGS_FILE",
    "scan_traces_from_mesc",
    "read_domains",
    "write_domains_template",
    "default_pf_dir",
    "run_voltage_pipeline",
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
    (:func:`~mbo_utilities._writers.add_processing_step`, ``voltage_<step>``)
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
            tick_s=2.0, log_s=0, warn_pct=90.0, logger=logger, log_peaks=False, on_sample=self._sample,
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
            self.history, f"voltage_{step}", duration_seconds=wall,
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
                scan.setdefault("domains", {}).setdefault(row["domain"], {})[step] = seconds
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
    (pf_dir / TIMINGS_FILE).write_text(json.dumps({**timing, "steps": usage.steps}, indent=2))
    return prov


def _read_progress(logger, progress_callback, scan_id, k, n_scans, i, n, _seconds) -> None:
    """``linescan_roi_read``'s per-ROI callback: one log line, one progress tick.

    The line carries no elapsed time: the log's timestamps say when each ROI
    landed. The reads are the first 20 % of the run, scan ``k`` of
    ``n_scans``; the last ROI of the last scan lands on 0.2 exactly.
    """
    logger.info(f"scan {scan_id}: read ROI {i + 1}/{n}")
    if progress_callback is not None:
        progress_callback(0.2 * min(1.0, (k + (i + 1) / n) / n_scans), f"scan {scan_id}: read ROI {i + 1}/{n}")


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
    """One AOD ROI unit (line scan, chessboard, ribbon) as vnoiser's :class:`ScanTraces`.

    ``convert=False`` keeps MESc's raw counts, as the archive's converter
    did; ``True`` applies the file's linear conversion so zero means no
    photons (the dF/F then differs from the archive's by a slowly varying
    factor; the z-score is nearly unaffected). ``frames=(start, stop)``
    keeps that half-open window of the recording. ``rois`` (0-based ROI
    indices: the unit's Z axis, ``mesc_z_axis_meaning == "roi_index"``)
    reads only those ROIs; the traces stay keyed by the ROI's own index.
    ``progress(i, K, seconds)`` is called after each ROI when given.
    """
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, MescArray
    from mbo_utilities.roi_workflow import linescan_roi_read

    arr = MescArray(mesc_path, unit=unit_key)
    md = arr.metadata
    if md.get("mesc_layout") not in ROI_LAYOUTS:
        raise ValueError(f"{unit_key} is a {md.get('mesc_modality_name')} unit with no AOD ROIs")
    if rois is not None and md.get("mesc_z_axis_meaning") != "roi_index":
        raise ValueError(f"{unit_key}: Z means {md.get('mesc_z_axis_meaning')!r} here, not ROIs")
    F, _ = linescan_roi_read(
        arr, channel=channel, convert=convert, batch_size=batch_size, dtype=np.float64,
        progress=progress, rois=rois,
    )
    if frames is not None:
        start, stop = int(frames[0]), int(frames[1])
        if not 0 <= start < stop <= F.shape[1]:
            raise ValueError(f"frames {frames} outside 0..{F.shape[1]}")
        F = F[:, start:stop]
    extents = md["mesc_roi_extents"]
    if rois is not None:
        extents = [extents[int(r)] for r in rois]
    traces = {int(e["index"]): F[i] for i, e in enumerate(extents)}
    weights = {int(e["index"]): float(int(e["height"]) * int(e["width"])) for e in extents}
    return ScanTraces(
        _munit_number(md.get("mesc_unit", unit_key)), float(md["fs"]), traces, weights,
        comment=str(md.get("comment", "") or ""),
    )


def read_domains(path) -> dict:
    """``{"domains", "scan_ids", "first_env"}`` from a ``domains.json`` or an
    archive ``scanIDs_ROIs.pkl``.

    The JSON holds ``{"domains": {"soma1": [0, 1, 2], ...}, "scans": ["35",
    "38"], "first_env": ["35"]}``; ``scans`` and ``first_env`` are optional.
    """
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        rois = load_scan_rois(path)
        return {"domains": rois["domains"], "scan_ids": rois["scan_ids"], "first_env": rois["first_env"]}
    doc = json.loads(path.read_text())
    domains = {str(k): [int(v) for v in rois] for k, rois in doc["domains"].items()}
    if not domains:
        raise ValueError(f"{path} defines no domains")
    return {
        "domains": domains,
        "scan_ids": [str(s) for s in doc.get("scans", [])],
        "first_env": [str(s) for s in doc.get("first_env", [])],
    }


def write_domains_template(mesc_path, path=None, *, units=None, per_domain: int = 1) -> Path:
    """Write a ``domains.json`` to edit: the file's AOD ROI units as
    ``scans`` and one domain per ROI (``roi0: [0]``, ...; ``per_domain``
    groups consecutive ROIs instead, the archive's three lines per
    domain), named from the unit comment when it lists as many names
    (``'soma,bas1-3,api1-5'`` does not). Returns the path."""
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, list_mesc_units

    mesc_path = Path(mesc_path)
    path = Path(path) if path is not None else mesc_path.parent / DOMAINS_FILE
    scans = [u for u in list_mesc_units(mesc_path) if u.get("kind") in ROI_LAYOUTS]
    if units:
        wanted = {str(u) for u in units}
        scans = [u for u in scans if u["key"] in wanted or u["munit"] in wanted]
    if not scans:
        raise ValueError(f"no AOD ROI units in {mesc_path}")
    n_rois = int(scans[0]["nrois"])
    names = [n.strip() for n in str(scans[0].get("comment") or "").split(",") if n.strip()]
    groups = [list(range(i, min(i + per_domain, n_rois))) for i in range(0, n_rois, per_domain)]
    if len(names) != len(groups):
        names = [f"roi{g[0]}" if len(g) == 1 else f"domain{i + 1}" for i, g in enumerate(groups)]
    doc = {
        "mesc": mesc_path.name,
        "scans": [_munit_number(u["key"]) for u in scans],
        "first_env": [_munit_number(scans[0]["key"])],
        "domains": {name: rois for name, rois in zip(names, groups, strict=True)},
        "note": "domains: name -> ROI indices (0-based, the order the lines or patches were drawn); "
                "scans: MUnit numbers in order; first_env: the first scan of each environment",
    }
    path.write_text(json.dumps(doc, indent=2))
    return path


def default_pf_dir(mesc_path) -> Path:
    """Where the PF folder goes: ``<animal>/<expt>/PF`` for the archive's
    ``<animal>/<expt>/<expt>/<expt>.mesc`` layout, else ``PF`` beside the file."""
    mesc_path = Path(mesc_path)
    if mesc_path.parent.name == mesc_path.stem and mesc_path.parent.parent != mesc_path.parent:
        return mesc_path.parent.parent / "PF"
    return mesc_path.parent / "PF"


def run_voltage_pipeline(
    mesc_path,
    *,
    domains,
    units=None,
    first_env=(),
    out=None,
    channel: int = 0,
    convert: bool = False,
    frames=None,
    planes=None,
    save_cwt: bool = False,
    spike_cfg: SpikeDetectConfig | None = None,
    detect: bool = True,
    dfof_cfg: DfofConfig | None = None,
    denoiser_factory=None,
    settings: VoltageSettings | None = None,
    overwrite: bool = False,
    provenance: dict | None = None,
    progress_callback=None,
    logger: logging.Logger | None = None,
) -> dict:
    """Every AOD ROI unit of ``mesc_path`` (or ``units``) into a PF folder.

    Parameters
    ----------
    mesc_path : path
    domains : mapping of domain -> ROI indices
    units : sequence of str, optional
        ``MUnit_n`` or ``MSession_s/MUnit_n`` keys, in scan order; default
        every AOD ROI unit in file order. One run takes scans of one frame
        rate.
    first_env : sequence of scan ids
    out : path, optional
        The PF folder; default :func:`default_pf_dir`.
    channel, convert, frames : see :func:`scan_traces_from_mesc`
    planes : sequence of int, optional
        The pipeline input contract's 1-based Z selection. On an AOD ROI
        unit Z is the ROI index, so these are the ROIs (lines, patches) to
        process: only they are read, every domain is cut down to them and a
        domain left with no ROI is dropped. Default every ROI.
    settings : VoltageSettings, optional
        The Run tab's settings (default :class:`VoltageSettings`), scaled to
        the scans' frame rate with ``at_fs``; a ``dfof_cfg``,
        ``denoiser_factory`` or ``spike_cfg`` given explicitly is used as
        it is instead.
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
    ``domains.csv``, and per scan ``scan<id>_rois.npy`` (ROI, frame),
    ``scan<id>_dfof.npy`` / ``_zscore.npy`` / ``_denoised.npy`` (domain,
    frame) in ``domains.csv`` row order, ``scan<id>_peaks.csv``, and
    ``scan<id>_denoised.png`` / ``scan<id>_rois.png`` figures.
    ``pipeline.json`` carries the run's ``timing`` (wall, CPU and peak
    memory, totals per step, per scan and per domain, and the denoiser's
    stages ``cwt`` / ``cluster`` / ``reduce`` / ``mask`` / ``baseline`` /
    ``baseline_100hz`` / ``peaks`` summed over domains) and its
    ``processing_history`` (one ``voltage_<step>`` entry per step, the shape
    suite2p's ``ops.npy`` uses); ``timings.json`` beside it is the same with
    one flat row per step, a denoise row carrying its stages as ``<stage>_s``. With ``settings.runtime.output_format == "zarr"`` the pickles are
    replaced by one ``<yyyy-mm-dd>_<tags>.zarr`` results file in the folder
    (:mod:`mbo_utilities.results`; ``test.h5``, ``traces``, ``pipeline.json``
    and ``timings.json`` stay) whose ``provenance`` attr holds the same
    timing.
    """
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, list_mesc_units

    logger = logger or log.get()
    mesc_path = Path(mesc_path)
    all_units = list_mesc_units(mesc_path)
    roi_units = [u for u in all_units if u.get("kind") in ROI_LAYOUTS]
    roi_munits = {u["munit"] for u in roi_units}
    if units:
        chosen = []
        for name in units:
            match = [u for u in roi_units if u["key"] == name or u["munit"] == name]
            if not match:
                other = [u for u in all_units if u["key"] == name or (u["munit"] == name and name not in roi_munits)]
                if other:
                    raise ValueError(f"{name} is a {other[0]['modality_name']} unit with no AOD ROIs")
                raise ValueError(f"{name} is not in {mesc_path.name}")
            chosen.append(match[0])
    else:
        chosen = roi_units
    if not chosen:
        raise ValueError(f"no AOD ROI units in {mesc_path}")
    rates = sorted({round(float(u["fs"] or 0), 3) for u in chosen})
    if len(rates) > 1:
        listed = ", ".join(f"{r:g}" for r in rates)
        raise ValueError(f"the chosen scans differ in frame rate ({listed} Hz); run them separately")
    settings = (settings or VoltageSettings()).at_fs(float(chosen[0]["fs"]))
    if settings.runtime.output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {settings.runtime.output_format!r}")
    dfof_cfg = dfof_cfg or settings.dfof.config()
    spike_cfg = spike_cfg or settings.events.config()
    denoiser_factory = denoiser_factory or settings.denoiser.factory
    rois = None
    if planes is not None:
        rois = sorted({int(p) - 1 for p in planes})
        n_rois = min(int(u["nrois"]) for u in chosen)
        bad = [r + 1 for r in rois if not 0 <= r < n_rois]
        if bad:
            raise ValueError(f"planes {bad} are outside 1..{n_rois}, the scans' ROIs")
        kept = {name: [int(r) for r in rs if int(r) in rois] for name, rs in domains.items()}
        dropped = [name for name in domain_names(domains) if not kept[name]]
        domains = {name: rs for name, rs in kept.items() if rs}
        if not domain_names(domains):
            raise ValueError(f"no domain has an ROI among planes {[r + 1 for r in rois]}")
        if dropped:
            logger.info(f"voltage: planes {[r + 1 for r in rois]} leave no ROI in domain(s) {dropped}; dropped")
    names = domain_names(domains)
    pf_dir = Path(out) if out is not None else default_pf_dir(mesc_path)

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

    usage = _RunUsage(logger)
    system = mem_snapshot(usage.proc)
    logger.info(
        f"voltage: {len(chosen)} scan(s) x {len(names)} domain(s) -> {pf_dir} "
        f"({settings.runtime.output_format}); {os.cpu_count()} cpus, {system['total_gb']:.0f} GB RAM "
        f"({system['sys_pct']:.0f}% in use); vnoiser {versions['vnoiser']}, mbo_utilities {mbo_version}"
    )
    try:
        scans = []
        for k, u in enumerate(chosen):
            sid = _munit_number(u["key"])
            logger.info(f"scan {sid}: reading {u['key']} ({u['nrois']} ROIs x {u['nframes']} frames at {u['fs']:.2f} Hz)")
            usage.begin("read", scan=sid, unit=u["key"])
            scan = scan_traces_from_mesc(
                mesc_path, u["key"], channel=channel, convert=convert, frames=frames, rois=rois,
                progress=partial(_read_progress, logger, progress_callback, sid, k, len(chosen)),
            )
            scans.append(scan)
            usage.end(
                f"scan {sid}: read {len(scan.traces)} ROIs x {scan.n_frames} frames",
                n_rois=len(scan.traces), n_frames=scan.n_frames,
            )

        example = denoiser_factory(scans[0].fs_hz)
        info = {
            "pipeline": "mbo_utilities.vnoiser.pipeline.run_voltage_pipeline",
            "versions": versions,
            "dfof": asdict(dfof_cfg),
            "denoiser": example.describe(),
            "events": asdict(spike_cfg) if detect else None,
            "comments": {s.scan_id: s.comment for s in scans if s.comment},
            "roi_weights": {s.scan_id: {int(k): float(v) for k, v in s.weights.items()} for s in scans},
            "source": {
                "mesc": str(mesc_path),
                "units": {s.scan_id: u["key"] for s, u in zip(scans, chosen, strict=True)},
                "channel": int(channel),
                "convert": bool(convert),
                "frames": None if frames is None else [int(frames[0]), int(frames[1])],
                "planes": None if planes is None else [int(p) for p in planes],
                "mbo_utilities": mbo_version,
            },
        }
        info.update(provenance or {})
        processed = {s.scan_id for s in scans}
        first_env = [str(s) for s in first_env if str(s) in processed]
        writer = PfWriter(
            pf_dir,
            domains=domains,
            scan_ids=[s.scan_id for s in scans],
            first_env=first_env,
            roi_list={s.scan_id: s.rois for s in scans},
            spike_params=spike_cfg.to_param_pickle() if detect else None,
            save_cwt=save_cwt,
            freq_scales=example.freq_scales,
            provenance=info,
            overwrite=overwrite,
        )
        n_domains = max(len(names), 1)
        for k, scan in enumerate(scans):
            sid = scan.scan_id
            # one domain at a time: each row of dF/F and z is independent of the others, and the
            # baseline filters take about ten seconds per domain on a five-minute scan
            dfof_rows, z_rows = [], []
            for row, name in enumerate(names):
                if progress_callback is not None:
                    # dF/F is the next 30 % of the run, scan k of the run, domain row of the scan
                    fraction = 0.2 + 0.3 * min(1.0, (k + row / n_domains) / len(scans))
                    progress_callback(fraction, f"scan {sid}: dF/F of {name} ({row + 1}/{len(names)})")
                usage.begin("dfof", scan=sid, domain=name)
                one = domain_zscore(scan.traces, {name: domains[name]}, scan.weights, dfof_cfg)
                dfof_rows.append(one.dfof_raw[0])
                z_rows.append(one.z[0])
                usage.end(f"scan {sid}: dF/F and z-score of {name} ({row + 1}/{len(names)})")
            traces = DomainTraces(names=list(names), dfof_raw=np.stack(dfof_rows), z=np.stack(z_rows))
            writer.add_scan(sid, scan.fs_hz, traces.dfof_raw, traces.z)
            for row, name in enumerate(traces.names):
                if progress_callback is not None:
                    # the denoising is the next 40 %
                    fraction = 0.5 + 0.4 * min(1.0, (k + row / n_domains) / len(scans))
                    progress_callback(fraction, f"scan {sid}: denoising {name} ({row + 1}/{len(names)})")
                logger.info(f"scan {sid}: denoising {name} ({row + 1}/{len(names)})")
                usage.begin("denoise", scan=sid, domain=name)
                result = process_domain(
                    traces.z[row], scan.fs_hz, denoiser=denoiser_factory(scan.fs_hz),
                    spike_cfg=spike_cfg, detect=detect, keep_cwt=save_cwt,
                )
                writer.add_domain(sid, name, result)
                result.cwt = None
                n_events = None if result.peaks is None else int(len(result.peaks))
                events = "" if n_events is None else f", {n_events} events"
                stages = {stage: float(seconds) for stage, seconds in (result.timing or {}).items()}
                inside = ", ".join(f"{stage} {seconds:.1f}" for stage, seconds in stages.items())
                usage.end(
                    f"scan {sid}: denoised {name} ({row + 1}/{len(names)}){events} [{inside} s]",
                    n_events=n_events, **{f"{stage}_s": round(seconds, 3) for stage, seconds in stages.items()},
                )

        if progress_callback is not None:
            progress_callback(0.9, f"writing {pf_dir.name}")
        usage.begin("write_pf")
        paths = writer.finish()
        usage.end(f"wrote {len(paths)} PF files to {pf_dir}")
        files = read_pf(pf_dir)
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
                rows.writerow([i, final_domain_name(name), " ".join(str(r) for r in domains[name])])
        with h5py.File(pf_dir / DFOF_FILE, "r") as f:
            for s in scans:
                sid = s.scan_id
                rois = np.stack([np.asarray(s.traces[r]) for r in sorted(s.traces)])
                np.save(traces_dir / f"scan{sid}_rois.npy", rois.astype(np.float32))
                np.save(traces_dir / f"scan{sid}_dfof.npy", f[sid]["dfof_raw"][:].astype(np.float32))
                np.save(traces_dir / f"scan{sid}_zscore.npy", f[sid]["dfof_zscore"][:].astype(np.float32))
                denoised = np.stack([files.traces[sid][final_domain_name(n)] for n in names])
                np.save(traces_dir / f"scan{sid}_denoised.npy", denoised.astype(np.float32))
                with (traces_dir / f"scan{sid}_peaks.csv").open("w", newline="") as fh:
                    rows = csv.writer(fh)
                    rows.writerow(["domain", "frame", "time_s"])
                    for n in names:
                        for frame in (files.peaks or {}).get(sid, {}).get(final_domain_name(n), ()):
                            rows.writerow([final_domain_name(n), int(frame), f"{int(frame) / s.fs_hz:.6f}"])
        usage.end(f"wrote {TRACES_DIR}/ for {len(scans)} scan(s)")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        usage.begin("figures")
        for s in scans:
            sid = s.scan_id
            t = np.arange(s.n_frames) / s.fs_hz
            denoised = np.load(traces_dir / f"scan{sid}_denoised.npy")
            fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.6 * len(names) + 0.8), sharex=True, squeeze=False)
            for ax, n, trace in zip(axes[:, 0], names, denoised, strict=True):
                ax.plot(t, trace, lw=0.5, color="k")
                frames = (files.peaks or {}).get(sid, {}).get(final_domain_name(n), ())
                if len(frames):
                    ax.plot(t[frames], trace[frames], ".", color="r", ms=4)
                ax.set_ylabel(final_domain_name(n), rotation=0, ha="right", va="center")
                ax.spines[["top", "right"]].set_visible(False)
            axes[-1, 0].set_xlabel("s")
            axes[0, 0].set_title(f"scan {sid} denoised")
            fig.tight_layout()
            fig.savefig(traces_dir / f"scan{sid}_denoised.png", dpi=120)
            plt.close(fig)
            rois = np.load(traces_dir / f"scan{sid}_rois.npy")
            fig, ax = plt.subplots(figsize=(12, 0.5 * len(rois) + 1.2))
            for i, trace in enumerate(rois):
                span = np.ptp(trace) or 1.0
                ax.plot(t, (trace - trace.mean()) / span - i, lw=0.4, color="k")
            ax.set_yticks(-np.arange(len(rois)), [str(i) for i in range(len(rois))])
            ax.set_ylabel("ROI")
            ax.set_xlabel("s")
            ax.set_title(f"scan {sid} raw ROIs")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(traces_dir / f"scan{sid}_rois.png", dpi=120)
            plt.close(fig)
        usage.end(f"wrote {2 * len(scans)} figures")
        paths.update({f"{TRACES_DIR}/{p.name}": p for p in sorted(traces_dir.iterdir())})
        prov = _write_timing(pf_dir, usage)
        if settings.runtime.output_format == "zarr":
            from mbo_utilities.results import results_from_pf, results_name, write_results

            if progress_callback is not None:
                progress_callback(0.97, "writing the results zarr")
            usage.begin("results")
            result_units, root = results_from_pf(pf_dir)
            results_path = write_results(pf_dir / results_name(mesc_path), result_units, overwrite=True, **root)
            # the results file replaces the archive's pickles; the h5, the traces folder and pipeline.json stay
            for name in PIPELINE_FILES:
                if name.endswith(".pkl"):
                    (pf_dir / name).unlink(missing_ok=True)
                    paths.pop(name, None)
            paths[results_path.name] = results_path
            usage.end(f"wrote {results_path.name}")
            prov = _write_timing(pf_dir, usage)
            zarr.open_group(str(results_path), mode="r+").attrs["provenance"] = prov
        paths[TIMINGS_FILE] = pf_dir / TIMINGS_FILE
        timing = prov["timing"]
        wall, cpu = timing["wall_seconds"], timing["cpu_seconds"]
        totals = ", ".join(f"{step} {seconds:.1f} s" for step, seconds in timing["totals"].items())
        stages = ", ".join(f"{stage} {seconds:.1f} s" for stage, seconds in timing["denoise_stages"].items())
        logger.info(
            f"voltage done in {int(wall // 60)}m {wall % 60:04.1f}s (cpu {cpu:.0f} s, {cpu / max(wall, 1e-9):.1f} cores); "
            f"peak process memory {timing['peak_rss_gb']:.2f} GB; {totals}; denoise stages: {stages}; "
            f"timings in {pf_dir / TIMINGS_FILE}"
        )
        if progress_callback is not None:
            progress_callback(1.0, f"wrote {len(paths)} files to {pf_dir}")
        return paths
    finally:
        usage.close()
