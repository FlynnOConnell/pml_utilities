"""The spatial JEDI voltage pipeline on a raw AOD ``.mesc``.

Each unit with AOD ROIs (:data:`mbo_utilities.arrays.mesc.ROI_LAYOUTS`: the
lines of a line scan, the patches of a chessboard, the boxes of a ribbon
scan) is one scan; each ROI's mean fluorescence per frame
(``roi_workflow.linescan_roi_read``, the same read ``mbo linescan`` does)
goes into vnoiser's stages 0-5 (:func:`vnoiser.run_pipeline`): domains are
pixel-weighted means of their ROIs, dF/F, z-score, wavelet denoising, peaks,
and a ``PF`` folder that ``mbo curate`` opens. Domains (which ROIs make the
soma, each branch; which patch is which cell) come from a ``domains.json``
beside the file, or an archive's ``scanIDs_ROIs.pkl``. Settings are written
for the archive's frame rate and scaled to the scans'
(:meth:`~mbo_utilities.vnoiser.params.VoltageSettings.at_fs`).

Reproduces the archive: ``stan112_expt12/stan112_expt12/stan112_expt12.mesc``
units 35 and 38, read this way with ``convert=False``, give the per-ROI
traces the lab's ``VI_2025-07-24.pkl`` holds to float precision, and the
traces of its ``PF`` folder follow (see ``tests/test_voltage_pipeline.py``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from vnoiser import DfofConfig, ScanTraces, SpikeDetectConfig, read_pf, run_pipeline
from vnoiser.pf import DFOF_FILE, final_domain_name
from vnoiser.pipeline import load_scan_rois
from vnoiser.preprocess import domain_names

from mbo_utilities.vnoiser.params import VoltageSettings

__all__ = [
    "DOMAINS_FILE",
    "scan_traces_from_mesc",
    "read_domains",
    "write_domains_template",
    "default_pf_dir",
    "run_voltage_pipeline",
    "TRACES_DIR",
]

DOMAINS_FILE = "domains.json"
TRACES_DIR = "traces"


def _munit_number(unit_key: str) -> str:
    return str(unit_key).rsplit("/", 1)[-1].rsplit("_", 1)[-1]


def scan_traces_from_mesc(
    mesc_path,
    unit_key: str,
    *,
    channel: int = 0,
    convert: bool = False,
    frames=None,
    batch_size: int = 20000,
    progress=None,
) -> ScanTraces:
    """One AOD ROI unit (line scan, chessboard, ribbon) as vnoiser's :class:`ScanTraces`.

    ``convert=False`` keeps MESc's raw counts, as the archive's converter
    did; ``True`` applies the file's linear conversion so zero means no
    photons (the dF/F then differs from the archive's by a slowly varying
    factor; the z-score is nearly unaffected). ``frames=(start, stop)``
    keeps that half-open window of the recording.
    """
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, MescArray
    from mbo_utilities.roi_workflow import linescan_roi_read

    arr = MescArray(mesc_path, unit=unit_key)
    md = arr.metadata
    if md.get("mesc_layout") not in ROI_LAYOUTS:
        raise ValueError(f"{unit_key} is a {md.get('mesc_modality_name')} unit with no AOD ROIs")
    F, _ = linescan_roi_read(
        arr, channel=channel, convert=convert, batch_size=batch_size, dtype=np.float64,
        progress=progress,
    )
    if frames is not None:
        start, stop = int(frames[0]), int(frames[1])
        if not 0 <= start < stop <= F.shape[1]:
            raise ValueError(f"frames {frames} outside 0..{F.shape[1]}")
        F = F[:, start:stop]
    extents = md["mesc_roi_extents"]
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


def write_domains_template(mesc_path, path=None, *, units=None, per_domain: int | None = None) -> Path:
    """Write a ``domains.json`` to fill in: the file's AOD ROI units as
    ``scans`` and their ROIs grouped ``per_domain`` at a time (three lines of
    a line scan, one patch of a chessboard or ribbon scan), named from the
    unit comment when it lists as many names (``'soma,bas1-3,api1-5'`` does
    not; those stay ``domain1``...). Returns the path."""
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, list_mesc_units

    mesc_path = Path(mesc_path)
    path = Path(path) if path is not None else mesc_path.parent / DOMAINS_FILE
    scans = [u for u in list_mesc_units(mesc_path) if u.get("kind") in ROI_LAYOUTS]
    if units:
        wanted = {str(u) for u in units}
        scans = [u for u in scans if u["key"] in wanted or u["munit"] in wanted]
    if not scans:
        raise ValueError(f"no AOD ROI units in {mesc_path}")
    if per_domain is None:
        per_domain = 3 if scans[0]["kind"] == "packed" else 1
    n_rois = int(scans[0]["nrois"])
    names = [n.strip() for n in str(scans[0].get("comment") or "").split(",") if n.strip()]
    groups = [list(range(i, min(i + per_domain, n_rois))) for i in range(0, n_rois, per_domain)]
    if len(names) != len(groups):
        names = [f"domain{i + 1}" for i in range(len(groups))]
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
    save_cwt: bool = False,
    spike_cfg: SpikeDetectConfig | None = None,
    detect: bool = True,
    dfof_cfg: DfofConfig | None = None,
    denoiser_factory=None,
    settings: VoltageSettings | None = None,
    overwrite: bool = False,
    provenance: dict | None = None,
    progress=None,
    log=print,
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
    settings : VoltageSettings, optional
        The Run tab's settings (default :class:`VoltageSettings`), scaled to
        the scans' frame rate with ``at_fs``; a ``dfof_cfg``,
        ``denoiser_factory`` or ``spike_cfg`` given explicitly is used as
        it is instead.
    save_cwt, spike_cfg, detect, dfof_cfg, denoiser_factory, overwrite :
        see :func:`vnoiser.run_pipeline`
    provenance : dict, optional
        Extra keys for ``pipeline.json`` beside the source block.
    progress : callable(scan_id, domain), optional
    log : callable(str)
        Where the per-unit read is reported.

    Returns ``{file name: path}`` of the written folder. Beside the archive
    format a ``traces`` subfolder holds plain files: ``scans.csv``,
    ``domains.csv``, and per scan ``scan<id>_rois.npy`` (ROI, frame),
    ``scan<id>_dfof.npy`` / ``_zscore.npy`` / ``_denoised.npy`` (domain,
    frame) in ``domains.csv`` row order, ``scan<id>_peaks.csv``.
    """
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, list_mesc_units

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
    dfof_cfg = dfof_cfg or settings.dfof.config()
    spike_cfg = spike_cfg or settings.events.config()
    denoiser_factory = denoiser_factory or settings.denoiser.factory

    scans = []
    for u in chosen:
        log(f"reading {u['key']} ({u['nrois']} ROIs x {u['nframes']} frames at {u['fs']:.2f} Hz)")
        scans.append(scan_traces_from_mesc(
            mesc_path, u["key"], channel=channel, convert=convert, frames=frames,
        ))
    from importlib.metadata import PackageNotFoundError, version

    mbo_version = None
    for dist in ("pml_utilities", "mbo_utilities"):
        try:
            mbo_version = version(dist)
            break
        except PackageNotFoundError:
            continue
    info = {
        "source": {
            "mesc": str(mesc_path),
            "units": {s.scan_id: u["key"] for s, u in zip(scans, chosen, strict=True)},
            "channel": int(channel),
            "convert": bool(convert),
            "frames": None if frames is None else [int(frames[0]), int(frames[1])],
            "mbo_utilities": mbo_version,
        }
    }
    info.update(provenance or {})
    pf_dir = Path(out) if out is not None else default_pf_dir(mesc_path)
    processed = {s.scan_id for s in scans}
    first_env = [str(s) for s in first_env if str(s) in processed]
    paths = run_pipeline(
        scans, pf_dir, domains=domains, first_env=first_env, dfof_cfg=dfof_cfg,
        denoiser_factory=denoiser_factory, spike_cfg=spike_cfg,
        detect=detect, save_cwt=save_cwt, provenance=info, overwrite=overwrite,
        progress=progress,
    )
    files = read_pf(pf_dir)
    names = domain_names(domains)
    traces_dir = pf_dir / TRACES_DIR
    traces_dir.mkdir(exist_ok=True)
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
    paths.update({f"{TRACES_DIR}/{p.name}": p for p in sorted(traces_dir.iterdir())})
    return paths
