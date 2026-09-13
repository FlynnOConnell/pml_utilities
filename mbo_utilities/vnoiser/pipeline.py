"""The spatial JEDI voltage pipeline on a raw line-scan ``.mesc``.

Each line-scan unit is one scan; each line ROI's mean fluorescence per frame
(``roi_workflow.linescan_roi_read``, the same read ``mbo linescan`` does)
goes into vnoiser's stages 0-5 (:func:`vnoiser.run_pipeline`): domains are
pixel-weighted means of their ROIs, dF/F, z-score, wavelet denoising, peaks,
and a ``PF`` folder that ``mbo curate`` opens. Domains (which ROIs make the
soma, each branch) come from a ``domains.json`` beside the file, or an
archive's ``scanIDs_ROIs.pkl``.

Reproduces the archive: ``stan112_expt12/stan112_expt12/stan112_expt12.mesc``
units 35 and 38, read this way with ``convert=False``, give the per-ROI
traces the lab's ``VI_2025-07-24.pkl`` holds to float precision, and the
traces of its ``PF`` folder follow (see ``tests/test_voltage_pipeline.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from vnoiser import Denoiser, DfofConfig, ScanTraces, SpikeDetectConfig, run_pipeline
from vnoiser.pipeline import load_scan_rois

__all__ = [
    "DOMAINS_FILE",
    "scan_traces_from_mesc",
    "read_domains",
    "write_domains_template",
    "default_pf_dir",
    "run_voltage_pipeline",
]

DOMAINS_FILE = "domains.json"


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
    """One line-scan unit as vnoiser's :class:`ScanTraces`.

    ``convert=False`` keeps MESc's raw counts, as the archive's converter
    did; ``True`` applies the file's linear conversion so zero means no
    photons (the dF/F then differs from the archive's by a slowly varying
    factor; the z-score is nearly unaffected). ``frames=(start, stop)``
    keeps that half-open window of the recording.
    """
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.roi_workflow import linescan_roi_read

    arr = MescArray(mesc_path, unit=unit_key)
    md = arr.metadata
    if md.get("mesc_layout") != "packed":
        raise ValueError(f"{unit_key} is a {md.get('mesc_modality_name')} unit, not a line scan")
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


def write_domains_template(mesc_path, path=None, *, units=None, per_domain: int = 3) -> Path:
    """Write a ``domains.json`` to fill in: the file's line-scan units as
    ``scans`` and their ROIs grouped ``per_domain`` at a time, named from
    the unit comment when it lists as many names (``'soma,bas1-3,api1-5'``
    does not; those stay ``domain1``...). Returns the path."""
    from mbo_utilities.arrays.mesc import list_mesc_units

    mesc_path = Path(mesc_path)
    path = Path(path) if path is not None else mesc_path.parent / DOMAINS_FILE
    packed = [u for u in list_mesc_units(mesc_path) if u.get("kind") == "packed"]
    if units:
        wanted = {str(u) for u in units}
        packed = [u for u in packed if u["key"] in wanted or u["munit"] in wanted]
    if not packed:
        raise ValueError(f"no line-scan units in {mesc_path}")
    n_rois = int(packed[0]["nrois"])
    names = [n.strip() for n in str(packed[0].get("comment") or "").split(",") if n.strip()]
    groups = [list(range(i, min(i + per_domain, n_rois))) for i in range(0, n_rois, per_domain)]
    if len(names) != len(groups):
        names = [f"domain{i + 1}" for i in range(len(groups))]
    doc = {
        "mesc": mesc_path.name,
        "scans": [_munit_number(u["key"]) for u in packed],
        "first_env": [_munit_number(packed[0]["key"])],
        "domains": {name: rois for name, rois in zip(names, groups, strict=True)},
        "note": "domains: name -> ROI indices (0-based, the order the lines were drawn); "
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
    overwrite: bool = False,
    provenance: dict | None = None,
    progress=None,
    log=print,
) -> dict:
    """Every line-scan unit of ``mesc_path`` (or ``units``) into a PF folder.

    Parameters
    ----------
    mesc_path : path
    domains : mapping of domain -> ROI indices
    units : sequence of str, optional
        ``MUnit_n`` or ``MSession_s/MUnit_n`` keys, in scan order; default
        every line-scan unit in file order.
    first_env : sequence of scan ids
    out : path, optional
        The PF folder; default :func:`default_pf_dir`.
    channel, convert, frames : see :func:`scan_traces_from_mesc`
    save_cwt, spike_cfg, detect, dfof_cfg, denoiser_factory, overwrite :
        see :func:`vnoiser.run_pipeline`
    provenance : dict, optional
        Extra keys for ``pipeline.json`` beside the source block.
    progress : callable(scan_id, domain), optional
    log : callable(str)
        Where the per-unit read is reported.

    Returns ``{file name: path}`` of the written folder.
    """
    from mbo_utilities.arrays.mesc import list_mesc_units

    mesc_path = Path(mesc_path)
    all_units = list_mesc_units(mesc_path)
    packed = [u for u in all_units if u.get("kind") == "packed"]
    packed_munits = {u["munit"] for u in packed}
    if units:
        chosen = []
        for name in units:
            match = [u for u in packed if u["key"] == name or u["munit"] == name]
            if not match:
                other = [u for u in all_units if u["key"] == name or (u["munit"] == name and name not in packed_munits)]
                if other:
                    raise ValueError(f"{name} is a {other[0]['modality_name']} unit, not a line scan")
                raise ValueError(f"{name} is not in {mesc_path.name}")
            chosen.append(match[0])
    else:
        chosen = packed
    if not chosen:
        raise ValueError(f"no line-scan units in {mesc_path}")

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
    return run_pipeline(
        scans, pf_dir, domains=domains, first_env=first_env, dfof_cfg=dfof_cfg,
        denoiser_factory=denoiser_factory or Denoiser.upstream, spike_cfg=spike_cfg,
        detect=detect, save_cwt=save_cwt, provenance=info, overwrite=overwrite,
        progress=progress,
    )
