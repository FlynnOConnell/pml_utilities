"""One results file for every pipeline: a zarr v3 group named
``<input stem>.<yyyy-mm-dd-HH-MM-SS>.<pipeline>.zarr``.

A pipeline's native outputs (suite2p's ``F.npy`` and ``stat.npy``, masknmf's
suite2p-shaped plane dirs, the voltage pipeline's ``PF`` pickles) differ in
every detail; this module fixes one shape they all mold into so a reader,
a viewer or a notebook opens any of them the same way. The contract is
AGENTS.md §7.5; the schema is::

    <stem>.<stamp>.<pipeline>.zarr/     zarr v3 group
      attrs: mbo_results, pipeline, created, tags, units, source, settings,
             metadata, provenance
      <unit>/                           one group per plane (zplane01) or scan (scan35)
        attrs: kind, index, fs, n_rois, n_timepoints, roi_names, member_kind,
               image_shape, plus anything the pipeline adds
        traces/<kind>                   (n_rois, n_timepoints) float32, TRACE_KINDS
        rois/offsets rois/member rois/weight
                                        ragged ROI membership: ROI k is
                                        member[offsets[k]:offsets[k+1]], pixels
                                        (flat y * X + x) or line indices
        rois/iscell                     (n_rois, 2) float32, suite2p's accept flag
        members/<kind>                  (n_members, n_timepoints) float32, the members'
                                        own traces when they have them (a line scan's lines)
        events/frame events/roi         detected events, (n_events,), sorted by ROI
      _sidecar/                         the pipeline's own files (pipeline.json,
                                        timings.json, its native h5 and npy)
        images/<kind>                   (Y, X) float32 summary images, IMAGE_KINDS

:func:`results_from_suite2p` molds suite2p and masknmf output folders,
:func:`results_from_pf` the voltage pipeline's ``PF`` folder; a new pipeline
builds :class:`ResultUnit` objects and calls :func:`write_results`.

:func:`results_name` names the file after its input so it sits beside it,
:func:`results_stamp` reads the timestamp back with ``datetime.strptime``
and :func:`newest_results` picks the latest run in a folder. Nothing matches
a results name by pattern: ask those three.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

from mbo_utilities import log
from mbo_utilities._writers import _make_json_serializable
from mbo_utilities.arrays.features._dim_tags import TAG_REGISTRY, filename_tags
from mbo_utilities.metadata.base import strip_for_export

__all__ = [
    "IMAGE_KINDS",
    "MEMBER_KINDS",
    "RESULTS_ATTR",
    "RESULTS_VERSION",
    "SIDECAR",
    "TRACE_KINDS",
    "UNIT_KINDS",
    "ResultUnit",
    "Results",
    "newest_results",
    "read_results",
    "results_from_pf",
    "results_from_suite2p",
    "results_name",
    "results_pipeline",
    "results_stamp",
    "results_summary",
    "unit_name",
    "write_results",
]

logger = log.get("results")

RESULTS_VERSION = 1
RESULTS_ATTR = "mbo_results"
# a pipeline's own files live inside the results file, beside its groups
SIDECAR = "_sidecar"
# strftime both ways; results_stamp parses it, nothing matches it by pattern
RESULTS_STAMP = "%Y-%m-%d-%H-%M-%S"
TRACE_KINDS = {
    "raw": "fluorescence in the recording's units (suite2p F; a line's mean counts)",
    "neuropil": "neuropil fluorescence (suite2p Fneu)",
    "dff": "dF/F (masknmf norm_traces in percent; the voltage pipeline's dfof_raw)",
    "zscore": "z-scored dF/F",
    "denoised": "denoised trace (the voltage pipeline's curated trace)",
    "spikes": "deconvolved activity (suite2p spks)",
}
IMAGE_KINDS = {
    "mean": "mean image (suite2p meanImg)",
    "max": "max projection (suite2p max_proj)",
    "corr": "correlation image (suite2p Vcorr)",
    "ref": "registration reference (suite2p refImg)",
}
MEMBER_KINDS = ("pixel", "line")
UNIT_KINDS = ("plane", "scan")
_UNIT_ATTRS = ("kind", "index", "fs", "n_rois", "n_timepoints", "roi_names", "member_kind", "image_shape")
_ROOT_ATTRS = (RESULTS_ATTR, "pipeline", "created", "tags", "units", "source", "settings", "metadata", "provenance")
_CHUNK = (256, 8192)


@dataclass
class ResultUnit:
    """One plane or scan of a results file: ``n_rois`` ROIs with ``(n_rois, n_timepoints)`` traces.

    ``members[k]`` lists ROI ``k``'s members (flat pixel indices into
    ``image_shape``, or line indices) with ``weights[k]`` their weights;
    ``member_traces`` holds the members' own traces when they have them.
    ``events`` maps an ROI name to the frames of its detected events.
    """

    name: str
    kind: str
    index: int
    fs: float | None = None
    roi_names: list[str] = field(default_factory=list)
    traces: dict[str, np.ndarray] = field(default_factory=dict)
    member_kind: str = "pixel"
    members: list[np.ndarray] = field(default_factory=list)
    weights: list[np.ndarray] | None = None
    image_shape: tuple[int, int] | None = None
    iscell: np.ndarray | None = None
    member_traces: dict[str, np.ndarray] = field(default_factory=dict)
    events: dict[str, np.ndarray] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    @property
    def n_rois(self) -> int:
        return len(self.roi_names)

    @property
    def n_timepoints(self) -> int:
        for arr in (*self.traces.values(), *self.member_traces.values()):
            return int(np.shape(arr)[1])
        return 0

    def member_roi(self, member) -> int | None:
        """The index of the ROI whose members include ``member``, or None."""
        for k, members in enumerate(self.members):
            if member in members:
                return k
        return None


@dataclass
class Results:
    """A results file read back, every array in memory."""

    path: Path
    pipeline: str
    created: str
    tags: list[str]
    source: dict
    settings: dict
    metadata: dict
    provenance: dict
    units: dict[str, ResultUnit]

    def __getitem__(self, name: str) -> ResultUnit:
        return self.units[name]


def unit_name(kind: str, index: int) -> str:
    """``zplane01`` for plane 1 (the Z tag of the filename system), ``scan35`` for scan 35."""
    if kind == "plane":
        definition = TAG_REGISTRY["Z"]
        return f"{definition.label}{int(index):0{definition.zero_pad}d}"
    if kind == "scan":
        return f"scan{int(index)}"
    raise ValueError(f"kind must be one of {UNIT_KINDS}, got {kind!r}")


def _clean(part: str) -> str:
    """``part`` as one dot-separated field of a results filename."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", part).strip("_")


def results_name(source, when: datetime | None = None, extra_tags=(), pipeline: str = "") -> str:
    """``<stem>.<yyyy-mm-dd-HH-MM-SS>.<pipeline>.zarr`` for ``source``.

    The stem is the source filename's, so the file sits beside its input
    under the input's own name (``session1.mesc`` ->
    ``session1.2026-09-21-14-30-22.voltage.zarr``); ``extra_tags`` follow it.
    The timestamp is local time to the second, so two runs in a day are two
    files and a listing sorts chronologically; :func:`results_stamp` reads it
    back with ``datetime.strptime``.
    """
    when = when or datetime.now()
    name = Path(source).name
    stem = name[: -len(".zarr")] if name.endswith(".zarr") else Path(name).stem
    parts = [_clean(stem) or "results"]
    parts += [c for c in (_clean(str(t)) for t in extra_tags) if c]
    parts.append(f"{when:{RESULTS_STAMP}}")
    if pipeline:
        parts.append(_clean(str(pipeline)))
    return ".".join(parts) + ".zarr"


def results_stamp(path) -> datetime | None:
    """When a results file was written, from the timestamp in its name, or
    None when the name carries none."""
    for part in Path(path).name.split("."):
        try:
            return datetime.strptime(part, RESULTS_STAMP)
        except ValueError:
            continue
    return None


def newest_results(folder, pipeline: str | None = None) -> Path | None:
    """The newest results file directly in ``folder`` by the timestamp in its
    name, limited to one ``pipeline`` when given; None when there is none.

    A file named before this convention has no timestamp and sorts oldest.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    found = [
        p
        for p in folder.glob("*.zarr")
        if (kind := results_pipeline(p)) is not None and pipeline in (None, kind)
    ]
    if not found:
        return None
    return max(found, key=lambda p: (results_stamp(p) or datetime.min, p.name))


def results_pipeline(path) -> str | None:
    """The ``pipeline`` attr of a results file, or None when ``path`` is not one.

    Reads only ``zarr.json``, so it is cheap enough for ``can_open``.
    """
    meta = Path(path) / "zarr.json"
    if not meta.is_file():
        return None
    try:
        doc = json.loads(meta.read_text())
    except (OSError, ValueError):
        return None
    attrs = doc.get("attributes") or {}
    if doc.get("node_type") != "group" or RESULTS_ATTR not in attrs:
        return None
    return str(attrs.get("pipeline", ""))


def results_summary(path) -> dict | None:
    """``{"pipeline", "units", "n_rois", "created"}`` of a results file from its
    ``zarr.json`` files alone (no arrays opened), or None when ``path`` is not one."""
    path = Path(path)
    pipeline = results_pipeline(path)
    if pipeline is None:
        return None
    attrs = json.loads((path / "zarr.json").read_text()).get("attributes") or {}
    units = [str(u) for u in attrs.get("units") or []]
    n_rois = 0
    for name in units:
        meta = path / name / "zarr.json"
        if meta.is_file():
            n_rois += int((json.loads(meta.read_text()).get("attributes") or {}).get("n_rois") or 0)
    return {"pipeline": pipeline, "units": units, "n_rois": n_rois, "created": str(attrs.get("created", ""))}


def write_results(
    path,
    units,
    *,
    pipeline: str,
    source: dict | None = None,
    settings: dict | None = None,
    metadata: dict | None = None,
    provenance: dict | None = None,
    tags=None,
    overwrite: bool = False,
) -> Path:
    """Write ``units`` as one results file; returns its path.

    ``source`` says what was processed (path, units or planes, reader
    kwargs), ``settings`` is the pipeline's ``Settings.to_dict()``,
    ``metadata`` the source array's metadata (stripped for export) and
    ``provenance`` whatever else the pipeline recorded. ``tags`` default to
    the ones in the file's own name.
    """
    path = Path(path)
    units = list(units)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True")
    if not units:
        raise ValueError("no units to write")
    names = [u.name for u in units]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate unit names: {names}")
    for unit in units:
        if unit.kind not in UNIT_KINDS:
            raise ValueError(f"{unit.name}: kind must be one of {UNIT_KINDS}, got {unit.kind!r}")
        if unit.member_kind not in MEMBER_KINDS:
            raise ValueError(f"{unit.name}: member_kind must be one of {MEMBER_KINDS}, got {unit.member_kind!r}")
        k, t = unit.n_rois, unit.n_timepoints
        for kind, arr in unit.traces.items():
            if kind not in TRACE_KINDS:
                raise ValueError(f"{unit.name}: unknown trace kind {kind!r}; use one of {sorted(TRACE_KINDS)}")
            if np.shape(arr) != (k, t):
                raise ValueError(f"{unit.name}: traces/{kind} is {np.shape(arr)}, expected ({k}, {t})")
        for kind, arr in unit.member_traces.items():
            if kind not in TRACE_KINDS:
                raise ValueError(f"{unit.name}: unknown trace kind {kind!r}; use one of {sorted(TRACE_KINDS)}")
            if np.ndim(arr) != 2 or np.shape(arr)[1] != t:
                raise ValueError(f"{unit.name}: members/{kind} is {np.shape(arr)}, expected (n_members, {t})")
        if len(unit.members) != k:
            raise ValueError(f"{unit.name}: {len(unit.members)} member lists for {k} ROIs")
        if unit.weights is not None and [len(w) for w in unit.weights] != [len(m) for m in unit.members]:
            raise ValueError(f"{unit.name}: weights do not match members")
        if unit.iscell is not None and np.shape(unit.iscell) != (k, 2):
            raise ValueError(f"{unit.name}: iscell is {np.shape(unit.iscell)}, expected ({k}, 2)")
        for kind, arr in unit.images.items():
            if kind not in IMAGE_KINDS:
                raise ValueError(f"{unit.name}: unknown image kind {kind!r}; use one of {sorted(IMAGE_KINDS)}")
            if np.ndim(arr) != 2:
                raise ValueError(f"{unit.name}: images/{kind} must be (Y, X), got {np.shape(arr)}")
        unknown = [r for r in unit.events if r not in unit.roi_names]
        if unknown:
            raise ValueError(f"{unit.name}: events for unknown ROIs {unknown}")
        clash = [a for a in unit.attrs if a in _UNIT_ATTRS]
        if clash:
            raise ValueError(f"{unit.name}: attrs {clash} are written by the schema")

    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            RESULTS_ATTR: RESULTS_VERSION,
            "pipeline": str(pipeline),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tags": [str(t) for t in (tags if tags is not None else (x.to_string() for x in filename_tags(path)))],
            "units": names,
            "source": _make_json_serializable(dict(source or {})),
            "settings": _make_json_serializable(dict(settings or {})),
            "metadata": _make_json_serializable(strip_for_export(dict(metadata or {}))),
            "provenance": _make_json_serializable(dict(provenance or {})),
        }
    )
    for unit in units:
        k, t = unit.n_rois, unit.n_timepoints
        group = root.create_group(unit.name)
        group.attrs.update(
            {
                "kind": unit.kind,
                "index": int(unit.index),
                "fs": None if unit.fs is None else float(unit.fs),
                "n_rois": k,
                "n_timepoints": t,
                "roi_names": [str(n) for n in unit.roi_names],
                "member_kind": unit.member_kind,
                "image_shape": None if unit.image_shape is None else [int(v) for v in unit.image_shape],
                **_make_json_serializable(dict(unit.attrs)),
            }
        )
        traces = group.create_group("traces")
        for kind, arr in unit.traces.items():
            data = np.ascontiguousarray(arr, dtype=np.float32)
            chunks = tuple(max(1, min(n, c)) for n, c in zip(data.shape, _CHUNK, strict=True))
            traces.create_array(kind, data=data, chunks=chunks, dimension_names=("roi", "t"))
        rois = group.create_group("rois")
        offsets = np.zeros(k + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(m) for m in unit.members])
        member = np.concatenate([np.asarray(m, dtype=np.int64).ravel() for m in unit.members]) if k else np.zeros(0, np.int64)
        if unit.weights is None:
            weight = np.ones(len(member), dtype=np.float32)
        else:
            weight = np.concatenate([np.asarray(w, dtype=np.float32).ravel() for w in unit.weights]) if k else np.zeros(0, np.float32)
        rois.create_array("offsets", data=offsets, chunks=(max(1, k + 1),))
        rois.create_array("member", data=member, chunks=(max(1, min(len(member), 1 << 20)),))
        rois.create_array("weight", data=weight, chunks=(max(1, min(len(weight), 1 << 20)),))
        iscell = np.ones((k, 2), dtype=np.float32) if unit.iscell is None else np.asarray(unit.iscell, dtype=np.float32)
        rois.create_array("iscell", data=iscell, chunks=(max(1, k), 2))
        if unit.member_traces:
            members = group.create_group("members")
            for kind, arr in unit.member_traces.items():
                data = np.ascontiguousarray(arr, dtype=np.float32)
                chunks = tuple(max(1, min(n, c)) for n, c in zip(data.shape, _CHUNK, strict=True))
                members.create_array(kind, data=data, chunks=chunks, dimension_names=("member", "t"))
        events = group.create_group("events")
        frames, roi_index = [], []
        for i, roi in enumerate(unit.roi_names):
            found = np.asarray(unit.events.get(roi, ()), dtype=np.int64).ravel()
            frames.append(np.sort(found))
            roi_index.append(np.full(found.size, i, dtype=np.int32))
        frame = np.concatenate(frames) if frames else np.zeros(0, np.int64)
        roi_index = np.concatenate(roi_index) if roi_index else np.zeros(0, np.int32)
        events.create_array("frame", data=frame, chunks=(max(1, min(frame.size, 1 << 20)),))
        events.create_array("roi", data=roi_index, chunks=(max(1, min(roi_index.size, 1 << 20)),))
        if unit.images:
            images = group.create_group("images")
            for kind, arr in unit.images.items():
                data = np.ascontiguousarray(arr, dtype=np.float32)
                images.create_array(kind, data=data, chunks=data.shape, dimension_names=("y", "x"))
    logger.info(f"wrote {len(units)} unit(s) of {pipeline} results to {path}")
    return path


def read_results(path) -> Results:
    """Read a results file back, every array in memory."""
    path = Path(path)
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    if RESULTS_ATTR not in attrs:
        raise ValueError(f"{path} is not a results file (no {RESULTS_ATTR!r} attr)")
    units = {}
    for name in attrs["units"]:
        group = root[name]
        a = dict(group.attrs)
        k = int(a["n_rois"])
        offsets = group["rois/offsets"][:]
        member = group["rois/member"][:]
        weight = group["rois/weight"][:]
        traces = {kind: group["traces"][kind][:] for kind in group["traces"].array_keys()}
        member_traces = {kind: group["members"][kind][:] for kind in group["members"].array_keys()} if "members" in group else {}
        images = {kind: group["images"][kind][:] for kind in group["images"].array_keys()} if "images" in group else {}
        frame = group["events/frame"][:]
        roi_index = group["events/roi"][:]
        roi_names = [str(n) for n in a["roi_names"]]
        events = {roi_names[i]: frame[roi_index == i] for i in np.unique(roi_index)}
        units[name] = ResultUnit(
            name=name,
            kind=str(a["kind"]),
            index=int(a["index"]),
            fs=None if a.get("fs") is None else float(a["fs"]),
            roi_names=roi_names,
            traces=traces,
            member_kind=str(a["member_kind"]),
            members=[member[offsets[i] : offsets[i + 1]] for i in range(k)],
            weights=[weight[offsets[i] : offsets[i + 1]] for i in range(k)],
            image_shape=None if a.get("image_shape") is None else (int(a["image_shape"][0]), int(a["image_shape"][1])),
            iscell=group["rois/iscell"][:],
            member_traces=member_traces,
            events=events,
            images=images,
            attrs={key: value for key, value in a.items() if key not in _UNIT_ATTRS},
        )
    return Results(
        path=path,
        pipeline=str(attrs.get("pipeline", "")),
        created=str(attrs.get("created", "")),
        tags=[str(t) for t in attrs.get("tags", [])],
        source=dict(attrs.get("source") or {}),
        settings=dict(attrs.get("settings") or {}),
        metadata=dict(attrs.get("metadata") or {}),
        provenance=dict(attrs.get("provenance") or {}),
        units=units,
    )


def results_from_suite2p(path) -> tuple[list[ResultUnit], dict]:
    """Mold a suite2p or masknmf output folder into result units.

    ``path`` is one plane dir or a folder of ``zplaneNN`` / ``planeNN`` dirs.
    ``F`` is ``raw``, ``Fneu`` ``neuropil``, ``spks`` ``spikes``,
    ``norm_traces`` (masknmf's percent dF/F) ``dff``; ``stat`` gives each
    ROI's pixels and ``lam`` weights; ``meanImg``, ``max_proj``, ``Vcorr`` and
    ``refImg`` are the images. Returns ``(units, root)`` where ``root`` holds
    the ``pipeline``, ``settings`` and ``metadata`` for :func:`write_results`.
    """
    from mbo_utilities.metadata import get_param

    path = Path(path)
    plane_dirs = [path] if (path / "F.npy").is_file() else sorted(p for p in path.iterdir() if p.is_dir() and (p / "F.npy").is_file())
    if not plane_dirs:
        raise FileNotFoundError(f"no F.npy in {path} or its plane dirs")
    units, root = [], {"pipeline": "suite2p", "settings": {}, "metadata": {}}
    for plane_dir in plane_dirs:
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item() if (plane_dir / "ops.npy").is_file() else {}
        stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
        F = np.load(plane_dir / "F.npy")
        match = re.search(r"plane(\d+)", plane_dir.name, re.IGNORECASE)
        index = int(match.group(1)) if match else int(ops.get("plane", len(units) + 1) or len(units) + 1)
        ly, lx = int(ops.get("Ly", 0) or 0), int(ops.get("Lx", 0) or 0)
        if not ly or not lx:
            for key in ("meanImg", "max_proj", "refImg"):
                if isinstance(ops.get(key), np.ndarray) and ops[key].ndim == 2:
                    ly, lx = ops[key].shape
                    break
        members, weights = [], []
        for s in stat:
            ypix, xpix = np.asarray(s["ypix"], dtype=np.int64).ravel(), np.asarray(s["xpix"], dtype=np.int64).ravel()
            members.append(ypix * lx + xpix)
            weights.append(np.asarray(s.get("lam", np.ones(ypix.size)), dtype=np.float32).ravel())
        traces = {"raw": F}
        for file, kind in (("Fneu.npy", "neuropil"), ("spks.npy", "spikes"), ("norm_traces.npy", "dff")):
            if (plane_dir / file).is_file():
                traces[kind] = np.load(plane_dir / file)
        images = {
            kind: np.asarray(ops[key])
            for key, kind in (("meanImg", "mean"), ("max_proj", "max"), ("Vcorr", "corr"), ("refImg", "ref"))
            if isinstance(ops.get(key), np.ndarray) and ops[key].ndim == 2
        }
        units.append(
            ResultUnit(
                name=unit_name("plane", index),
                kind="plane",
                index=index,
                fs=get_param(ops, "fs") if ops else None,
                roi_names=[str(i) for i in range(len(stat))],
                traces=traces,
                member_kind="pixel",
                members=members,
                weights=weights,
                image_shape=(ly, lx) if ly and lx else None,
                iscell=np.load(plane_dir / "iscell.npy") if (plane_dir / "iscell.npy").is_file() else None,
                images=images,
                attrs={"plane_dir": str(plane_dir)},
            )
        )
        if ops.get("pipeline") == "masknmf":
            root["pipeline"] = "masknmf"
            root["settings"] = dict(ops.get("masknmf") or {})
        if not root["metadata"]:
            root["metadata"] = ops
    return units, root


def results_from_pf(pf_dir) -> tuple[list[ResultUnit], dict]:
    """Mold the voltage pipeline's ``PF`` folder into result units, one scan each.

    The domains are the ROIs (``roi_names``), their lines the members, the
    curated trace is ``denoised``, ``test.h5`` gives ``dff`` and ``zscore``,
    the peaks are the events and ``traces/scan<id>_rois.npy`` (when the
    folder has it) the members' ``raw`` traces. Returns ``(units, root)``
    with the folder's ``pipeline.json`` as ``provenance``.
    """
    import h5py

    from mbo_utilities.arrays.pf import PfArray

    pf = PfArray(pf_dir, source=False)
    provenance = dict(pf.provenance)
    rates = {str(k): float(v) for k, v in (provenance.get("fs_hz") or {}).items()}
    units = []
    for scan in pf.scan_ids:
        names = list(pf.domain_names)
        denoised = np.stack([pf.traces[scan][d] for d in names]) if names else np.zeros((0, 0), np.float32)
        traces = {"denoised": denoised}
        dfof_file = pf.pf_dir / "test.h5"
        if dfof_file.is_file():
            with h5py.File(dfof_file, "r") as f:
                if scan in f:
                    traces["dff"] = f[scan]["dfof_raw"][:]
                    traces["zscore"] = f[scan]["dfof_zscore"][:]
        member_traces = {}
        raw_file = pf.pf_dir / "traces" / f"scan{scan}_rois.npy"
        if raw_file.is_file():
            member_traces["raw"] = np.load(raw_file)
        try:
            index = int(scan)
        except ValueError:
            index = pf.scan_ids.index(scan)
        units.append(
            ResultUnit(
                name=unit_name("scan", index),
                kind="scan",
                index=index,
                fs=rates.get(scan, pf.fs_by_scan.get(scan)),
                roi_names=names,
                traces=traces,
                member_kind="line",
                members=[np.asarray(pf.domains.get(d, []), dtype=np.int64) for d in names],
                member_traces=member_traces,
                events={d: pf.events(d, scan) for d in names if pf.events(d, scan).size},
                attrs={
                    "scan_id": scan,
                    "source_unit": pf.source_units.get(scan, ""),
                    "first_env": scan in pf.first_env,
                    "member_ids": list(pf.roi_list.get(scan, [])),
                },
            )
        )
    root = {
        "pipeline": "voltage",
        "source": dict(provenance.get("source") or {}),
        "settings": dict(provenance.get("settings") or {}),
        "metadata": dict(provenance.get("source_metadata") or {}),
        "provenance": provenance,
    }
    return units, root
