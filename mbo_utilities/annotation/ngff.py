"""The rois zarr: one recording's drawn ROIs, quick traces and loaded runs.

``<stem>.rois.zarr`` is a zarr v3 group written whole on every save (into a
temporary name, then swapped in, so a crash never leaves half a file):

```
rois/                 the truth; the results zarr's membership encoding
  offsets member weight     ROI k is member[offsets[k]:offsets[k+1]], flat y * X + x
  uid plane class polygon_offsets polygon
  attrs: note, source, color (per ROI, in rois/uid order)
traces/               quick traces and pixel averages, ragged along T
  offsets raw neuropil      row k is raw[offsets[k]:offsets[k+1]]; neuropil NaN when unmeasured
  attrs: rows               one RoiTrace.record() (plus extra) per row
labels/               OME-NGFF 0.5 labels group, derived from rois/, never read back
  drawn/              one label image: multiscales + image-label, value = uid
    0                 uint32 (planes..., Y, X)
attrs: mbo_rois, source {path, reader_kwargs, unit}, shape, plane_axes,
       label_names, next_uid, runs [{path, discarded, classes, colors, iscell}]
```

:func:`write_label_image` writes any label volume as an NGFF 0.5 label
image (the suite2p zarr export's ``labels`` uses it too).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities.annotation.store import RoiLabelStore, RoiRecord
from mbo_utilities.annotation.traces import RoiTrace
from mbo_utilities.arrays.features._dim_tags import dim_to_ome_axis

__all__ = [
    "NGFF_VERSION",
    "ROIS_VERSION",
    "SAVED_SOURCES",
    "RoiFile",
    "RoiFileContents",
    "write_label_image",
]

NGFF_VERSION = "0.5"
ROIS_VERSION = 1
ROIS_ATTR = "mbo_rois"
LABEL_IMAGE = "drawn"
SAVED_SOURCES = (
    "quick",
    "pixel",
)  # the trace rows a rois zarr keeps: measured here and nowhere else
_COMPRESSION_LEVEL = 4


@dataclass
class RoiFileContents:
    """A rois zarr's contents as :meth:`RoiFile.load` returns them."""

    store: RoiLabelStore
    traces: list[RoiTrace] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    source: dict = field(default_factory=dict)


def _plane_axes_ome(store: RoiLabelStore) -> tuple[list[dict], tuple[int, ...]]:
    """The label image's axes and the shape of its plane dims: the store's
    plane axes named for OME (``c`` for the channel axis, ``z`` for the
    z-plane, anything else by its own lowercase name), else one ``z`` axis
    over the planes.
    """
    if not store.plane_axes:
        return [dim_to_ome_axis("Z")], (store.nz,)
    axes = []
    for name, _size in store.plane_axes:
        if name == store.axis_name("c"):
            axes.append(dim_to_ome_axis("C"))
        elif name == store.axis_name("z"):
            axes.append(dim_to_ome_axis("Z"))
        else:
            axes.append({"name": str(name).lower()})
    return axes, tuple(int(n) for _name, n in store.plane_axes)


def write_label_image(
    labels_group,
    name: str,
    volume: np.ndarray,
    axes: list[dict],
    *,
    scale: list[float] | None = None,
    colors: list[dict] = (),
    properties: list[dict] = (),
    source_image: str | None = None,
    chunk_planes: int = 1,
) -> None:
    """Write ``volume`` as the NGFF 0.5 label image ``name`` under a
    ``labels`` group: a multiscales group with one dataset ``0``, its
    ``image-label`` colors and properties keyed by label value, and the
    group's ``labels`` list updated to include it.

    ``axes`` are the OME axes of ``volume`` (one per dim), ``scale`` their
    physical steps (1.0 each when None), ``source_image`` the relative path
    to the image the labels sit on when there is one.
    """
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    volume = np.asarray(volume)
    if volume.ndim != len(axes):
        raise ValueError(f"{volume.ndim}-D labels but {len(axes)} axes")
    if not np.issubdtype(volume.dtype, np.integer):
        raise ValueError(f"a label image is integer, not {volume.dtype}")
    scale = [1.0] * volume.ndim if scale is None else [float(s) for s in scale]
    listed = list(dict(labels_group.attrs.get("ome") or {}).get("labels") or [])
    if name not in listed:
        listed.append(name)
    labels_group.attrs["ome"] = {"version": NGFF_VERSION, "labels": listed}
    image = labels_group.create_group(name, overwrite=True)
    chunks = (chunk_planes,) * (volume.ndim - 2) + tuple(volume.shape[-2:])
    chunks = tuple(min(c, s) for c, s in zip(chunks, volume.shape))
    arr = zarr.create_array(
        store=image.store,
        name=f"{image.path}/0",
        shape=volume.shape,
        chunks=chunks,
        dtype=volume.dtype,
        compressors=[GzipCodec(level=_COMPRESSION_LEVEL)],
        serializer=BytesCodec(),
        dimension_names=[str(a["name"]) for a in axes],
        overwrite=True,
    )
    arr[:] = volume
    image_label = {
        "version": NGFF_VERSION,
        "colors": list(colors),
        "properties": list(properties),
    }
    if source_image:
        image_label["source"] = {"image": str(source_image)}
    image.attrs["ome"] = {
        "version": NGFF_VERSION,
        "multiscales": [
            {
                "name": name,
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": scale}
                        ],
                    }
                ],
            }
        ],
        "image-label": image_label,
    }


def _label_volume(store: RoiLabelStore) -> np.ndarray:
    """``(P, Y, X)`` uint32 with each pixel's ROI uid, the smallest ROI on
    top where they overlap (the in-memory label image, relabelled).
    """
    out = np.zeros((store.nz, store.ny, store.nx), np.uint32)
    lut = np.zeros(len(store.rois) + 1, np.uint32)
    for i, r in enumerate(store.rois):
        lut[i + 1] = r.uid
    for plane in sorted({r.plane for r in store.rois}):
        out[plane] = lut[store.label_image(plane)]
    return out


def _relative_image(source: dict | None, path: Path) -> str | None:
    """The recording as a path relative to ``labels/drawn`` when it is a
    zarr store (the only kind an NGFF reader can follow), else None.
    """
    recording = (source or {}).get("path")
    if not recording or not str(recording).endswith(".zarr"):
        return None
    return os.path.relpath(
        Path(recording).resolve(), (path / "labels" / LABEL_IMAGE).resolve()
    )


def _trace_record(trace: RoiTrace) -> dict:
    record = trace.record()
    extra = {}
    for key, value in (trace.extra or {}).items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        extra[key] = value
    record["extra"] = extra
    return record


class RoiFile:
    """Reader / writer for one rois zarr."""

    def __init__(self, path):
        self.path = Path(path)

    def save(
        self,
        store: RoiLabelStore,
        traces=(),
        *,
        source: dict | None = None,
        runs=(),
        scale: tuple[float | None, float | None, float | None] | None = None,
    ) -> None:
        """Write the whole file: the store, the trace rows whose ``source``
        is in ``SAVED_SOURCES``, the ``runs`` entries, and the derived label
        image. ``source`` names the recording (``path``, ``reader_kwargs``,
        ``unit``); ``scale`` is its ``(dz, dy, dx)`` in micrometers.
        """
        import zarr

        path = self.path
        tmp = path.with_name(path.name + ".tmp")
        old = path.with_name(path.name + ".old")
        for leftover in (tmp, old):
            if leftover.exists():
                shutil.rmtree(leftover)
        path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(tmp), mode="w", zarr_format=3)
        rois = store.rois
        n = len(rois)
        offsets = np.zeros(n + 1, np.int64)
        offsets[1:] = np.cumsum([r.area for r in rois])
        member = (
            np.concatenate([r.pixels for r in rois]) if n else np.zeros(0, np.int64)
        )
        weight = (
            np.concatenate([r.weights for r in rois]) if n else np.zeros(0, np.float32)
        )
        polygons = [
            np.zeros((0, 2), np.float32) if r.polygon is None else r.polygon
            for r in rois
        ]
        polygon_offsets = np.zeros(n + 1, np.int64)
        polygon_offsets[1:] = np.cumsum([len(p) for p in polygons])
        polygon = (
            np.concatenate(polygons) if n else np.zeros((0, 2), np.float32)
        ).astype(np.float32)
        color = np.full((n, 3), -1, np.int16)
        for i, r in enumerate(rois):
            if r.color is not None:
                color[i] = r.color
        group = root.create_group("rois")
        for name, data in (
            ("offsets", offsets),
            ("member", member),
            ("weight", weight),
            ("uid", np.asarray([r.uid for r in rois], np.int64)),
            ("plane", np.asarray([r.plane for r in rois], np.int64)),
            ("class", np.asarray([r.class_index for r in rois], np.int64)),
            ("polygon_offsets", polygon_offsets),
            ("polygon", polygon),
            ("color", color),
        ):
            group.create_array(
                name, data=data, chunks=tuple(max(1, s) for s in data.shape)
            )
        group.attrs.update(
            {
                "note": [r.note for r in rois],
                "source": [r.source for r in rois],
                "polygon": "(m, 2) float32 (x, y) as drawn; empty for a mask from a run",
                "color": "uint8 rgb per ROI, -1 when unset",
            }
        )

        rows = [t for t in traces if t.source in SAVED_SOURCES and t.F is not None]
        t_offsets = np.zeros(len(rows) + 1, np.int64)
        t_offsets[1:] = np.cumsum([int(np.size(t.F)) for t in rows])
        raw = (
            np.concatenate([np.asarray(t.F, np.float32).ravel() for t in rows])
            if rows
            else np.zeros(0, np.float32)
        )
        neuropil = np.full(raw.shape, np.nan, np.float32)
        for k, t in enumerate(rows):
            if t.Fneu is not None:
                neuropil[t_offsets[k] : t_offsets[k + 1]] = np.asarray(
                    t.Fneu, np.float32
                ).ravel()
        group = root.create_group("traces")
        for name, data in (
            ("offsets", t_offsets),
            ("raw", raw),
            ("neuropil", neuropil),
        ):
            group.create_array(name, data=data, chunks=(max(1, data.size),))
        group.attrs["rows"] = [_trace_record(t) for t in rows]

        axes, plane_shape = _plane_axes_ome(store)
        volume = _label_volume(store).reshape(*plane_shape, store.ny, store.nx)
        dz, dy, dx = scale or (None, None, None)
        steps = [float(dz or 1.0) if a["name"] == "z" else 1.0 for a in axes] + [
            float(dy or 1.0),
            float(dx or 1.0),
        ]
        names = list(store.label_names)
        write_label_image(
            root.create_group("labels"),
            LABEL_IMAGE,
            volume,
            [*axes, dim_to_ome_axis("Y"), dim_to_ome_axis("X")],
            scale=steps,
            colors=[
                {"label-value": int(r.uid), "rgba": [*store.roi_rgb(i), 255]}
                for i, r in enumerate(rois)
            ],
            properties=[
                {
                    "label-value": int(r.uid),
                    "class": names[r.class_index] if r.class_index >= 0 else "",
                    "class-index": int(r.class_index),
                    "note": r.note,
                    "plane": int(r.plane),
                    "area": int(r.area),
                    "source": r.source,
                }
                for r in rois
            ],
            source_image=_relative_image(source, path),
        )
        root.attrs.update(
            {
                ROIS_ATTR: ROIS_VERSION,
                "source": dict(source or {}),
                "shape": [int(v) for v in store.shape],
                "plane_axes": [[str(n), int(s)] for n, s in store.plane_axes],
                "label_names": names,
                "next_uid": int(store.next_uid),
                "min_pixels": int(store.min_pixels),
                "runs": [_run_entry(e) for e in runs],
            }
        )
        if path.exists():
            path.rename(old)
        tmp.rename(path)
        if old.exists():
            shutil.rmtree(old)

    @staticmethod
    def load(path) -> RoiFileContents:
        """Read a rois zarr back.

        Raises ``FileNotFoundError`` when there is none at ``path`` and
        ``ValueError`` when the group is not one.
        """
        import zarr

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)
        if ROIS_ATTR not in attrs:
            raise ValueError(f"{path} is not a rois zarr (no {ROIS_ATTR!r} attr)")
        nz, ny, nx = (int(v) for v in attrs["shape"])
        group = root["rois"]
        offsets = group["offsets"][:]
        member = group["member"][:]
        weight = group["weight"][:]
        uid = group["uid"][:]
        plane = group["plane"][:]
        klass = group["class"][:]
        polygon_offsets = group["polygon_offsets"][:]
        polygon = group["polygon"][:]
        color = group["color"][:]
        notes = list(group.attrs.get("note") or [])
        sources = list(group.attrs.get("source") or [])
        label_names = tuple(str(n) for n in attrs.get("label_names") or ())
        rois = []
        for k in range(len(uid)):
            lo, hi = int(offsets[k]), int(offsets[k + 1])
            plo, phi = int(polygon_offsets[k]), int(polygon_offsets[k + 1])
            class_index = int(klass[k])
            if not -1 <= class_index < max(len(label_names), 1):
                class_index = -1
            rois.append(
                RoiRecord(
                    plane=int(plane[k]),
                    pixels=member[lo:hi],
                    weights=weight[lo:hi],
                    class_index=class_index,
                    note=str(notes[k]) if k < len(notes) else "",
                    uid=int(uid[k]),
                    source=str(sources[k]) if k < len(sources) else "",
                    color=None
                    if int(color[k][0]) < 0
                    else tuple(int(v) for v in color[k]),
                    polygon=polygon[plo:phi] if phi > plo else None,
                )
            )
        store = RoiLabelStore(
            nz,
            ny,
            nx,
            label_names=label_names,
            min_pixels=int(attrs.get("min_pixels", 1) or 1),
            rois=rois,
            next_uid=int(attrs.get("next_uid", 0) or 0),
        )
        store.plane_axes = tuple(
            (str(n), int(s)) for n, s in (attrs.get("plane_axes") or ())
        )
        traces = []
        if "traces" in root:
            group = root["traces"]
            t_offsets = group["offsets"][:]
            raw = group["raw"][:]
            neuropil = group["neuropil"][:]
            for k, row in enumerate(group.attrs.get("rows") or []):
                lo, hi = int(t_offsets[k]), int(t_offsets[k + 1])
                fneu = neuropil[lo:hi]
                extra = dict(row.get("extra") or {})
                if "pixel" in extra:
                    extra["pixel"] = tuple(int(v) for v in extra["pixel"])
                traces.append(
                    RoiTrace(
                        uid=int(row["uid"]),
                        z=int(row["z"]),
                        c=int(row["c"]),
                        engine=str(row["engine"]),
                        source=str(row["source"]),
                        member=row.get("member"),
                        F=raw[lo:hi],
                        Fneu=None if np.isnan(fneu).all() else fneu,
                        frames=None
                        if row.get("frames") is None
                        else tuple(int(v) for v in row["frames"]),
                        frame_average=int(row.get("frame_average", 1) or 1),
                        fs=row.get("fs"),
                        label=str(row.get("label") or ""),
                        path=None if row.get("path") is None else Path(row["path"]),
                        extra=extra,
                    )
                )
        return RoiFileContents(
            store=store,
            traces=traces,
            runs=[_run_entry(e) for e in attrs.get("runs") or []],
            source=dict(attrs.get("source") or {}),
        )


def _run_entry(entry: dict) -> dict:
    """One loaded run as the file keeps it: its path, the rows marked for
    deletion, the per-row class labels and group colors, and the accept /
    reject overrides on the run's own ``iscell``.
    """
    return {
        "path": str(entry["path"]),
        "discarded": sorted(int(i) for i in entry.get("discarded", ())),
        "classes": {str(k): int(v) for k, v in (entry.get("classes") or {}).items()},
        "colors": {
            str(k): [float(x) for x in v]
            for k, v in (entry.get("colors") or {}).items()
        },
        "iscell": {str(k): bool(v) for k, v in (entry.get("iscell") or {}).items()},
    }
