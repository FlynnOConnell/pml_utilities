"""``get_param`` and friends: read a metadata dict through the registry's canonical names, aliases and transforms."""

from __future__ import annotations

import contextlib
from typing import Any

from .base import ALIAS_MAP, METADATA_PARAMS, VoxelSize


def _rate_precedence() -> tuple[dict[str, tuple[int, str]], int]:
    """Lowercase rate key -> (rank, "rate" | "interval"), and the OME time scale's rank.

    ``fs`` ranks first because user overrides (the metadata editor, suite2p
    custom_metadata merges) write it; then ``frame_rate``, ``finterval``, the
    OME time scale, and the remaining aliases in registry order, ``fps`` last.
    """
    order: dict[str, tuple[int, str]] = {
        "fs": (0, "rate"),
        "frame_rate": (1, "rate"),
        "finterval": (2, "interval"),
    }
    ome_rank = 3
    rank = 4
    for alias in METADATA_PARAMS["finterval"].aliases:
        key = alias.lower()
        if key not in order:
            order[key] = (rank, "interval")
            rank += 1
    fs_aliases = [a for a in METADATA_PARAMS["fs"].aliases if a.lower() != "fps"]
    fs_aliases.append("fps")
    for alias in fs_aliases:
        key = alias.lower()
        if key not in order:
            order[key] = (rank, "rate")
            rank += 1
    return order, ome_rank


def _exact_rate_spellings() -> frozenset[str]:
    return frozenset(
        {
            "fs",
            "finterval",
            *METADATA_PARAMS["fs"].aliases,
            *METADATA_PARAMS["finterval"].aliases,
        }
    )


_RATE_PRECEDENCE, _OME_RATE_RANK = _rate_precedence()
_EXACT_RATE_SPELLINGS = _exact_rate_spellings()
_CANONICAL_RATE_MAX_RANK = 2  # a None under fs, frame_rate or finterval means "no valid rate", not "look further"
_STALE_WARNED: set = (
    set()
)  # divergence signatures already warned about; get_param(md, "fs") runs per frame
_STALE_WARNED_MAX = 256


def scale_frame_rate(metadata: dict, factor: float) -> dict:
    """A copy of ``metadata`` retimed for ``factor`` source frames per frame.

    Every registered rate spelling is divided and every interval spelling
    multiplied, so no alias is left claiming the pre-binning rate. Nested OME
    ``multiscales`` describe the file on disk and stay.
    """
    factor = float(factor)
    if factor == 1.0:
        return dict(metadata)
    out = dict(metadata)
    for key, value in metadata.items():
        entry = _RATE_PRECEDENCE.get(str(key).lower())
        kind = entry[1] if entry else ("interval" if key == "_ome_time_scale" else None)
        if kind is None:
            continue
        numeric = _rate_value(value)
        if numeric is None:
            continue
        out[key] = numeric * factor if kind == "interval" else numeric / factor
    return out


def _rate_value(val: Any) -> float | None:
    """``val`` as a finite non-zero float, else None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f == 0 or f != f:
        return None
    return f


def _ome_time_scale(metadata: dict) -> tuple[float, str] | None:
    """The level-0 time-axis scale in seconds per frame and where it came from (``ome.multiscales``, ``multiscales`` or the reader's ``_ome_time_scale``); 1.0 is the writer default and counts as unset."""
    sources = []
    ome = metadata.get("ome")
    if isinstance(ome, dict):
        sources.append((ome.get("multiscales"), "ome.multiscales"))
    sources.append((metadata.get("multiscales"), "multiscales"))
    for multiscales, label in sources:
        try:
            entry = multiscales[0]
            axes = entry["axes"]
            idx = next(
                i
                for i, ax in enumerate(axes)
                if isinstance(ax, dict) and ax.get("type") == "time"
            )
            scale = float(
                entry["datasets"][0]["coordinateTransformations"][0]["scale"][idx]
            )
        except (TypeError, KeyError, IndexError, ValueError, StopIteration):
            continue
        if scale > 0 and scale != 1.0:
            return scale, label
    raw = _rate_value(metadata.get("_ome_time_scale"))
    if raw is not None and raw > 0 and raw != 1.0:
        return raw, "_ome_time_scale"
    return None


def resolve_effective_rate(
    metadata: dict | None,
) -> tuple[float | None, float | None, str]:
    """``(fs_hz, finterval_s, source_key)`` from the one rate spelling that wins, ``(None, None, "")`` when none is usable.

    Older stores stamped every alias at ingest and retimed only the canonical
    keys on decimating writes, so a dict can hold a correct ``fs`` beside a
    stale ``framerate``. Precedence: ``fs``, ``frame_rate``, ``finterval``,
    the OME time scale, then the remaining aliases in registry order (``fps``
    last), case-insensitive; at equal rank the registered spelling beats a
    case variant. A canonical key stored as None (a non-contiguous selection)
    blocks the lower ranks. A candidate disagreeing with the winner by over
    0.1% logs one warning per signature, DEBUG on repeats.
    """
    if not isinstance(metadata, dict) or not metadata:
        return None, None, ""

    candidates: list[
        tuple[int, str, float, float, Any]
    ] = []  # (rank, key, fs, finterval, raw)
    canonical_nulled = False
    for key, val in metadata.items():
        if not isinstance(key, str):
            continue
        low = key.lower()
        if low == "_ome_time_scale":
            continue
        info = _RATE_PRECEDENCE.get(low)
        if info is None:
            continue
        rank, kind = info
        value = _rate_value(val)
        if value is None:
            if val is None and rank <= _CANONICAL_RATE_MAX_RANK:
                canonical_nulled = True
            continue
        if kind == "interval":
            candidates.append((rank, key, 1.0 / value, value, val))
        else:
            candidates.append((rank, key, value, 1.0 / value, val))

    ome = _ome_time_scale(metadata)
    if ome is not None:
        scale, label = ome
        candidates.append((_OME_RATE_RANK, label, 1.0 / scale, scale, scale))

    if not candidates:
        return None, None, ""

    candidates.sort(key=lambda c: (c[0], 0 if c[1] in _EXACT_RATE_SPELLINGS else 1))
    best_rank, best_key, best_fs, best_finterval, _ = candidates[0]
    if canonical_nulled and best_rank > _CANONICAL_RATE_MAX_RANK:
        return None, None, ""

    stale = [
        (key, raw)
        for _rank, key, fs, _fint, raw in candidates[1:]
        if abs(fs - best_fs) > 1e-3 * abs(best_fs)
    ]
    if stale:
        listing = ", ".join(f"{k}={v!r}" for k, v in stale)
        message = (
            f"metadata carries stale frame-rate aliases: resolved "
            f"fs={best_fs:.6g} Hz from {best_key!r}, but {listing} disagree"
        )
        signature = (
            best_key,
            round(best_fs, 6),
            frozenset((k, round(float(v), 6)) for k, v in stale),
        )
        from mbo_utilities import log

        logger = log.get("metadata.params")
        if signature in _STALE_WARNED:
            logger.debug(message)
        else:
            if len(_STALE_WARNED) >= _STALE_WARNED_MAX:
                _STALE_WARNED.clear()
            _STALE_WARNED.add(signature)
            logger.warning(message)
    return best_fs, best_finterval, best_key


def get_param(
    metadata: dict | None,
    name: str,
    default: Any = None,
    *,
    override: Any = None,
    shape: tuple | None = None,
    _apply_transforms: bool = True,
) -> Any:
    """The value of ``name`` (any spelling) in ``metadata``, converted to its registered dtype, else ``default`` or the registered default.

    ``override`` is returned as given. The lookup walks canonical key,
    aliases, the ``pixel_resolution`` tuple for ``dx``/``dy``, transform
    aliases, and ``shape`` for ``Lx``/``Ly``; ``fs`` and ``finterval`` go
    through ``resolve_effective_rate``. An unregistered name is a plain dict
    lookup.

    >>> get_param({"umPerPixX": 0.5, "frame_rate": 30.0}, "fs")
    30.0
    """
    if override is not None:
        return override

    canonical = ALIAS_MAP.get(name.lower())
    if canonical is None:
        if metadata is not None and name in metadata:
            return metadata[name]
        return default

    param = METADATA_PARAMS[canonical]
    final_default = default if default is not None else param.default

    if metadata is None:
        if shape is not None:
            if canonical == "Lx" and len(shape) >= 1:
                return int(shape[-1])
            if canonical == "Ly" and len(shape) >= 2:
                return int(shape[-2])
        return final_default

    if canonical in ("fs", "finterval") and _apply_transforms:
        fs_hz, finterval_s, _src = resolve_effective_rate(metadata)
        resolved = fs_hz if canonical == "fs" else finterval_s
        if resolved is not None:
            with contextlib.suppress(TypeError, ValueError):
                return param.dtype(resolved)
        return final_default

    for key in (param.canonical, *param.aliases):
        val = metadata.get(key)
        if val is not None:
            try:
                if param.dtype == tuple:
                    return tuple(val) if isinstance(val, (list, tuple)) else val
                return param.dtype(val)
            except (TypeError, ValueError):
                continue

    if canonical in ("dx", "dy"):
        pixel_res = metadata.get("pixel_resolution")
        if pixel_res is not None:
            if isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 2:
                try:
                    idx = 0 if canonical == "dx" else 1
                    return float(pixel_res[idx])
                except (TypeError, ValueError, IndexError):
                    pass
            elif isinstance(pixel_res, (int, float)):
                return float(pixel_res)

    # _apply_transforms=False on the nested lookup stops the A->B->A recursion
    if _apply_transforms and param.transforms:
        for src_key, (to_canonical, _from_canonical) in param.transforms.items():
            src_val = get_param(metadata, src_key, _apply_transforms=False)
            if src_val is not None:
                try:
                    return param.dtype(to_canonical(src_val))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue

    if shape is not None:
        if canonical == "Lx" and len(shape) >= 1:
            return int(shape[-1])
        if canonical == "Ly" and len(shape) >= 2:
            return int(shape[-2])

    if canonical == "dtype":
        arr_dtype = metadata.get("dtype")
        if arr_dtype is not None:
            return str(arr_dtype)

    return final_default


def get_voxel_size(
    metadata: dict | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> VoxelSize:
    """The voxel size in µm: the given overrides, else the canonical keys, the ``pixel_resolution`` tuple, the OME ``PhysicalSize*`` keys, ScanImage's piezo step, else 1.0.

    ``dz`` is None for an LBM stack with none stored: the beamlet spacing is
    user-supplied, never inferred.
    """
    if metadata is None:
        metadata = {}

    pixel_res = metadata.get("pixel_resolution")
    px_x, px_y = None, None
    if pixel_res is not None:
        if isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 2:
            try:
                px_x = float(pixel_res[0])
                px_y = float(pixel_res[1])
            except (TypeError, ValueError):
                pass
        elif isinstance(pixel_res, (int, float)):
            px_x = px_y = float(pixel_res)

    si_dz = None
    is_lbm = metadata.get("lbm_stack", False) or metadata.get("stack_type") == "lbm"
    if not is_lbm:
        si = metadata.get("si", {})
        if isinstance(si, dict):
            h_stack = si.get("hStackManager", {})
            if isinstance(h_stack, dict):
                si_dz = h_stack.get("actualStackZStepSize")
                if si_dz is None:
                    si_dz = h_stack.get("stackZStepSize")

    resolved_dx = dx
    if resolved_dx is None:
        resolved_dx = _first_float(metadata, ["dx", "PhysicalSizeX"])
    if resolved_dx is None and px_x is not None:
        resolved_dx = px_x
    if resolved_dx is None:
        resolved_dx = 1.0

    resolved_dy = dy
    if resolved_dy is None:
        resolved_dy = _first_float(metadata, ["dy", "PhysicalSizeY"])
    if resolved_dy is None and px_y is not None:
        resolved_dy = px_y
    if resolved_dy is None:
        resolved_dy = 1.0

    resolved_dz = dz
    if resolved_dz is None:
        resolved_dz = _first_float(
            metadata, ["dz", "z_step", "PhysicalSizeZ", "spacing"]
        )
    if resolved_dz is None and si_dz is not None:
        with contextlib.suppress(TypeError, ValueError):
            resolved_dz = float(si_dz)
    if resolved_dz is None and not is_lbm:
        resolved_dz = 1.0

    return VoxelSize(dx=resolved_dx, dy=resolved_dy, dz=resolved_dz)


def _first_float(metadata: dict, keys: list[str]) -> float | None:
    """The first of ``keys`` present in ``metadata`` that converts to a float."""
    for key in keys:
        val = metadata.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def normalize_resolution(
    metadata: dict,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> dict:
    """Stamp ``metadata`` in place with the voxel size under every registered spelling (canonical, legacy, OME, ``voxel_size``); returns it."""
    vs = get_voxel_size(metadata, dx=dx, dy=dy, dz=dz)
    metadata.update(vs.to_dict(include_aliases=True))
    return metadata


def normalize_metadata(
    metadata: dict,
    shape: tuple | None = None,
    **overrides,
) -> dict:
    """Stamp ``metadata`` in place with every registered parameter under its canonical key, aliases and transform aliases (write time only, §6.4); returns it.

    ``overrides`` (``dx=0.5``, ``fs=30.0``) win over stored values; ``shape``
    supplies ``Lx`` / ``Ly`` when absent.
    """
    vs = get_voxel_size(
        metadata,
        dx=overrides.get("dx"),
        dy=overrides.get("dy"),
        dz=overrides.get("dz"),
    )
    metadata.update(vs.to_dict(include_aliases=True))

    for name, param in METADATA_PARAMS.items():
        if name in ("dx", "dy", "dz"):
            continue
        value = get_param(metadata, name, override=overrides.get(name), shape=shape)
        if value is not None:
            metadata[name] = value
            for alias in param.aliases:
                metadata[alias] = value
            for src_key, (_to, from_canonical) in param.transforms.items():
                try:
                    metadata[src_key] = from_canonical(value)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

    return metadata
