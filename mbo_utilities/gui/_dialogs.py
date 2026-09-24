"""
File dialog handlers and data loading.

This module contains file/folder dialog handling and data loading logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_IMAGE_FILTERS = ["All Files", "*"]


_PLANE_DIR_RE = re.compile(r"^plane\d+$", re.IGNORECASE)


def outdir_from_fpath(fpath) -> str | None:
    """Default output dir from a loaded fpath: the parent folder when fpath
    is a file, or the folder itself when fpath IS a directory. Used to
    seed the Run-tab output field so re-runs land alongside the source
    data unless the user explicitly browses elsewhere.
    """
    if fpath is None:
        return None
    if isinstance(fpath, (list, tuple)):
        if not fpath:
            return None
        fpath = fpath[0]
    try:
        p = Path(str(fpath))
    except (TypeError, ValueError):
        return None
    if not p.exists():
        return None
    return str(p if p.is_dir() else p.parent)


def suite2p_output_dir(fpath) -> str | None:
    """Detect a suite2p output location from a file or directory path.

    Returns the directory suite2p would have written into (the parent of
    the `plane*/` subdirs), or None if `fpath` doesn't look like a suite2p
    output. Used to auto-populate the GUI's output-folder field when the
    user opens an existing data.bin / ops.npy / volumetric results dir.

    Cases handled:
      - file inside `…/<root>/plane0/` (e.g. data.bin, ops.npy) → `<root>`
      - directory `…/<root>/plane0/`                            → `<root>`
      - directory `…/<root>/` containing one or more `plane*/`  → `<root>`
    """
    if fpath is None:
        return None
    if isinstance(fpath, (list, tuple)):
        if not fpath:
            return None
        fpath = fpath[0]
    try:
        p = Path(str(fpath))
    except (TypeError, ValueError):
        return None
    if not p.exists():
        return None

    parent = p.parent if p.is_file() else p
    if _PLANE_DIR_RE.match(parent.name):
        return str(parent.parent)
    try:
        for child in parent.iterdir():
            if child.is_dir() and _PLANE_DIR_RE.match(child.name):
                return str(parent)
    except (OSError, PermissionError):
        pass
    return None


def _resolve_plane_dir(p: Path) -> Path | None:
    """Return the plane directory for `p` if it looks like a suite2p output.

    Accepts:
      - `…/plane_dir/data.bin` or `…/plane_dir/data_raw.bin` → `plane_dir`
      - `…/plane_dir/` itself (contains ops.npy / settings.npy / data.bin)
      - `…/volume_root/` (contains one or more `plane*/` subdirs) →
        first plane subdir (settings/db.npy are identical across planes
        — they were passed into lsp.pipeline once)
    """
    if p.is_file() and p.name in ("data.bin", "data_raw.bin"):
        return p.parent
    if p.is_dir():
        if any(
            (p / n).is_file()
            for n in ("ops.npy", "settings.npy", "data.bin", "data_raw.bin")
        ):
            return p
        try:
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / "ops.npy").is_file():
                    return child
        except (OSError, PermissionError):
            return None
    return None


def _try_hydrate_s2p_from_binary(parent: Any, path: str | Path) -> bool:
    """If `path` points at suite2p output (data.bin file, plane dir, or
    volumetric root containing plane*/ subdirs), hydrate parent.s2p /
    parent.s2p_db from the plane folder's settings files AND point
    parent._s2p_outdir at the suite2p output ROOT so re-running the
    pipeline recreates the same layout in the same place.

    Source-of-truth preference:
      1. settings.npy + db.npy (canonical upstream-shape pair, exactly
         what was passed into lsp.pipeline)
      2. ops.npy (flat fused dict — may have been mutated by reactive
         fs/dz rescaling, includes detection outputs)

    No-op when the path doesn't resolve to a plane dir or none of the
    sibling settings files exist. Returns True iff settings were applied.
    """
    plane_dir = _resolve_plane_dir(Path(path))
    if plane_dir is None:
        return False
    settings_file = plane_dir / "settings.npy"
    db_file = plane_dir / "db.npy"
    ops_file = plane_dir / "ops.npy"

    import numpy as np  # local — keep _dialogs.py import surface tight

    def _load_npy_dict(fp: Path) -> dict | None:
        try:
            arr = np.load(str(fp), allow_pickle=True)
            d = arr.item() if hasattr(arr, "item") and arr.ndim == 0 else arr
            return d if isinstance(d, dict) else None
        except Exception as e:
            parent.logger.warning(f"suite2p hydrate: failed to read {fp}: {e}")
            return None

    loaded: dict = {}
    sources: list[str] = []
    try:
        from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
            from_flat as _from_flat,
        )
        from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
            from_structured as _from_structured,
        )
        from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
            warm_up_suite2p_schema as _warm_up_schema,
        )
    except Exception as e:
        parent.logger.warning(f"suite2p hydrate: schema import failed: {e}")
        return False
    # `is_default` answers True while the schema is unloaded, so nothing would
    # show as modified until some later import
    try:
        _warm_up_schema()
    except Exception as e:
        parent.logger.warning(f"suite2p hydrate: schema warm-up failed: {e}")

    raw_db: dict | None = None
    raw_ops: dict | None = None
    if settings_file.is_file():
        d = _load_npy_dict(settings_file)
        if d:
            loaded.update(_from_structured(d))
            sources.append("settings.npy")
    if db_file.is_file():
        d = _load_npy_dict(db_file)
        if d:
            raw_db = d
            # db.npy is flat (paths/nplanes/keep_movie_raw at top level),
            # so route via from_flat — it covers keep_movie_raw and any
            # other db fields with entries in _FLAT_TO_MBO.
            loaded.update(_from_flat(d))
            sources.append("db.npy")
    # ops.npy alone carries lsp's dff knobs; setdefault, because settings.npy is
    # canonical for everything the two share
    if ops_file.is_file():
        d = _load_npy_dict(ops_file)
        if d:
            raw_ops = d
            ops_view = _from_flat(d)
            new_keys = []
            for k, v in ops_view.items():
                if k not in loaded:
                    loaded[k] = v
                    new_keys.append(k)
            if new_keys:
                if "ops.npy" not in sources:
                    sources.append("ops.npy")
                parent.logger.debug(
                    f"suite2p hydrate: ops.npy contributed {len(new_keys)} fields "
                    f"not in settings.npy/db.npy: {sorted(new_keys)}"
                )

    if not loaded:
        parent.logger.info(
            f"suite2p hydrate: no settings/db/ops sibling files at {plane_dir}; "
            "leaving pipeline settings untouched."
        )
        return False

    # The Skip / Run / Force gates say what one past run did, not what the
    # user wants next. A registration pass writes roidetect=0 into the plane
    # dir's ops.npy (register() runs suite2p with detection off), so
    # hydrating that key turned Detection to Skip for every later run and
    # suite2p quietly regenerated figures instead of finding ROIs
    # ("Suite2p disabled by user toggles"). Restore parameters, not gates.
    for _gate in ("do_detection", "do_registration"):
        if _gate in loaded:
            loaded.pop(_gate)
            parent.logger.debug(
                f"suite2p hydrate: ignoring {_gate} from the run's records; "
                "the Skip/Run/Force gates stay where the user left them"
            )

    # dead custom file paths recorded on another machine fail deep inside
    # suite2p later; drop them so the run falls back to defaults.
    # cellpose_model doubles as a bare model NAME ('cpsam'), so only
    # separator-containing values are treated as paths and validated.
    for _pf in ("classifier_path", "cellpose_model"):
        try:
            v = loaded.get(_pf)
            if (
                isinstance(v, str)
                and v
                and ("/" in v or "\\" in v)
                and not Path(v).exists()
            ):
                loaded.pop(_pf)
                parent.logger.info(
                    f"suite2p hydrate: dropping {_pf}={v!r} "
                    "(path does not exist on this machine)"
                )
        except Exception:
            pass

    # access via lazy properties so the dataclasses get instantiated
    # if they haven't been touched yet.
    s2p = getattr(parent, "s2p", None)
    s2p_db = getattr(parent, "s2p_db", None)
    s2p_extras = getattr(parent, "s2p_extras", None)
    if s2p is None or s2p_db is None:
        parent.logger.info(
            "suite2p hydrate: Suite2pSettings/Suite2pDB unavailable "
            "(suite2p not installed?), skipping."
        )
        return False

    n_settings = 0
    n_db = 0
    n_extras = 0
    skipped: list[str] = []
    for field, value in loaded.items():
        target = None
        if hasattr(s2p, field):
            target = s2p
        elif hasattr(s2p_db, field):
            target = s2p_db
        elif s2p_extras is not None and hasattr(s2p_extras, field):
            target = s2p_extras
        else:
            skipped.append(field)
            continue
        # a leaked ndarray would crash bool()/int()/float(); best-effort, a failure
        # leaves the well-typed default
        cur = getattr(target, field)
        if value is not None and cur is not None:
            # defensive: skip multi-element arrays — they shouldn't
            # reach a scalar field, but if they do, leaving the
            # default is safer than raising.
            if type(value).__name__ == "ndarray" and getattr(value, "size", 1) > 1:
                parent.logger.debug(
                    f"suite2p hydrate: skipping {field}: got ndarray "
                    f"size={value.size}, expected scalar"
                )
                continue
            try:
                if isinstance(cur, bool):
                    value = bool(value)
                elif isinstance(cur, int) and not isinstance(value, bool):
                    value = int(value)
                elif isinstance(cur, float):
                    value = float(value)
                elif isinstance(cur, str):
                    value = str(value)
            except Exception as e:
                parent.logger.debug(
                    f"suite2p hydrate: type-coerce skipped for {field}: {e}"
                )
                continue
        try:
            setattr(target, field, value)
            if target is s2p:
                n_settings += 1
            elif target is s2p_db:
                n_db += 1
            else:
                n_extras += 1
        except Exception as e:
            parent.logger.debug(
                f"suite2p hydrate: could not set {field}={value!r}: {e}"
            )

    # plane_dir is e.g. .../res/zplane01_tp00001-01574 — the ROOT is its
    # parent (.../res). re-running with output=ROOT recreates the same
    # plane subdir in place.
    parent._s2p_outdir = str(plane_dir.parent)

    # one-line heads-up when the run dir was copied from another machine:
    # its recorded paths are dead here, and re-runs target _s2p_outdir.
    try:
        from mbo_utilities.metadata.base import PROVENANCE_PATH_KEYS

        stale_example = None
        for src in (raw_db, raw_ops):
            if not isinstance(src, dict):
                continue
            for key in PROVENANCE_PATH_KEYS:
                v = src.get(key)
                entries = v if isinstance(v, (list, tuple)) else [v]
                for entry in entries:
                    s = str(entry) if isinstance(entry, (str, Path)) else ""
                    if not s or ("/" not in s and "\\" not in s):
                        continue
                    try:
                        dead = not Path(s).exists()
                    except OSError:
                        dead = True
                    if dead:
                        stale_example = s
                        break
                if stale_example:
                    break
            if stale_example:
                break
        if stale_example:
            parent.logger.info(
                f"suite2p hydrate: run was recorded on another machine "
                f"({stale_example}); outputs rebased to {parent._s2p_outdir}"
            )
    except Exception:
        pass

    parent.logger.debug(
        f"suite2p hydrate: loaded {n_settings} settings + {n_db} db + "
        f"{n_extras} mbo-extras fields from {' + '.join(sources)}; "
        f"output dir set to {parent._s2p_outdir}"
        + (f" (skipped {len(skipped)} unmapped keys)" if skipped else "")
    )
    return True
