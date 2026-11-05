"""The Studio's open dialogs, and loading what they return into the running viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities.arrays import ScanImageArray
from mbo_utilities.gui._files import PathPrompt, draw_path_prompt
from mbo_utilities.lazy_array import base_array
from mbo_utilities.metadata.base import PROVENANCE_PATH_KEYS
from mbo_utilities.preferences import add_recent_file, get_last_dir, set_last_dir
from mbo_utilities.reader import imread

_IMAGE_FILTERS = ["All Files", "*"]


def open_prompts(parent: Any) -> tuple[PathPrompt, PathPrompt]:
    """The typed-path popups for Open File and Open Folder, made on first use."""
    file_prompt = getattr(parent, "_open_file_prompt", None)
    if file_prompt is None:
        file_prompt = PathPrompt(
            "Open file", action="open", hint="one image file, read by this process"
        )
        parent._open_file_prompt = file_prompt
    folder_prompt = getattr(parent, "_open_folder_prompt", None)
    if folder_prompt is None:
        folder_prompt = PathPrompt(
            "Open folder",
            action="open",
            hint="a folder of image files, read by this process",
        )
        parent._open_folder_prompt = folder_prompt
    return file_prompt, folder_prompt


def _open_start_dir(parent: Any, kind: str) -> str:
    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
    if fpath and Path(fpath).exists():
        return str(Path(fpath).parent)
    return str(get_last_dir(f"open_{kind}") or Path.home())


def start_open_prompt(parent: Any, kind: str = "file") -> None:
    """Ask for a path to open: a typed field, with the native dialog as a
    browse shortcut. ``kind`` is ``"file"`` or ``"folder"``.

    The dialog opens where the process runs, which for a notebook served
    from another machine is the wrong one, so the field is the route that
    always works. Nothing happens while a native dialog is already up.
    """
    if parent._file_dialog is not None or parent._folder_dialog is not None:
        return
    file_prompt, folder_prompt = open_prompts(parent)
    prompt = folder_prompt if kind == "folder" else file_prompt
    other = file_prompt if kind == "folder" else folder_prompt
    other.open = False
    prompt.start(_open_start_dir(parent, kind))


def _browse(parent: Any, kind: str, start: str) -> None:
    """Launch the native dialog for the prompt's browse button."""
    if parent._file_dialog is not None or parent._folder_dialog is not None:
        return
    start = start if start and Path(start).exists() else _open_start_dir(parent, kind)
    if Path(start).is_file():
        start = str(Path(start).parent)
    if kind == "folder":
        parent._folder_dialog = pfd.select_folder("Select Data Folder", start)
    else:
        parent._file_dialog = pfd.open_file(
            "Select Data File(s)", start, _IMAGE_FILTERS, pfd.opt.multiselect
        )


def _draw_open_prompts(parent: Any) -> None:
    """Draw the two path popups and act on what they return."""
    file_prompt, folder_prompt = open_prompts(parent)
    for kind, prompt in (("file", file_prompt), ("folder", folder_prompt)):
        path, browse = draw_path_prompt(prompt)
        if browse:
            _browse(parent, kind, prompt.path)
        if path is None:
            continue
        p = Path(path).expanduser()
        if not p.exists():
            prompt.status = f"not found on this machine: {p}"
            continue
        if kind == "folder" and not p.is_dir():
            prompt.status = f"not a folder: {p}"
            continue
        prompt.open = False
        add_recent_file(str(p), file_type=kind)
        set_last_dir(f"open_{kind}", str(p))
        load_new_data(parent, str(p))


def _resolve_plane_dir(p: Path | None) -> Path | None:
    """The suite2p plane dir ``p`` names: a ``data.bin``'s folder, the plane dir itself, or the first plane child of a volume root."""
    if p is None:
        return None
    if p.is_file() and p.name in ("data.bin", "data_raw.bin"):
        return p.parent
    if not p.is_dir():
        return None
    if any(
        (p / n).is_file()
        for n in ("ops.npy", "settings.npy", "data.bin", "data_raw.bin")
    ):
        return p
    for child in sorted(p.iterdir()):
        if child.is_dir() and (child / "ops.npy").is_file():
            return child
    return None


def _npy_dict(fp: Path) -> dict | None:
    arr = np.load(str(fp), allow_pickle=True)
    d = arr.item() if arr.ndim == 0 else arr
    return d if isinstance(d, dict) else None


def _stale_recorded_path(records: list[dict]) -> str | None:
    """The first provenance path in ``records`` that does not exist on this machine."""
    for record in records:
        for key in PROVENANCE_PATH_KEYS:
            value = record.get(key)
            entries = value if isinstance(value, (list, tuple)) else [value]
            for entry in entries:
                s = str(entry) if isinstance(entry, (str, Path)) else ""
                if ("/" in s or "\\" in s) and not Path(s).exists():
                    return s
    return None


def _try_hydrate_s2p_from_binary(parent: Any, path: str | Path | None) -> bool:
    """Fill ``parent.s2p`` / ``parent.s2p_db`` from a suite2p plane dir's ``settings.npy`` + ``db.npy`` (else ``ops.npy``) and point ``parent._s2p_outdir`` at its run root, so a re-run lands beside it.

    True when any setting was applied. Nothing happens when ``path`` is not
    suite2p output, has no settings files, or suite2p is not installed.
    """
    from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
        from_flat,
        from_structured,
        warm_up_suite2p_schema,
    )

    plane_dir = _resolve_plane_dir(None if path is None else Path(path))
    if plane_dir is None:
        return False
    s2p, s2p_db, s2p_extras = parent.s2p, parent.s2p_db, parent.s2p_extras
    if s2p is None or s2p_db is None:
        parent.logger.info("suite2p hydrate: suite2p not installed, skipping")
        return False

    # `is_default` answers True while the schema is unloaded, so nothing would show as modified
    warm_up_suite2p_schema()

    loaded: dict = {}
    sources: list[str] = []
    records: list[dict] = []
    settings = (
        _npy_dict(plane_dir / "settings.npy")
        if (plane_dir / "settings.npy").is_file()
        else None
    )
    if settings:
        loaded.update(from_structured(settings))
        sources.append("settings.npy")
    db = _npy_dict(plane_dir / "db.npy") if (plane_dir / "db.npy").is_file() else None
    if db:
        records.append(db)
        loaded.update(from_flat(db))
        sources.append("db.npy")
    ops = (
        _npy_dict(plane_dir / "ops.npy") if (plane_dir / "ops.npy").is_file() else None
    )
    if ops:
        records.append(ops)
        # ops.npy alone carries lsp's dff knobs; settings.npy is canonical for everything they share
        new_keys = [k for k in from_flat(ops) if k not in loaded]
        loaded.update({k: from_flat(ops)[k] for k in new_keys})
        if new_keys:
            sources.append("ops.npy")
            parent.logger.debug(
                f"suite2p hydrate: ops.npy contributed {len(new_keys)} fields "
                f"not in settings.npy/db.npy: {sorted(new_keys)}"
            )
    if not loaded:
        parent.logger.info(
            f"suite2p hydrate: no settings/db/ops files at {plane_dir}; "
            "leaving pipeline settings untouched."
        )
        return False

    # NOTE: a registration-only pass writes roidetect=0; restoring it turned Detection to Skip for
    # every later run. the Skip/Run/Force gates stay where the user left them.
    for gate in ("do_detection", "do_registration"):
        loaded.pop(gate, None)
    # cellpose_model doubles as a bare model name ('cpsam'), so only a value with a separator is a path
    for key in ("classifier_path", "cellpose_model"):
        value = loaded.get(key)
        if (
            isinstance(value, str)
            and ("/" in value or "\\" in value)
            and not Path(value).exists()
        ):
            loaded.pop(key)
            parent.logger.info(
                f"suite2p hydrate: dropping {key}={value!r} (path does not exist on this machine)"
            )

    n_applied = 0
    skipped: list[str] = []
    targets = [t for t in (s2p, s2p_db, s2p_extras) if t is not None]
    for field, value in loaded.items():
        target = next((t for t in targets if hasattr(t, field)), None)
        if target is None:
            skipped.append(field)
            continue
        if isinstance(value, np.ndarray) and value.size > 1:
            parent.logger.debug(
                f"suite2p hydrate: skipping {field}: got ndarray size={value.size}, expected scalar"
            )
            continue
        cur = getattr(target, field)
        if value is not None and cur is not None:
            try:
                if isinstance(cur, bool):
                    value = bool(value)
                elif isinstance(cur, int) and not isinstance(value, bool):
                    value = int(value)
                elif isinstance(cur, float):
                    value = float(value)
                elif isinstance(cur, str):
                    value = str(value)
            except (TypeError, ValueError) as e:
                parent.logger.debug(
                    f"suite2p hydrate: type-coerce skipped for {field}: {e}"
                )
                continue
        setattr(target, field, value)
        n_applied += 1

    # plane_dir is <root>/zplane01_tp00001-01574; re-running with output=<root> recreates it in place
    parent._s2p_outdir = str(plane_dir.parent)
    stale = _stale_recorded_path(records)
    if stale:
        parent.logger.info(
            f"suite2p hydrate: run was recorded on another machine ({stale}); "
            f"outputs rebased to {parent._s2p_outdir}"
        )
    parent.logger.debug(
        f"suite2p hydrate: loaded {n_applied} fields from {' + '.join(sources)}; "
        f"output dir set to {parent._s2p_outdir}"
        + (f" (skipped {len(skipped)} unmapped keys)" if skipped else "")
    )
    return True


def check_file_dialogs(parent: Any):
    """Draw the open prompts, and load whatever a native dialog returned."""
    _draw_open_prompts(parent)
    file_prompt, folder_prompt = open_prompts(parent)

    if parent._file_dialog is not None and parent._file_dialog.ready():
        result = parent._file_dialog.result()
        parent.logger.info(f"File dialog result: {result}")
        if result:
            file_prompt.open = False
            file_prompt.path = result[0]
            add_recent_file(result[0], file_type="file")
            set_last_dir("open_file", result[0])
            load_new_data(parent, result[0])
        else:
            parent.logger.info("File dialog cancelled or empty result")
        parent._file_dialog = None

    if parent._folder_dialog is not None and parent._folder_dialog.ready():
        result = parent._folder_dialog.result()
        if result:
            folder_prompt.open = False
            folder_prompt.path = result
            add_recent_file(result, file_type="folder")
            set_last_dir("open_folder", result)
            load_new_data(parent, result)
        parent._folder_dialog = None


def _reset_per_data_state(parent: Any) -> None:
    """Put every per-dataset display setting back to its launch default.

    The side panel's controls bind to attributes on the parent, which
    ``_refresh_widgets`` does not rebuild; ``PreviewDataWidget._init_state``
    uses the same defaults.
    """
    parent._mean_subtraction = False
    parent._invert_deflection = False
    parent._mean_images = {}
    parent._gaussian_sigma = 0.0
    parent._proj = "mean"
    parent._window_size = 1
    parent._frame_average = 1
    parent._frame_average_source = None  # the un-averaged array is the old file's
    parent._saveas_frame_average = 1
    parent._s2p_frame_average = 1
    parent._masknmf_frame_average = 1
    parent._auto_contrast_on_z = False
    parent._saveas_selected_roi = set()
    parent._saveas_rois = False


def swap_viewer_array(parent: Any, arr, title: str | None = None) -> None:
    """Show ``arr`` (a sibling of the open recording, say another MESc unit) in place of the current array, keeping the viewer.

    The spatial func captured the previous array's mean image, so it is
    rebuilt before the swap; the slider labels are re-stamped with the data
    since a swap can change both their count and meaning.
    """
    from mbo_utilities.arrays.features import get_slider_dims
    from mbo_utilities.gui.run_gui import _squeeze_for_viewer
    from mbo_utilities.gui.viewers import TimeSeriesViewer

    iw = parent.image_widget
    _reset_per_data_state(parent)
    parent._rebuild_spatial_func()

    display = _squeeze_for_viewer(arr)
    iw.data[0] = display
    custom_labels = getattr(arr, "slider_dim_labels", None)
    iw._slider_dim_names = (
        tuple(custom_labels) if custom_labels else get_slider_dims(arr)
    ) or None
    if iw.n_sliders > 0:
        iw.indices = [0] * iw.n_sliders

    parent.shape = display.shape
    _, parent.nc, parent.nz = arr.shape[:3]
    parent._custom_metadata = {}
    parent._update_window_funcs()
    parent.set_context_info()
    if title:
        iw.figure[0, 0].title = title
    if isinstance(parent._viewer, TimeSeriesViewer):
        parent.refresh_zstats()
    parent._refresh_widgets()


def load_new_data(parent: Any, path: str):
    """Open ``path`` in the running viewer: new array, viewer class, widgets and ROI tool; the status line reports the outcome."""
    from mbo_utilities.arrays import TiffArray
    from mbo_utilities.gui.run_gui import _squeeze_for_viewer, _wrap_for_viewer
    from mbo_utilities.gui.viewers import TimeSeriesViewer, get_viewer_class
    from mbo_utilities.gui.widgets.pipelines.isoview import maybe_spawn_raw_projections

    path_obj = Path(path)
    if not path_obj.exists():
        parent.logger.error(f"Path does not exist: {path}")
        return

    try:
        parent.logger.info(f"Loading data from: {path}")
        raw_data = _wrap_for_viewer(imread(path), parent.logger)
        new_data = _squeeze_for_viewer(raw_data)
        _reset_per_data_state(parent)

        # the ROI widget is bound to the old data's shape and graphics, so it goes before the swap;
        # its parked state goes too, or a same-shape file would adopt it
        roi_was_on = parent.manual_roi is not None
        if roi_was_on:
            from mbo_utilities.gui.manual_roi import detach_roi_widget

            detach_roi_widget(parent)
        parent._manual_roi_store = None
        parent._manual_roi_runs = None

        parent.image_widget.data[0] = (
            new_data  # clears window and spatial funcs, every slider at 0
        )
        parent._rebuild_spatial_func()
        parent.fpath = path
        parent.shape = new_data.shape
        parent.is_mbo_scan = isinstance(
            base_array(raw_data), (ScanImageArray, TiffArray)
        )
        if not _try_hydrate_s2p_from_binary(parent, path) and not parent._s2p_outdir:
            parent._s2p_outdir = str(path_obj.parent / "suite2p")
        _, parent.nc, parent.nz = raw_data._shape5d()[:3]
        parent.set_context_info()

        base_name = (
            path_obj.parent.name
            if path_obj.stem in ("data", "data_raw")
            else path_obj.stem
        )
        if len(base_name) > 24:
            base_name = base_name[:21] + "..."
        parent.image_widget.figure[0, 0].title = base_name

        parent._refresh_widgets()
        parent._custom_metadata = {}
        if parent._viewer:
            parent._viewer.cleanup()
        parent._viewer = get_viewer_class(new_data)(
            parent.image_widget, parent.fpath, parent=parent
        )
        parent._viewer.on_data_loaded()
        parent.logger.debug(f"Viewer switched to: {parent._viewer.name}")
        if roi_was_on:
            from mbo_utilities.gui.manual_roi import attach_roi_widget

            attach_roi_widget(parent)
        parent._seed_playback_fps()
        maybe_spawn_raw_projections(parent)
        if isinstance(parent._viewer, TimeSeriesViewer):
            parent.refresh_zstats()
    except Exception as e:
        parent.logger.exception(f"Error loading data: {e}")
