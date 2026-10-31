"""Worker tasks, one ``task_<name>(args, logger)`` per pipeline, and the ``TASKS`` registry."""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path

import psutil

from mbo_utilities import imread, log
from mbo_utilities.arrays._registration import (
    compute_axial_shifts,
    validate_axial_shifts,
)
from mbo_utilities.arrays.features import apply_read_features
from mbo_utilities.metadata import get_param
from mbo_utilities.preferences import get_debug_logging, get_mbo_dirs
from mbo_utilities.writer import imwrite

logger = log.get("worker.tasks")


def _auto_workers(num_planes: int, *, use_gpu: bool = False) -> int:
    """Worker count bounded by the planes, the cpus minus two, 8 GB of free RAM each, and 2 on a gpu."""
    cpu_workers = max(1, (os.cpu_count() or 4) - 2)
    avail_gb = psutil.virtual_memory().available / (1024**3)
    mem_workers = max(1, int(avail_gb // 8))
    cap = 2 if use_gpu else 12
    return max(1, min(num_planes, cpu_workers, mem_workers, cap))


class TaskMonitor:
    """Writes a task's progress to the JSON sidecar the Process Console reads."""

    def __init__(self, output_dir: Path | str, uuid: str | None = None):
        self.output_dir = Path(output_dir)
        self.pid = os.getpid()
        self.uuid = uuid
        self.log_dir = get_mbo_dirs()["logs"]
        stem = self.uuid if self.uuid else self.pid
        self.progress_file = self.log_dir / f"progress_{stem}.json"

    def update(
        self,
        progress: float,
        message: str,
        state: str = "running",
        details: dict | None = None,
    ):
        """Write ``progress`` (0 to 1), ``message`` and ``state`` (running, completed, error)."""
        data = {
            "pid": self.pid,
            "uuid": self.uuid,
            "timestamp": time.time(),
            "status": state,
            "progress": progress,
            "message": message,
            "details": details or {},
        }
        tmp_file = self.progress_file.with_suffix(".tmp")
        with tmp_file.open("w") as f:
            json.dump(data, f)
        # windows denies the rename while the gui holds the sidecar open for read
        for _ in range(10):
            try:
                tmp_file.replace(self.progress_file)
                return
            except PermissionError:
                time.sleep(0.05)

    def finish(self, message: str = "Task completed"):
        self.update(1.0, message, state="completed")

    def fail(self, error: str, details: str | dict | None = None):
        self.update(0.0, f"Error: {error}", state="error", details=details)


def _save_progress(monitor: TaskMonitor, current, total=None, **kwargs) -> None:
    """Map a writer's progress onto 0.1 to 1.0.

    Writers call back as ``cb(fraction)``, ``cb(current, total)`` or
    ``cb(fraction, message)``, so a numeric ``total`` is the only way to tell
    the forms apart.
    """
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        fraction = float(current) / max(float(total), 1.0)
        message = f"Writing frame {current}/{total}"
    else:
        fraction = float(current)
        message = "Writing..." if total is None else str(total)
    monitor.update(0.1 + 0.9 * fraction, message)


def task_save_as(args: dict, logger: logging.Logger) -> None:
    """Save any readable array as ``.zarr``, ``.h5``, ``.tiff``, ``.bin``, ..."""
    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.0, "Initializing save task...")

    input_path = args["input_path"]
    output_path = Path(args["output_path"])
    output_dir = args.get("output_dir")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monitor.update(0.05, f"Loading {Path(input_path).name}...")
    logger.info(f"Loading {input_path}")
    arr = imread(input_path, **(args.get("reader_kwargs") or {}))
    # reader_kwargs may already carry the viewer's frame_average; the explicit arg wins
    arr, _ = apply_read_features(
        arr,
        fix_phase=args.get("fix_phase", True),
        use_fft=args.get("use_fft", True),
        frame_average=args.get("frame_average", None),
    )

    ext = args.get("ext") or output_path.suffix or ".zarr"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monitor.update(0.1, f"Saving to {output_path.name}...")
    logger.info(f"Writing to {output_path} with ext={ext}")
    try:
        imwrite(
            arr,
            output_path,
            ext=ext,
            planes=args.get("planes"),
            channels=args.get("channels"),
            timepoints=args.get("frames"),
            roi=args.get("rois"),
            register_z=bool(args.get("register_z", False)),
            max_frames=int(args.get("max_frames", 200)),
            chunk_frames=int(args.get("chunk_frames", 10)),
            max_reg_xy=int(args.get("max_reg_xy", 30)),
            overwrite=True,
            metadata=args.get("metadata", {}),
            progress_callback=functools.partial(_save_progress, monitor),
            **args.get("kwargs", {}),
        )
        monitor.finish(f"Saved to {output_path.name}")
        logger.info("Save completed")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        raise


def _suite2p_progress(
    monitor: TaskMonitor, plane=0, total_planes=1, step="", message="", **kw
) -> None:
    """Map lbm_suite2p_python's (plane, step) progress onto 0.1 to 0.95."""
    base = 0.1 + 0.85 * (plane / max(total_planes, 1))
    offsets = {
        "plane_start": 0.0,
        "writing_binary": 0.01,
        "suite2p": 0.05,
        "postprocessing": 0.70,
        "plane_done": 0.85 / max(total_planes, 1),
        "done": 0.85 / max(total_planes, 1),
    }
    monitor.update(min(base + offsets.get(step, 0), 0.95), message)


def task_suite2p(args: dict, logger: logging.Logger) -> None:
    """Run ``lbm_suite2p_python.pipeline`` on the selected planes with the Run tab's ops."""
    from lbm_suite2p_python import pipeline

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing Suite2p pipeline...")

    input_path = args["input_path"]
    output_dir = Path(args["output_dir"])
    planes = args.get("planes")
    # the gui sends -1 for "all frames"; as a literal nframes it would name the plane dir "zplane01"
    num_timepoints = args.get("num_timepoints")
    if num_timepoints is not None and num_timepoints <= 0:
        num_timepoints = None
    # the config is a flat ops dict or lsp's (settings, db) pair; the rest of the task mutates flat ops
    ops = dict(args.get("ops") or {})
    incoming_settings = args.get("settings")
    incoming_db = args.get("db")
    if incoming_settings is not None or incoming_db is not None:
        from lbm_suite2p_python.db_settings import db_settings_to_ops

        flattened = db_settings_to_ops(incoming_db, incoming_settings)
        flattened.update(ops)
        ops = flattened
    s2p_settings = args.get("s2p_settings", {})
    logger.info(
        f"task_suite2p: parallel knobs from GUI -> "
        f"workers={s2p_settings.get('workers')!r}, "
        f"threads_per_worker={s2p_settings.get('threads_per_worker')!r}, "
        f"skip_volumetric={s2p_settings.get('skip_volumetric')!r}; "
        f"s2p_settings has workers key: {'workers' in s2p_settings}"
    )

    # the metadata editor's values win over ops defaults and the source file
    custom_metadata = args.get("custom_metadata") or {}
    if custom_metadata:
        ops.update(custom_metadata)
        logger.info(f"Applied custom metadata to ops: {sorted(custom_metadata)}")

    from mbo_utilities.metadata import OutputMetadata

    _src_arr = imread(input_path, **(args.get("reader_kwargs") or {}))
    src_meta = dict(_src_arr.metadata or {})
    src_shape = tuple(_src_arr._shape5d())
    src_meta.update(custom_metadata)

    raw_src_fs = get_param(src_meta, "fs")
    raw_src_dz = get_param(src_meta, "dz")
    logger.info(
        f"task_suite2p: source fs={raw_src_fs}, dz={raw_src_dz}, "
        f"input_path={input_path!r}"
    )

    if raw_src_fs is None:
        logger.warning(
            "task_suite2p: source metadata has NO fs field — "
            "ops.npy fs will fall through to lbm_suite2p_python's default "
            "(10 Hz). To fix: set fs via the metadata editor, or fix the "
            "source TIFF metadata."
        )
    elif ops.get("fs") in (None, 10.0):
        # lsp's default_ops ships fs=10.0, which leaked into ops.npy as the source rate
        ops["fs"] = float(raw_src_fs)
        logger.info(
            f"task_suite2p: replaced lbm default fs=10 with source fs={raw_src_fs}"
        )

    if raw_src_dz is not None and ops.get("dz") in (None, 1.0):
        ops["dz"] = float(raw_src_dz)
        logger.info(f"task_suite2p: pulled source dz={raw_src_dz} into ops")

    tp_indices = args.get("tp_indices")
    selected_planes_0based = args.get("selected_planes_0based")
    if tp_indices is not None or selected_planes_0based is not None:
        selections = {}
        if tp_indices is not None:
            selections["T"] = list(tp_indices)
        if selected_planes_0based is not None:
            selections["Z"] = list(selected_planes_0based)
        scaled = OutputMetadata(
            source=src_meta,
            source_shape=src_shape,
            source_dims=("T", "C", "Z", "Y", "X"),
            selections=selections,
        ).to_dict()
        # a scaled None removes fs and dz so nothing downstream reads a stale default
        for key in (
            "fs",
            "dz",
            "dx",
            "dy",
            "z_step",
            "umPerPixZ",
            "umPerPixX",
            "umPerPixY",
        ):
            if scaled.get(key) is not None:
                ops[key] = scaled[key]
            elif key in ("fs", "dz"):
                ops.pop(key, None)
        logger.info(
            f"task_suite2p: applied reactive metadata -> "
            f"fs={ops.get('fs')}, dz={ops.get('dz')} "
            f"(t-stride from {len(tp_indices) if tp_indices else 0} indices, "
            f"z-stride from {len(selected_planes_0based) if selected_planes_0based else 0} planes)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(input_path, list):
        if len(input_path) == 1:
            input_path = input_path[0]
            display_name = Path(input_path).name
        else:
            display_name = f"{len(input_path)} files ({Path(input_path[0]).name}...)"
    else:
        display_name = Path(input_path).name

    monitor.update(0.05, f"Running pipeline: {display_name}...")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Planes: {planes}")

    # the shifts go to ops["plane_shifts"] for AxialShiftView; the bin pixels stay unshifted
    if args.get("register_z", False) and src_shape[2] > 1:
        num_planes_reg = src_shape[2]
        combined_meta = dict(src_meta)
        if not validate_axial_shifts(combined_meta, num_planes_reg):
            logger.info("task_suite2p: computing axial plane shifts...")
            try:
                compute_axial_shifts(
                    _src_arr,
                    metadata=combined_meta,
                    max_frames=int(args.get("max_frames", 200)),
                    chunk_frames=int(args.get("chunk_frames", 10)),
                    max_reg_xy=int(args.get("max_reg_xy", 30)),
                )
            except Exception as e:  # NOTE: the shifts are an add-on; the binaries are still written, unshifted
                logger.warning(f"task_suite2p: axial registration failed: {e}")
        if validate_axial_shifts(combined_meta, num_planes_reg):
            ops["plane_shifts"] = list(combined_meta["plane_shifts"])
            logger.info(
                f"task_suite2p: axial shifts stored in ops ({num_planes_reg} planes)"
            )
        else:
            logger.warning("task_suite2p: no valid plane_shifts produced.")

    # lsp's generate_plane_dirname needs a frame count for the directory name
    ops["nframes"] = num_timepoints if num_timepoints else int(src_shape[0])

    # each parallel worker re-opens from the path, so the channel travels in reader_kwargs
    channel = args.get("channel")
    pipeline_input = input_path
    reader_kwargs: dict = dict(args.get("reader_kwargs") or {})
    if channel is not None:
        ops["functional_chan"] = 1
        ops["align_by_chan"] = 1
        reader_kwargs["channel"] = int(channel) - 1
        logger.info(
            f"Single-channel extraction: channel {channel} (zero-based: {channel - 1})"
        )

    # workers 0 or None is auto; resolved here so the logs say a real count
    raw_workers = s2p_settings.get("workers")
    if raw_workers in (None, 0):
        n_planes = len(planes) if planes else 1
        from mbo_utilities.gpu import gpu_compute_disabled

        device = str(
            ops.get("torch_device") or s2p_settings.get("torch_device") or "cuda"
        ).lower()
        use_gpu = (not gpu_compute_disabled()) and not device.startswith("cpu")
        resolved = _auto_workers(n_planes, use_gpu=use_gpu)
        if resolved != raw_workers:
            logger.info(
                f"task_suite2p: workers=auto resolved to {resolved} "
                f"(num_planes={n_planes}, cpu={os.cpu_count()})"
            )
            s2p_settings = dict(s2p_settings)
            s2p_settings["workers"] = resolved

    writer_kwargs = {
        "fix_phase": args.get("fix_phase", True),
        "use_fft": args.get("use_fft", True),
        "frame_average": int(args.get("frame_average") or 1),
    }

    # a detection-only re-run reuses the source ops.npy, where an older export left the summary images as json lists
    from mbo_utilities.metadata import repair_ops_tree

    for target in (input_path, output_dir):
        fixed = repair_ops_tree(target)
        if fixed:
            logger.info(
                f"task_suite2p: repaired {fixed} ops.npy file(s) under {target} "
                "(summary images were stored as lists)"
            )

    # lsp reuses a cached model.npy whose isort length matches, so Force has to delete it
    if (
        s2p_settings.get("force_rastermap")
        and s2p_settings.get("rastermap_kwargs") is not None
    ):
        cached_models = list(Path(output_dir).glob("plane*/model.npy"))
        for cached in cached_models:
            cached.unlink()
        if cached_models:
            logger.info(
                f"force_rastermap: removed {len(cached_models)} cached model.npy file(s)"
            )

    # parallel plane workers downloading the cpsam checkpoint to one path corrupt it; fetch it once here
    algo = str(ops.get("algorithm", "")).lower()
    anatomical_on = int(ops.get("anatomical_only", 0) or 0) > 0
    custom_model = ops.get("pretrained_model")
    uses_default_model = not custom_model or str(custom_model).endswith("cpsam")
    if (algo == "cellpose" or anatomical_on) and uses_default_model:
        from mbo_utilities._cellpose_model import ensure_cellpose_model

        ensure_cellpose_model(logger=logger)

    try:
        monitor.update(0.1, "Running Suite2p...")
        if tp_indices:
            logger.info(
                f"task_suite2p: frame_indices len={len(tp_indices)}, "
                f"first={tp_indices[0]}, last={tp_indices[-1]} "
                f"(expected dirname tp{tp_indices[0] + 1:05d}-{tp_indices[-1] + 1:05d})"
            )
        else:
            logger.info(
                "task_suite2p: frame_indices=None (no slice — full range will be used)"
            )

        pipeline(
            pipeline_input,
            save_path=str(output_dir),
            ops=ops,
            planes=planes,
            num_timepoints=num_timepoints,
            frame_indices=tp_indices,
            keep_raw=s2p_settings.get("keep_raw", False),
            keep_reg=s2p_settings.get("keep_reg", True),
            force_reg=s2p_settings.get("force_reg", False),
            force_detect=s2p_settings.get("force_detect", False),
            accept_all_cells=s2p_settings.get("accept_all_cells", False),
            norm_method=s2p_settings.get("norm_method", "dff"),
            dff_window_size=s2p_settings.get("dff_window_size", 300),
            dff_percentile=s2p_settings.get("dff_percentile", 20),
            dff_smooth_window=s2p_settings.get("dff_smooth_window"),
            save_json=s2p_settings.get("save_json", False),
            correct_neuropil=s2p_settings.get("correct_neuropil", True),
            cell_filters=s2p_settings.get("cell_filters") or None,
            rastermap_kwargs=s2p_settings.get("rastermap_kwargs"),
            writer_kwargs=writer_kwargs,
            reader_kwargs=reader_kwargs or None,
            progress_callback=functools.partial(_suite2p_progress, monitor),
            workers=s2p_settings.get("workers", 1),
            skip_volumetric=s2p_settings.get("skip_volumetric", False),
            threads_per_worker=s2p_settings.get("threads_per_worker"),
        )

        monitor.finish("Suite2p pipeline completed.")
        logger.info("Suite2p completed successfully")

    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"Suite2p failed: {e}")
        raise


def _masknmf_progress(
    monitor: TaskMonitor, plane=0, total_planes=1, step="", message="", **kw
) -> None:
    """Map the masknmf runner's (plane, stage) progress onto 0.1 to 0.95."""
    slot = 0.85 / max(total_planes, 1)
    offsets = {
        "registration": 0.02,
        "compression": 0.35,
        "demixing": 0.55,
        "exports": 0.85,
        "figures": 0.92,
    }
    frac = 0.1 + slot * plane + slot * offsets.get(step, 0.0)
    monitor.update(min(frac, 0.95), message)


def _heartbeat(stop: threading.Event, logger: logging.Logger, name: str) -> None:
    """Log every ten minutes so the worker watchdog sees a silent gpu stage as alive."""
    while not stop.wait(600):
        logger.info(f"{name}: still running")


def task_masknmf(args: dict, logger: logging.Logger) -> None:
    """Run ``mbo_utilities.masknmf.run_volume``: registration, PMD compression and demixing per plane, into one run folder each."""
    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing masknmf pipeline...")

    input_path = args["input_path"]
    output_dir = Path(args["output_dir"])
    planes = args.get("planes")
    settings = args.get("settings") or {}
    custom_metadata = args.get("custom_metadata") or {}
    tp_indices = args.get("tp_indices")
    channel = args.get("channel")

    try:
        src_arr = imread(input_path, **(args.get("reader_kwargs") or {}))
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"masknmf: cannot open input {input_path!r}: {e}")
        raise

    # the runner reads fs from here and scales it by the frame stride itself
    metadata = dict(src_arr.metadata or {})
    metadata.update(custom_metadata)

    writer_kwargs = {
        "fix_phase": args.get("fix_phase", True),
        "use_fft": args.get("use_fft", True),
        "frame_average": int(args.get("frame_average") or 1),
        "invert_deflection": bool(args.get("invert_deflection", False)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Planes: {planes}")

    # a stage resuming from a plane dir reads its ops.npy, where an older export left the summary images as json lists
    from mbo_utilities.metadata import repair_ops_tree

    for target in (input_path, output_dir):
        fixed = repair_ops_tree(target)
        if fixed:
            logger.info(
                f"task_masknmf: repaired {fixed} ops.npy file(s) under {target} "
                "(summary images were stored as lists)"
            )

    _stop_heartbeat = threading.Event()
    threading.Thread(
        target=_heartbeat, args=(_stop_heartbeat, logger, "masknmf"), daemon=True
    ).start()

    try:
        monitor.update(0.05, "Running masknmf pipeline...")
        from mbo_utilities.masknmf import run_volume

        runs = run_volume(
            src_arr,
            output_dir,
            planes=planes,
            settings=settings,
            metadata=metadata,
            frame_indices=tp_indices,
            channel=channel,
            writer_kwargs=writer_kwargs,
            logger=logger,
            progress_callback=functools.partial(_masknmf_progress, monitor),
        )

        monitor.finish("masknmf pipeline completed.")
        logger.info(f"masknmf completed: {', '.join(str(r) for r in runs)}")

    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"masknmf failed: {e}")
        raise
    finally:
        _stop_heartbeat.set()


def _bridge_consolidate_logging(worker_logger: logging.Logger) -> None:
    """Attach the worker's handlers to the consolidator's logger.

    Every ``mbo.*`` logger has ``propagate=False`` and the worker's file
    handler sits on ``mbo.worker`` only, so without this the consolidator's
    INFO lines go nowhere in the subprocess.
    """
    cons_logger = logging.getLogger("mbo.arrays.isoview.consolidate")
    cons_logger.setLevel(logging.INFO)
    sources: list[logging.Handler] = list(worker_logger.handlers)
    for h in logging.getLogger("mbo").handlers:
        if h not in sources:
            sources.append(h)
    for h in sources:
        if h not in cons_logger.handlers:
            cons_logger.addHandler(h)


class _ConsolidateProgress:
    """Maps the consolidator's ``(stage, current, total)`` calls onto one bar that never runs backwards.

    Stage weights are the measured share of the 61-timepoint corrected demo:
    image 360 s, seg 240 s, projections 360 s, aux and misc 30 s. A bucket's
    weight is split evenly over the sub-stages seen so far (``image_l0``,
    ``image_l1``, ...).
    """

    WEIGHTS = {"image": 0.40, "seg": 0.15, "proj": 0.30, "aux": 0.10, "misc": 0.05}

    def __init__(self, monitor: TaskMonitor, logger: logging.Logger):
        self.monitor = monitor
        self.logger = logger
        self.consumed = {k: 0.0 for k in self.WEIGHTS}
        self.seen = {k: set() for k in self.WEIGHTS}

    @staticmethod
    def bucket(stage: str) -> str:
        stem = stage.split("_", 1)[0]
        if stem in ("image", "seg", "aux"):
            return stem
        if stem in ("max", "raw"):
            return "proj"
        return "misc"

    def __call__(self, stage: str, current: int, total: int) -> None:
        if total <= 0:
            return
        bucket = self.bucket(stage)
        if stage not in self.seen[bucket]:
            self.seen[bucket].add(stage)
            self.logger.info(f"consolidate: stage '{stage}' starting (total={total})")
        substage_weight = self.WEIGHTS[bucket] / len(self.seen[bucket])
        done = (len(self.seen[bucket]) - 1 + current / total) * substage_weight
        self.consumed[bucket] = max(self.consumed[bucket], done)
        # finalize() pushes the bar to 1.0 on success
        self.monitor.update(
            min(sum(self.consumed.values()), 0.99), f"{stage}: {current}/{total}"
        )


def task_isoview(args: dict, logger: logging.Logger) -> None:
    """Run ``consolidate_isoview`` on a corrected or fused tree."""
    from mbo_utilities.arrays.isoview import consolidate_isoview

    _bridge_consolidate_logging(logger)

    monitor = TaskMonitor(args.get("output_path", "."), uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing isoview consolidator...")

    input_path = args["input_path"]
    output_path = args["output_path"]
    kind = args.get("kind")
    overwrite = bool(args.get("overwrite", False))
    pyramid = bool(args.get("pyramid", True))
    pyramid_max_layers = int(args.get("pyramid_max_layers", 4))
    compressor = args.get("compressor", "zstd")
    compression_level = int(args.get("compression_level", 3))

    logger.info(f"consolidate: input={input_path}")
    logger.info(f"consolidate: output={output_path}")
    logger.info(
        f"consolidate: kind={kind or 'auto'} overwrite={overwrite} "
        f"pyramid={pyramid} max_levels={pyramid_max_layers} "
        f"codec={compressor}@{compression_level}"
    )

    t_start = time.monotonic()
    try:
        out = consolidate_isoview(
            input_path,
            output_path,
            kind=kind,
            overwrite=overwrite,
            pyramid=pyramid,
            pyramid_max_layers=pyramid_max_layers,
            compressor=compressor,
            compression_level=compression_level,
            progress_callback=_ConsolidateProgress(monitor, logger),
        )
        _record_isoview_runtime(
            "consolidate",
            time.monotonic() - t_start,
            _isoview_config_root(input_path),
            logger,
        )
        monitor.finish(f"Wrote {out}")
        logger.info(f"isoview consolidator wrote {out}")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"isoview consolidator failed: {e}")
        raise


def _build_isoview_processing_config(args: dict):
    """Build an ``isoview.ProcessingConfig`` from the keys the widget set; the rest keep isoview's defaults."""
    from isoview import ProcessingConfig

    config_kwargs: dict = {"input_dir": Path(args["input_path"])}
    if args.get("output_dir"):
        config_kwargs["output_dir"] = Path(args["output_dir"])

    for key in (
        "output_suffix",
        "stitcher_suffix",
        "output_format",
        "compression",
        "compression_level",
        "zarr_sharded",
        "overwrite",
        "workers",
        "log",
        "pyramid",
        "pyramid_max_layers",
        "timepoints",
        "specimens",
        "tiled_override",
        "cameras",
        "segment_mode",
        "gauss_kernel",
        "gauss_sigma",
        "segment_threshold",
        "splitting",
        "apply_segmentation_mask",
        "subtract_background",
        "background_mode",
        "background_percentile",
        "mask_percentile",
        "subsample_factor",
        "do_tenengrad",
        "blending_method",
        "blending_range",
        "transition_plane",
        "flip_z",
        "flip_horizontal",
        "flip_vertical",
        "rotation",
        "front_flag",
        "blending_method_by_view",
        "blending_range_by_view",
        "transition_plane_by_view",
        "front_flag_by_view",
        "flip_z_by_view",
        "flip_horizontal_by_view",
        "flip_vertical_by_view",
        "rotation_by_view",
        "search_offsets_x_by_view",
        "search_offsets_y_by_view",
        "pixel_spacing_z",
        "detection_objective_mag",
        "pixel_spacing_camera",
        "crop_left",
        "crop_top",
        "crop_front",
        "crop_width",
        "crop_height",
        "crop_depth",
        "tile_crops",
    ):
        val = args.get(key)
        if val is not None:
            config_kwargs[key] = val

    if args.get("median_kernel") is not None:
        config_kwargs["median_kernel"] = tuple(args["median_kernel"])
    if args.get("search_offsets_x") is not None:
        config_kwargs["search_offsets_x"] = tuple(args["search_offsets_x"])
    if args.get("search_offsets_y") is not None:
        config_kwargs["search_offsets_y"] = tuple(args["search_offsets_y"])
    for key in ("zarr_chunks", "zarr_shards"):
        if args.get(key) is not None:
            config_kwargs[key] = tuple(int(v) for v in args[key])
    for key in ("search_offsets_x_by_view", "search_offsets_y_by_view"):
        d = args.get(key)
        if d is not None:
            config_kwargs[key] = {int(v): tuple(off) for v, off in d.items()}

    # json turned the camera-index keys into strings; get_crop(camera) needs ints
    for key in (
        "crop_left",
        "crop_top",
        "crop_front",
        "crop_width",
        "crop_height",
        "crop_depth",
    ):
        d = config_kwargs.get(key)
        if isinstance(d, dict):
            config_kwargs[key] = {int(k): int(v) for k, v in d.items()}

    tc = config_kwargs.get(
        "tile_crops"
    )  # {"SPM##": {crop_param: {camera: value}}}, camera keys are ints
    if isinstance(tc, dict):
        config_kwargs["tile_crops"] = {
            str(spm): {
                str(param): {int(cam): int(v) for cam, v in (cams or {}).items()}
                for param, cams in (params or {}).items()
            }
            for spm, params in tc.items()
        }

    return ProcessingConfig(**config_kwargs)


ISOVIEW_CONFIG_FILENAME = "isoview_config.json"


def _format_runtime(seconds: float) -> str:
    """``1h 3m 7s``."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _isoview_config_root(path) -> Path | None:
    """The dataset root (the parent of the ``.corrected`` or ``.fused`` tree ``path`` is in), else None."""
    if not path:
        return None
    p = Path(path)
    for anc in (p, *p.parents):
        name = anc.name
        for suf in (".corrected", ".fused"):
            idx = name.find(suf)
            if idx >= 0:
                rest = name[idx + len(suf) :]
                if rest == "" or rest.startswith("_"):
                    return anc.parent
    return None


def _record_isoview_runtime(
    step: str,
    elapsed: float,
    config_root,
    logger: logging.Logger,
) -> None:
    """Log the step's wall time and record it under ``runtimes`` in the dataset's ``isoview_config.json``; a None root only logs."""
    logger.info(f"Total runtime ({step}): {_format_runtime(elapsed)} ({elapsed:.1f}s)")
    if config_root is None:
        return
    cfg_path = Path(config_root) / ISOVIEW_CONFIG_FILENAME
    data: dict = {}
    if cfg_path.exists():
        with cfg_path.open() as f:
            data = json.load(f)
    data.setdefault("runtimes", {})[step] = round(float(elapsed), 1)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"isoview_config runtime saved: {cfg_path}")


class _ForwardingHandler(logging.Handler):
    """Mirrors records onto the worker's handlers at its own level.

    isoview's ``setup_logging()`` flips its logger to DEBUG partway through
    ``multi_fuse``, so this handler's level is the gate. ``handle`` rather
    than ``emit`` so each target's own level and filters apply.
    """

    def __init__(self, targets: list[logging.Handler], level=logging.INFO):
        super().__init__(level=level)
        self._targets = list(targets)

    def emit(self, record: logging.LogRecord) -> None:
        for h in self._targets:
            h.handle(record)


def _bridge_isoview_logging(worker_logger: logging.Logger) -> None:
    """Mirror the ``isoview`` logger's INFO records (DEBUG with debug logging on) onto the worker's handlers.

    isoview writes INFO to ``<fused_dir>/<method>/fusion.log``, which the
    Process Console never sees. One bridge per process; a second call
    refreshes its targets and level.
    """
    debug = log.debug_enabled() or get_debug_logging()
    level = logging.DEBUG if debug else logging.INFO

    iso_logger = logging.getLogger("isoview")
    iso_logger.setLevel(level)
    iso_logger.propagate = False

    # never the root mbo StreamHandler: the worker's stderr is already the log file
    targets: list[logging.Handler] = list(worker_logger.handlers)
    if not targets:
        targets = list(logging.getLogger("mbo").handlers)

    for handler in iso_logger.handlers:
        if isinstance(handler, _ForwardingHandler):
            handler._targets = targets
            handler.setLevel(level)
            return
    iso_logger.addHandler(_ForwardingHandler(targets, level=level))


def task_correct_stack(args: dict, logger: logging.Logger) -> None:
    """Run ``isoview.correct_stack`` on raw ``.stack`` data; it has no progress hook, so the bar only moves at the ends."""
    from isoview import correct_stack

    _bridge_isoview_logging(logger)

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing correct_stack...")

    t_start = time.monotonic()
    try:
        config = _build_isoview_processing_config(args)
        logger.info(
            f"correct_stack: input={config.input_dir}, "
            f"output={config.output_dir}, format={config.output_format}, "
            f"workers={config.workers}, segment_mode={config.segment_mode}"
        )
        logger.info(f"isoview per-run log: {config.output_dir}/correct.log")
        monitor.update(0.05, "Running correct_stack (see log file for details)...")
        correct_stack(config)
        _record_isoview_runtime(
            "correct_stack",
            time.monotonic() - t_start,
            config.input_dir.parent,
            logger,
        )
        monitor.finish("correct_stack pipeline complete.")
        logger.info("correct_stack completed successfully")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"correct_stack failed: {e}")
        raise


def task_isoview_raw_projections(args: dict, logger: logging.Logger) -> None:
    """Write the raw XY max projections into ``<raw_dir>.raw.projections/``, skipping files already there."""
    from mbo_utilities.arrays.isoview import make_raw_projections

    raw_dir = args.get("raw_dir")
    if not raw_dir:
        raise ValueError("raw_dir is required")
    overwrite = bool(args.get("overwrite", False))

    monitor = TaskMonitor(raw_dir, uuid=args.get("_uuid"))
    monitor.update(0.01, "Scanning raw stacks...")

    t_start = time.monotonic()
    try:
        result = make_raw_projections(
            raw_dir,
            overwrite=overwrite,
            progress_callback=functools.partial(_projection_progress, monitor),
        )
        _record_isoview_runtime(
            "raw_projections",
            time.monotonic() - t_start,
            None,
            logger,
        )
        monitor.finish(
            f"raw projections: {result['written']} written, "
            f"{result['skipped']} existing -> {result['dir']}"
        )
        logger.info(
            f"raw projections: {result['written']} written, "
            f"{result['skipped']} skipped in {result['dir']}"
        )
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"raw projections failed: {e}")
        raise


def _projection_progress(
    monitor: TaskMonitor, current: int, total: int, name: str
) -> None:
    frac = current / total if total else 1.0
    monitor.update(min(0.99, frac), f"projection {current}/{total}: {name}")


def _run_capturing(func, arg, result: dict) -> None:
    """Thread target: call ``func(arg)`` and park any exception under ``result["error"]``."""
    try:
        func(arg)
    except BaseException as exc:
        result["error"] = exc


def task_multi_fuse(args: dict, logger: logging.Logger) -> None:
    """Run ``isoview.multi_fuse`` on a corrected tree, counting fused timepoint dirs for progress."""
    from isoview import multi_fuse

    _bridge_isoview_logging(logger)

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing multi_fuse...")

    t_start = time.monotonic()
    try:
        config = _build_isoview_processing_config(args)
        fused_dir = Path(config.fused_dir)
        fusion_log_path = fused_dir / "fusion.log"
        logger.info(
            f"multi_fuse: input={config.input_dir}, "
            f"fused_dir={fused_dir}, blending={config.blending_method}, "
            f"workers={config.workers}"
        )
        logger.info(f"isoview per-run log: {fusion_log_path} (DEBUG records)")
        monitor.update(0.05, "Running multi_fuse (see log file for details)...")

        # multi_fuse has no progress callback and the watchdog kills a worker whose progress stalls,
        # so the count of fused timepoint dirs (SPM##/TM###### for a timelapse, SPM## tiled) is the progress
        total = max(len(config.timepoints), 1)
        result: dict = {}
        worker = threading.Thread(
            target=_run_capturing, args=(multi_fuse, config, result), daemon=True
        )
        worker.start()
        while worker.is_alive():
            worker.join(timeout=30)
            done = sum(1 for p in fused_dir.glob("SPM*/TM*") if p.is_dir()) or sum(
                1 for p in fused_dir.glob("SPM*") if p.is_dir()
            )
            frac = 0.05 + 0.9 * min(done / total, 1.0)
            monitor.update(frac, f"multi_fuse: {done}/{total} timepoints")
        if "error" in result:
            raise result["error"]

        _record_isoview_runtime(
            "fuse",
            time.monotonic() - t_start,
            config.input_dir.parent,
            logger,
        )
        monitor.finish("multi_fuse pipeline complete.")
        logger.info("multi_fuse completed successfully")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"multi_fuse failed: {e}")
        raise


def task_generate_bigstitcher(args: dict, logger: logging.Logger) -> None:
    """Write one BDV SpimData dataset per fusion method from the VW00/VW90 pairs under ``config.fused_dir``; no resampling."""
    from isoview import generate_bigstitcher_xml

    _bridge_isoview_logging(logger)

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing generate_bigstitcher_xml...")

    t_start = time.monotonic()
    try:
        config = _build_isoview_processing_config(args)
        logger.info(
            f"generate_bigstitcher_xml: input={config.input_dir}, "
            f"fused_dir={config.fused_dir}, "
            f"stitcher_dir={config.stitcher_dir}"
        )
        monitor.update(0.05, "Writing BDV zarr + dataset.xml...")
        xml_path = generate_bigstitcher_xml(
            config,
            method=args.get("method"),
            included_tiles=args.get("included_tiles"),
            bake_tile_positions=args.get("bake_tile_positions", True),
            orientation=args.get("orientation"),
            orientation_vw90=args.get("orientation_vw90"),
            camera_orientations=args.get("camera_orientations"),
            cameras=args.get("cameras"),
            source=args.get("source", "fused"),
            orient_to_cm00=args.get("orient_to_cm00", True),
            reverse_z=args.get("reverse_z", False),
            zarr_version=args.get("zarr_version", 2),
            link_existing=args.get("link_existing", False),
            tile_orientations=args.get("tile_orientations"),
        )
        logger.info(f"  wrote: {xml_path}")
        _record_isoview_runtime(
            "bigstitcher",
            time.monotonic() - t_start,
            config.input_dir.parent,
            logger,
        )
        monitor.finish(f"generate_bigstitcher_xml complete: {xml_path}")
        logger.info("generate_bigstitcher_xml completed successfully")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"generate_bigstitcher_xml failed: {e}")
        raise


def task_vnoiser(args: dict, logger: logging.Logger) -> None:
    """Run the vnoiser pipeline on the ticked units of a ``.mesc`` with the Run tab's settings and domain table.

    The viewer's Invert Deflection reaches the run as
    ``settings.traces.invert``, never as a second inversion of the movie.
    """
    from mbo_utilities.metadata import strip_for_export
    from mbo_utilities.vnoiser.params import VnoiserSettings
    from mbo_utilities.vnoiser.pipeline import run_vnoiser_pipeline

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Opening the source...")
    settings = VnoiserSettings.from_dict(args.get("settings"))
    units = list(args.get("units") or [])
    domains = {str(k): v for k, v in (args.get("domains") or {}).items()}
    frames = args.get("frames")
    try:
        src_arr = imread(args["input_path"], **(args.get("reader_kwargs") or {}))
        metadata = dict(src_arr.metadata or {})
        metadata.update(args.get("custom_metadata") or {})
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"vnoiser: cannot open input {args.get('input_path')!r}: {e}")
        raise
    logger.info(f"Input: {args['input_path']}  units: {units}")
    logger.info(f"Output: {args['output_dir']}")
    logger.info(f"Domains: {domains}")
    logger.info(
        f"Traces: {settings.traces.to_dict()}; polarity {settings.polarity}; "
        f"dF/F {settings.dfof.method}"
    )
    try:
        paths = run_vnoiser_pipeline(
            args["input_path"],
            domains=domains,
            units=units or None,
            first_env=args.get("first_env") or (),
            out=args["output_dir"],
            channel=int(args.get("channel") or 0),
            frames=None if frames is None else (int(frames[0]), int(frames[1])),
            planes=args.get("planes"),
            save_cwt=settings.runtime.save_cwt,
            settings=settings,
            detect=settings.events.detect,
            overwrite=settings.runtime.overwrite,
            provenance={
                "settings": settings.to_dict(),
                "source_metadata": strip_for_export(metadata),
            },
            progress_callback=monitor.update,
            logger=logger,
        )
        monitor.finish(
            f"vnoiser pipeline wrote {len(paths)} files to {args['output_dir']}"
        )
        logger.info(f"vnoiser completed: {sorted(paths)}")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"vnoiser failed: {e}")
        raise


TASKS = {
    "save_as": task_save_as,
    "suite2p": task_suite2p,
    "masknmf": task_masknmf,
    "vnoiser": task_vnoiser,
    "isoview": task_isoview,
    "isoview_correct": task_correct_stack,
    "isoview_raw_projections": task_isoview_raw_projections,
    "isoview_fuse": task_multi_fuse,
    "isoview_bigstitcher": task_generate_bigstitcher,
}

_PLUGIN_TASKS_LOADED = False


def register_task(name: str, func) -> None:
    """Register ``func(args, logger)`` as the worker for ``task_type`` ``name``; a second function under one name raises."""
    existing = TASKS.get(name)
    if existing is not None and existing is not func:
        raise ValueError(f"task {name!r} is already registered")
    TASKS[name] = func


def load_plugin_tasks() -> None:
    """Register every entry-point pipeline widget's ``task_type`` and ``task_func`` pair, once per process."""
    global _PLUGIN_TASKS_LOADED
    if _PLUGIN_TASKS_LOADED:
        return
    _PLUGIN_TASKS_LOADED = True

    from mbo_utilities.pipeline_registry import load_entry_point_pipelines

    for cls in load_entry_point_pipelines():
        name = getattr(cls, "task_type", None)
        func = getattr(cls, "task_func", None)
        if name and func is not None:
            register_task(name, func)
