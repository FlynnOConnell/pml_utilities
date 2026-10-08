"""masknmf pipeline runner: registration -> PMD compression -> demixing.

Each plane runs into a new masknmf run folder,
``<save_path>/<yyyymmddTHHMMSS>_masknmf_zplaneNN[_tpAAAAA-BBBBB]/``:
``results.hdf5`` with one group per stage, ``config.json``, the masknmf log,
``alignment.hdf5`` when registration ran on a denoised copy, the results zarr
(``mbo_utilities.results``) when demixing ran, and the QC figures. No movie
is written: the stages read the recording through ``imread`` and the
registered movie is the stored shifts replayed on it
(``arrays.masknmf_run.MasknmfRunArray``). Skip/Run/Force gates each stage on
the newest earlier run folder of the same plane: a stage whose stored
provenance matches is copied over instead of recomputed.

masknmf imports are function-local: the package is optional, heavy to import,
and its stage math runs in the spawned worker subprocess.
"""

import functools
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays.masknmf_run import RESULTS_FILE
from mbo_utilities.masknmf import outputs as _outputs
from mbo_utilities.masknmf.params import (
    ALIGN_FILE,
    PMD_FILE,
    MasknmfSettings,
    stage_action,
)

# the pipeline name a run's config.json records
PIPELINE = "mbo_utilities.masknmf"


def _no_progress(**_) -> None:
    pass


def _hash(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _stage_hash(cfg, tri_field: str) -> str:
    d = asdict(cfg)
    d.pop(tri_field, None)
    return _hash(d)


def _read_provenance(path: Path) -> dict | None:
    try:
        with h5py.File(path, "r") as f:
            return json.loads(f.attrs["mbo_provenance"])
    except Exception:
        return None


def _export_atomic(obj, path: Path, prov: dict | None = None) -> None:
    """Export to a temp name and rename, so a crash never leaves a partial
    file that later runs would trust as a cached stage output.
    """
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    obj.export(str(tmp))
    if prov is not None:
        with h5py.File(tmp, "a") as f:
            f.attrs["mbo_provenance"] = json.dumps(prov, sort_keys=True)
    os.replace(tmp, path)


def _matches(previous: Path | None, group: str, prov: dict) -> bool:
    """Whether the previous run's results file holds ``group`` made with ``prov``."""
    if previous is None:
        return False
    with h5py.File(previous / RESULTS_FILE, "r") as f:
        if group not in f or "mbo_provenance" not in f[group].attrs:
            return False
        return json.loads(f[group].attrs["mbo_provenance"]) == prov


def _copy_groups(previous: Path, results_path: Path, groups: list[str]) -> None:
    with (
        h5py.File(previous / RESULTS_FILE, "r") as src,
        h5py.File(results_path, "a") as dst,
    ):
        for group in groups:
            if group in src:
                src.copy(src[group], dst, group)


def _stamp(results_path: Path, group: str, prov: dict) -> None:
    with h5py.File(results_path, "a") as f:
        f[group].attrs["mbo_provenance"] = json.dumps(prov, sort_keys=True)


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _cuda_usable(device: str = "cuda") -> bool:
    """A torch build without this GPU's architecture reports cuda available
    but fails at the first kernel launch; probe with a real kernel.
    """
    import torch

    try:
        (torch.ones(2, device=device) * 2).sum().item()
        return True
    except Exception:
        return False


def _resolve_device(device: str, logger=None) -> str:
    import torch

    if device and device != "auto":
        if device.startswith("cuda") and not _cuda_usable(device):
            raise RuntimeError(
                f"device '{device}' cannot run torch kernels: this torch build "
                "has no kernels for the installed GPU. Reinstall torch for this "
                "GPU architecture or set device='cpu'."
            )
        return device
    if torch.cuda.is_available():
        if _cuda_usable():
            return "cuda"
        if logger is not None:
            logger.warning(
                "masknmf: CUDA available but this torch build has no kernels "
                "for the installed GPU; falling back to cpu (much slower). "
                "Install a torch build for this GPU, e.g.: uv pip install "
                "torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu126 "
                "(pre-Turing GPUs need cu126 or cu118, not cu13x)"
            )
    return "cpu"


def generate_plane_dirname(plane: int, frame_indices: list[int] | None = None) -> str:
    """lsp-compatible plane dir name: zplaneNN[_tpAAAAA-BBBBB] (1-based)."""
    name = f"zplane{plane:02d}"
    if frame_indices:
        name += f"_tp{frame_indices[0] + 1:05d}-{frame_indices[-1] + 1:05d}"
    return name


def _register(raw, cfg, runtime, device: str, logger):
    """``(registration array over raw, the movie its shifts were estimated on)``."""
    import masknmf

    corrector_cls = (
        masknmf.PiecewiseRigidMotionCorrector
        if cfg.strategy == "pwrigid"
        else masknmf.RigidMotionCorrector
    )
    strategy = corrector_cls(
        **cfg.strategy_kwargs(),
        device=device,
        batch_size=runtime.frame_batch_size,
    )
    reference = raw
    if cfg.denoised_reference:
        logger.info("masknmf: denoising a copy of the movie to register on")
        reference = masknmf.CompressStrategy(
            **cfg.reference_kwargs(),
            frame_batch_size=runtime.frame_batch_size,
            device=device,
        ).compress(raw)
    logger.info(f"masknmf: computing {cfg.strategy} registration template")
    strategy.compute_template(reference)
    logger.info("masknmf: estimating shifts")
    moco = strategy.motion_correct(reference, target_movie=raw)
    moco.output_device = moco.strategy.device
    return moco, reference


def _shift_mask(shifts, shape: tuple[int, int], border: int) -> np.ndarray:
    mask = np.ones(shape, dtype="float")
    if shifts is not None:
        try:
            from masknmf.motion_correction.moco_preprocessing import (
                construct_moco_template,
            )

            mask = construct_moco_template(shifts, shape).astype("float")
        except Exception:
            pass
    if border > 0:
        mask[:border, :] = 0
        mask[:, :border] = 0
        mask[-border:, :] = 0
        mask[:, -border:] = 0
    return mask


def _compress(moco, cfg, device: str, mask, fs, logger):
    """The PMD ``CompressionArray`` of ``moco``."""
    import masknmf

    kw = cfg.strategy_kwargs()
    kw["pixel_weighting"] = mask
    strat_cls = (
        masknmf.CompressDenoiseStrategy if cfg.denoise else masknmf.CompressStrategy
    )
    strat = strat_cls(device=device, **kw)
    if cfg.detrend:
        detrender = _spline_detrender(
            int(moco.shape[0]), fs, 40, 25, device, logger, "compression"
        )
        if detrender is not None:
            strat.detrender = detrender
    logger.info("masknmf: running PMD compression")
    return strat.compress(moco)


def _stage_compression(
    moco,
    cfg,
    runtime,
    plane_dir: Path,
    device: str,
    mask,
    fs,
    logger,
    upstream_key,
    upstream_computed,
):
    """``compression.hdf5`` in ``plane_dir``, cached by provenance; returns (CompressionArray, seconds, prov_key)."""
    pmd_path = plane_dir / PMD_FILE
    cached = pmd_path.exists()
    action = stage_action(cfg.do_compression, cached)
    prov = {
        "settings": _stage_hash(cfg, "do_compression"),
        "input": upstream_key,
        "fs": fs,
    }
    key = _hash(prov)
    if action == "skip" and not cached:
        logger.info(
            "masknmf: compression skipped (no cached PMD; demixing will be skipped)"
        )
        return None, 0.0, key

    import masknmf

    if action == "skip":
        logger.info(f"masknmf: reusing {PMD_FILE} (compression skipped)")
        return masknmf.CompressionArray.from_hdf5(str(pmd_path)), 0.0, key

    if action == "reuse" and upstream_computed:
        logger.info("masknmf: registration recomputed; recomputing compression")
        action = "compute"
    if action == "reuse":
        stored = _read_provenance(pmd_path)
        if stored is not None and stored != prov:
            logger.info(
                f"masknmf: cached {PMD_FILE} stale (settings or input changed); recomputing"
            )
            action = "compute"
    if action == "reuse":
        try:
            pmd = masknmf.CompressionArray.from_hdf5(str(pmd_path))
            logger.info(f"masknmf: reusing {PMD_FILE}")
            return pmd, 0.0, key
        except Exception as e:
            logger.warning(f"masknmf: cached {PMD_FILE} unusable ({e}); recomputing")

    t0 = time.time()
    pmd = _compress(moco, cfg, device, mask, fs, logger)
    _export_atomic(pmd, pmd_path, prov)
    # reload so demixing always consumes the exact persisted decomposition
    return masknmf.CompressionArray.from_hdf5(str(pmd_path)), time.time() - t0, key


def _spline_detrender(
    nframes: int,
    fs,
    window_seconds: float,
    knot_seconds: float,
    device: str,
    logger=None,
    stage: str = "",
):
    """A ``MaximinSplineDetrend`` for this movie, or None when it is too short.

    The detrender reflect-pads by half its window, and torch refuses a pad
    wider than the axis, so a movie shorter than the window crashes rather
    than degrading. High frame rates make that easy to hit — 40 s at 430 Hz is
    17k frames — so short recordings just skip detrending.
    """
    if not fs:
        return None
    window = int(window_seconds * fs)
    if window < 2 or nframes < 2 * window:
        if logger is not None:
            logger.info(
                f"masknmf: {nframes} frames < 2x the {stage} detrend window "
                f"({window} frames at fs={fs}); detrending disabled"
            )
        return None

    from masknmf.compression.preprocessing import MaximinSplineDetrend

    return MaximinSplineDetrend(
        num_frames=nframes,
        num_knots=max(4, int(nframes / fs / knot_seconds)),
        window=window,
        sigma=max(2.0, 0.3 * fs),
        device=device,
    )


def clamp_background_downsampling(cfg, ly: int, lx: int, logger=None):
    """A copy of ``cfg`` whose background downsampling fits an ``(ly, lx)`` field.

    masknmf's ring/background model average-pools the field by this factor and
    squeezes the result. Pooling is floor division, so a factor at or above a
    side length collapses that axis to one pixel, the squeeze drops it, and
    ``lowrank_background_svd`` raises ``IndexError: tuple index out of range``
    on the reshape that follows. The default factor is 30, which any field
    narrower than 30 px hits — an LBM strip, a drawn crop, a test movie. Keep
    at least four pooled pixels on the short side.
    """
    import copy

    want = max(
        1, min(int(cfg.background_downsampling_factor), min(int(ly), int(lx)) // 4)
    )
    if want == cfg.background_downsampling_factor:
        return cfg
    out = copy.copy(cfg)
    out.background_downsampling_factor = want
    if logger is not None:
        logger.info(
            f"masknmf: background downsampling {cfg.background_downsampling_factor} -> {want} "
            f"for a {ly}x{lx} field"
        )
    return out


def _demix(pmd, cfg, runtime, device: str, fs, logger):
    """``DemixingResults`` of ``pmd``: filtered passes seed the unfiltered ones."""
    import torch
    from masknmf.demixing import NoSignalsDetectedError

    import masknmf

    detrender = _spline_detrender(
        int(pmd.shape[0]), fs, 20, 20, device, logger, "demixing"
    )
    logger.info(f"masknmf: spatial highpass (sigma={cfg.filter_sigma})")
    filtered = masknmf.demixing.filters.spatial_filter_compressed_array(
        pmd,
        batch_size=runtime.frame_batch_size,
        filter_sigma=cfg.filter_sigma,
        target_device=device,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    demixer = masknmf.SignalDemixer(
        filtered, device=device, frame_batch_size=runtime.frame_batch_size
    )
    filtered_results = None
    for i in range(max(1, cfg.filtered_passes)):
        try:
            demixer.initialize_signals(**cfg.init_kwargs(detrender))
        except NoSignalsDetectedError:
            break
        logger.info(f"masknmf: demixing filtered pass {i + 1}")
        demixer.demix(
            **cfg.nmf_kwargs(cfg.support_threshold_lo, ring=False, detrender=detrender)
        )
        filtered_results = demixer.results
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if filtered_results is None:
        raise ValueError(
            "masknmf found no signals in the highpass-filtered movie; "
            "lower mad_correlation_threshold or inspect the data"
        )

    a_init = filtered_results.signals_array.export_spatial_demixed()
    c_init = filtered_results.signals_array.export_temporal_demixed()

    demixer = masknmf.SignalDemixer(
        pmd, device=device, frame_batch_size=runtime.frame_batch_size
    )
    results = None
    for i in range(max(1, cfg.unfiltered_passes)):
        try:
            if i == 0:
                demixer.initialize_signals(
                    is_custom=True,
                    spatial_footprints=a_init,
                    temporal_footprints=c_init,
                    c_nonneg=True,
                )
            else:
                demixer.initialize_signals(**cfg.init_kwargs(detrender))
        except NoSignalsDetectedError:
            break
        logger.info(f"masknmf: demixing unfiltered pass {i + 1}")
        demixer.demix(
            **cfg.nmf_kwargs(cfg.unfiltered_support_lo, ring=True, detrender=detrender)
        )
        results = demixer.results
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if results is None:
        raise ValueError("masknmf unfiltered demixing did not complete a pass")
    return results


def _movie_stats(moco) -> tuple[np.ndarray, np.ndarray]:
    """meanImg/max_proj from up to 500 evenly-sampled frames."""
    nframes = int(moco.shape[0])
    idx = np.unique(np.linspace(0, nframes - 1, min(nframes, 500)).astype(int))
    frames = _to_np(moco[idx.tolist()]).astype(np.float32)
    return frames.mean(axis=0), frames.max(axis=0)


def _extract_footprints(results):
    """(indices (2,nnz), values, baseline) numpy from DemixingResults."""
    coo = results.spatial_demixed.coalesce()
    return (
        _to_np(coo.indices()),
        _to_np(coo.values()),
        _to_np(results.static_baseline),
    )


def _results_unit(results, pmd, plane: int, run: Path, fs, images: dict):
    """``(unit, ROIs with a positive F0)``: the demixing as one pixel unit of a
    results file, with calibrated ``raw`` and ``dff`` traces.
    """
    from mbo_utilities.results import ResultUnit, unit_name

    indices, values, baseline = _extract_footprints(results)
    c = np.asarray(results.signals_array.export_temporal_demixed(), dtype=np.float32)
    n_rois = int(c.shape[1])
    footprints = _outputs.split_sparse_footprints(indices, values, n_rois)
    # the PMD standardisation images calibrate c back to movie units
    gain, f0 = _outputs.roi_calibration(
        footprints,
        var_img=_to_np(pmd.noise_variance_image),
        mean_img=_to_np(pmd.mean_image),
        baseline=baseline,
    )
    F, dff = _outputs.calibrated_traces(c, gain, f0)
    return ResultUnit(
        name=unit_name("plane", plane),
        kind="plane",
        index=plane,
        fs=fs,
        roi_names=[str(k) for k in range(n_rois)],
        traces={"raw": F, "dff": dff},
        member_kind="pixel",
        members=[pix.astype(np.int64) for pix, _ in footprints],
        weights=[lam.astype(np.float32) for _, lam in footprints],
        image_shape=tuple(int(s) for s in results.fov_shape),
        iscell=np.ones((n_rois, 2), dtype=np.float32),
        images=images,
        attrs={"plane_dir": str(run), "z": plane - 1},
    ), int((f0 > 0).sum())


def run_plane(
    input_data,
    save_path,
    plane: int = 1,
    settings: MasknmfSettings | dict | None = None,
    metadata: dict | None = None,
    frame_indices: list[int] | None = None,
    channel: int | None = None,
    replot: bool = True,
    writer_kwargs: dict | None = None,
    logger=None,
    progress_callback=None,
) -> Path:
    """Run the masknmf pipeline on one z-plane (1-based) into a new run folder; returns it.

    ``writer_kwargs`` are the read features applied to the recording
    (``fix_phase``, ``use_fft``, ``frame_average``, ``invert_deflection``);
    ``channel`` is 1-based; ``progress_callback(step=, message=)``.
    """
    import masknmf
    from mbo_utilities import imread
    from mbo_utilities.arrays._inverted_view import InvertedDeflectionView
    from mbo_utilities.arrays.features import apply_read_features
    from mbo_utilities.arrays.features._slicing import index_window
    from mbo_utilities.metadata import get_param
    from mbo_utilities.reader import source_reader_kwargs
    from mbo_utilities.results import Results, results_name
    from mbo_utilities.roi_workflow import PlaneMovie

    logger = logger or log.get()
    if not isinstance(settings, MasknmfSettings):
        settings = MasknmfSettings.from_dict(settings)
    reg, comp, demix, runtime = (
        settings.registration,
        settings.compression,
        settings.demixing,
        settings.runtime,
    )
    save_path = Path(save_path)
    device = _resolve_device(runtime.device, logger)
    progress = progress_callback or _no_progress

    arr = input_data if hasattr(input_data, "shape") else imread(input_data)
    source = getattr(arr, "source_path", None)
    reader_kwargs = source_reader_kwargs(arr)
    read_features = {k: v for k, v in (writer_kwargs or {}).items() if v is not None}
    arr, _ = apply_read_features(arr, read_features)
    nz = int(arr._shape5d()[2]) if hasattr(arr, "_shape5d") else 1
    z = plane - 1 if nz > 1 else 0
    c = int(channel) - 1 if channel is not None else 0
    inverted = isinstance(arr, InvertedDeflectionView)
    # 2 * mean - x holds a still mean against a moving frame: register first, invert after
    raw = PlaneMovie(arr.source if inverted else arr, z=z, c=c).select(frame_indices)
    window = index_window(frame_indices)
    fs = get_param(dict(metadata or {}), "fs") or get_param(
        dict(getattr(arr, "metadata", None) or {}), "fs"
    )
    fs = float(fs) / (window[2] if window else 1) if fs else None
    movie = {
        "path": None if source is None else str(source),
        "name": None if source is None else Path(source).name,
        "reader_kwargs": reader_kwargs,
        "read_features": read_features,
        "plane": plane,
        "z": z,
        "c": c,
        "frames": None if window is None else list(window),
        "tp_indices": None
        if frame_indices is None or window is not None
        else [int(t) for t in frame_indices],
        "fs": fs,
    }
    if inverted:
        movie["registered_before_inversion"] = True

    name = f"masknmf_{generate_plane_dirname(plane, frame_indices)}"
    earlier = sorted(
        p for p in save_path.glob(f"*_{name}") if (p / RESULTS_FILE).is_file()
    )
    previous = earlier[-1] if earlier else None
    run = masknmf.io.create_run_folder(save_path, name)
    handler = masknmf.io.log_to(run)
    results_path = run / RESULTS_FILE
    logger.info(f"masknmf: plane {plane} -> {run}")
    timings: dict = {}
    record = {"start": datetime.now().isoformat(timespec="seconds"), "device": device}
    try:
        progress(step="registration", message=f"Registering plane {plane}")
        t0 = time.time()
        cls = (
            masknmf.PiecewiseRigidRegistrationArray
            if reg.strategy == "pwrigid"
            else masknmf.RigidRegistrationArray
        )
        reg_prov = {
            "settings": _stage_hash(reg, "do_registration"),
            "input": _hash(movie),
        }
        reg_key = _hash(reg_prov)
        action = stage_action(
            reg.do_registration, _matches(previous, cls.__name__, reg_prov)
        )
        shifts = template = None
        if action == "skip":
            logger.info("masknmf: registration skipped")
            moco = raw
        else:
            if action == "reuse":
                logger.info(f"masknmf: reusing the registration of {previous.name}")
                _copy_groups(
                    previous, results_path, [cls.__name__, cls._strategy_cls.__name__]
                )
                if (previous / ALIGN_FILE).is_file():
                    shutil.copyfile(previous / ALIGN_FILE, run / ALIGN_FILE)
                moco = cls.from_hdf5(results_path, input_movie=raw, device=device)
            else:
                moco, reference = _register(raw, reg, runtime, device, logger)
                moco.export(results_path)
                _stamp(results_path, cls.__name__, reg_prov)
                if reference is not raw:
                    reference.export(run / ALIGN_FILE)
            shifts = _to_np(moco.shifts)
            template = getattr(moco.strategy, "template", None)
            template = None if template is None else _to_np(template)
        timings["registration"] = {
            "seconds": round(time.time() - t0, 2),
            "action": action,
        }
        if inverted:
            moco = masknmf.OphysArray(
                moco,
                negative_indicator=True,
                include_mean=True,
                device=device,
                batch_size=runtime.frame_batch_size,
            )

        progress(step="compression", message=f"Compressing plane {plane}")
        t0 = time.time()
        comp_prov = {
            "settings": _stage_hash(comp, "do_compression"),
            "input": reg_key,
            "fs": fs,
        }
        comp_key = _hash(comp_prov)
        group = masknmf.io.group_name_compression()
        cached = _matches(previous, group, comp_prov)
        action = stage_action(comp.do_compression, cached)
        if action == "skip" and not cached:
            logger.info("masknmf: compression skipped (demixing will be skipped)")
            pmd = None
        else:
            if action == "compute":
                mask = _shift_mask(shifts, raw.shape[1:], runtime.exclude_border_radius)
                _compress(moco, comp, device, mask, fs, logger).export(results_path)
                _stamp(results_path, group, comp_prov)
            else:
                logger.info(f"masknmf: reusing the compression of {previous.name}")
                _copy_groups(previous, results_path, [group])
            # demixing consumes the decomposition as the file holds it
            pmd = masknmf.CompressionArray.from_hdf5(results_path)
        timings["compression"] = {
            "seconds": round(time.time() - t0, 2),
            "action": action,
        }

        progress(step="demixing", message=f"Demixing plane {plane}")
        t0 = time.time()
        results = None
        if pmd is None:
            action = "skip"
        else:
            demix = clamp_background_downsampling(
                demix, pmd.shape[1], pmd.shape[2], logger
            )
            demix_prov = {
                "settings": _stage_hash(demix, "do_demixing"),
                "input": comp_key,
                "fs": fs,
            }
            group = masknmf.io.group_name_demixing()
            action = stage_action(
                demix.do_demixing, _matches(previous, group, demix_prov)
            )
            if action == "compute":
                _demix(pmd, demix, runtime, device, fs, logger).export(results_path)
                _stamp(results_path, group, demix_prov)
            elif action == "reuse":
                logger.info(f"masknmf: reusing the demixing of {previous.name}")
                _copy_groups(previous, results_path, [group])
            if action != "skip":
                results = masknmf.DemixingResults.from_hdf5(results_path, device=device)
        timings["demixing"] = {"seconds": round(time.time() - t0, 2), "action": action}

        progress(step="exports", message=f"Writing outputs for plane {plane}")
        if results is not None:
            mean_img, max_proj = _movie_stats(moco)
            images = {"mean": mean_img, "max": max_proj}
            ci = np.asarray(results.standard_correlation_images[:])
            images["corr"] = ci.max(axis=0) if ci.ndim == 3 else ci
            if template is not None:
                images["ref"] = template
            unit, n_calibrated = _results_unit(results, pmd, plane, run, fs, images)
            Results(
                pipeline="masknmf",
                units={unit.name: unit},
                source={
                    "path": movie["path"],
                    "planes": [plane],
                    "reader_kwargs": reader_kwargs,
                },
                settings=settings.to_dict(),
                metadata={"fs": fs} if fs else {},
            ).write(run / results_name(source or run.name, pipeline="masknmf"))
            logger.info(f"masknmf: plane {plane} -> {unit.n_rois} ROIs")
            if n_calibrated < unit.n_rois:
                logger.warning(
                    f"masknmf: plane {plane} - {unit.n_rois - n_calibrated}/{unit.n_rois} "
                    "ROIs have no positive F0; their dF/F is written as zeros"
                )
        if replot:
            progress(step="figures", message=f"Plotting QC for plane {plane}")
            from mbo_utilities.masknmf import qc

            qc.plot_plane_figures(
                run,
                shifts=shifts,
                pmd=pmd,
                results=results,
                fs=fs,
                vcorr=images["corr"] if results is not None else None,
                logger=logger,
            )
        record["status"] = "completed"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["end"] = datetime.now().isoformat(timespec="seconds")
        masknmf.io.write_run_config(
            run,
            PIPELINE,
            settings.to_dict(),
            run=record,
            inputs={"movie": movie},
            timings=timings,
        )
        logging.getLogger("masknmf").removeHandler(handler)
        handler.close()
    return run


def run_volume(
    input_data,
    save_path,
    planes: list[int] | None = None,
    settings: MasknmfSettings | dict | None = None,
    metadata: dict | None = None,
    frame_indices: list[int] | None = None,
    channel: int | None = None,
    replot: bool = True,
    writer_kwargs: dict | None = None,
    logger=None,
    progress_callback=None,
) -> list[Path]:
    """Run the masknmf pipeline over selected 1-based z-planes; returns their run folders.

    ``progress_callback(plane=, total_planes=, step=, message=)``.
    """
    logger = logger or log.get()
    from mbo_utilities import imread

    arr = input_data if hasattr(input_data, "shape") else imread(input_data)
    if planes is None:
        nz = int(arr._shape5d()[2]) if hasattr(arr, "_shape5d") else 1
        planes = list(range(1, nz + 1))
    return [
        run_plane(
            arr,
            save_path,
            plane=plane,
            settings=settings,
            metadata=metadata,
            frame_indices=frame_indices,
            channel=channel,
            replot=replot,
            writer_kwargs=writer_kwargs,
            logger=logger,
            progress_callback=None
            if progress_callback is None
            else functools.partial(
                progress_callback, plane=i, total_planes=len(planes)
            ),
        )
        for i, plane in enumerate(planes)
    ]
