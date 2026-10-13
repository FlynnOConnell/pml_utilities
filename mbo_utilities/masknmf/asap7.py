"""ASAP7 spine run: the batch form of asap7_spine_pipeline_detrend.ipynb.

1. compress the inverted raw movie (ASAP7 dims on depolarisation)
2. register the raw movie with shifts estimated on that compression
3. compress and denoise the registered movie with a maximin spline detrender
4. demix it with masknmf's glutamate spine pipeline (spines and the whole dendrite)

Steps 1-2 run here; the pipeline gets the registered movie with its own
registration off and runs 3-4 into its run folder, which also takes the
registration (``registration.hdf5``) and this run's settings and provenance
(``asap7.json``).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.masknmf.params import (
    ASAP7_FOLDER,
    ASAP7_REGISTRATION_FILE,
    ASAP7_RESULTS_FILE,
    ASAP7_SETTINGS_FILE,
    Asap7Settings,
)
from mbo_utilities.masknmf.reg_denoise import dense

logger = log.get("masknmf.asap7")


def default_save_path(source: Path, unit_key: str | None) -> Path:
    """``<folder>/masknmf_asap7/<unit>_detrend`` beside the source."""
    tag = unit_key.replace("/", "_") if unit_key else source.stem
    return source.parent / ASAP7_FOLDER / f"{tag}_detrend"


def asap7_detrender(n_frames: int, fs: float, window_seconds: float, device: str):
    """The notebook's ``MaximinSplineDetrend``: one knot per window, smoothing a fifth of it."""
    from masknmf.compression.preprocessing import MaximinSplineDetrend

    window = int(window_seconds * fs)
    if window < 2 or n_frames < 2 * window:
        raise ValueError(
            f"{n_frames} frames is under twice the {window_seconds} s detrend window "
            f"({window} frames at {fs:.1f} Hz)"
        )
    return MaximinSplineDetrend(
        num_frames=n_frames,
        num_knots=max(4, int(n_frames / fs / window_seconds)),
        window=window,
        sigma=max(2.0, 0.2 * window_seconds * fs),
        device=device,
    )


def asap7_demixing_config(settings: Asap7Settings):
    """The notebook's spine demixing: identical passes of superpixel init then NMF."""
    import masknmf

    passes = []
    for _ in range(settings.demixing_passes):
        init_config = masknmf.SuperpixelInitConfig(
            mad_correlation_threshold=settings.mad_correlation_threshold,
            residual_threshold=0.1,
            sign="positive",
        )
        nmf_config = masknmf.NMFConfig(
            support_threshold=(0.95, 0.7),
            ring_model_start_pt=41,
            min_brightness=0.0,
            merge_threshold=0.7,
            reassign_background=False,
        )
        passes.append(masknmf.SinglepassDemixingConfig(init_config, nmf_config))
    return masknmf.MultipassDemixingConfig(passes)


def _source_movie(arr, settings: Asap7Settings) -> np.ndarray:
    if arr.nz != 1:
        raise ValueError(
            f"the ASAP7 run is single-plane; {arr.source_path} has {arr.nz} z-planes"
        )
    c = settings.channel - 1
    if not 0 <= c < arr.nc:
        raise ValueError(f"channel {settings.channel} is not in 1..{arr.nc}")
    if arr.shape[0] <= settings.first_frame:
        raise ValueError(
            f"{arr.shape[0]} frames, none left after starting at frame {settings.first_frame}"
        )
    return np.asarray(arr[settings.first_frame :, c, 0], dtype=np.float32)


def run_asap7(
    source,
    unit: str | int | None = None,
    save_path: Path | str | None = None,
    settings: Asap7Settings | None = None,
) -> Path:
    """Run the four steps on one channel of ``source``; returns the pipeline's run folder.

    ``source`` is a path (``unit`` selects a ``.mesc`` unit) or a ``LazyArray``.
    ``save_path`` defaults to ``masknmf_asap7/<unit>_detrend/`` beside the source;
    each run makes a new timestamped run folder inside it.
    """
    import masknmf
    from mbo_utilities.reader import imread, source_reader_kwargs

    settings = settings or Asap7Settings()
    arr = imread(source, unit=unit) if unit is not None else imread(source)
    src = Path(arr.source_path)
    reader_kwargs = source_reader_kwargs(arr)
    fs = arr.fs
    if not fs:
        raise ValueError(f"{src.name} {reader_kwargs} declares no frame rate")
    save_path = (
        Path(save_path)
        if save_path
        else default_save_path(src, reader_kwargs.get("unit"))
    )
    device = settings.runtime.device
    batch = settings.runtime.frame_batch_size
    timings = {}

    t0 = time.time()
    raw = _source_movie(arr, settings)
    detrender = (
        asap7_detrender(raw.shape[0], fs, settings.detrend_window_seconds, device)
        if settings.compression.detrend
        else None
    )
    movie = masknmf.OphysArray(
        raw, negative_indicator=True, include_mean=True, device=device
    )
    timings["read"] = time.time() - t0
    logger.info(
        f"asap7: {src.name} {reader_kwargs} channel {settings.channel} frames "
        f"{settings.first_frame}: -> {raw.shape} at {fs:.1f} Hz, inverted, in {timings['read']:.1f}s"
    )

    t0 = time.time()
    reference = masknmf.CompressStrategy(
        device=device, frame_batch_size=batch, **settings.reference.strategy_kwargs()
    ).compress(movie)
    reference.to(device)
    timings["reference compression"] = time.time() - t0
    logger.info(
        f"asap7: compressed the raw movie in {timings['reference compression']:.1f}s"
    )

    t0 = time.time()
    corrector = masknmf.RigidMotionCorrector(
        device=device, batch_size=batch, **settings.registration.strategy_kwargs()
    )
    corrector.compute_template(reference)
    registration = corrector.motion_correct(
        reference_movie=reference, target_movie=movie
    )
    registered = dense(registration, batch)
    timings["registration"] = time.time() - t0
    logger.info(
        f"asap7: registered the raw movie to its compression in {timings['registration']:.1f}s"
    )

    comp = settings.compression
    config_cls = (
        masknmf.CompressDenoiseConfig if comp.denoise else masknmf.CompressConfig
    )
    pipeline = masknmf.GlutamateCalciumSpinePipeline(
        output_folder=save_path,
        motion_correct_config="skip",
        compress_config=config_cls(**comp.strategy_kwargs(), detrender=detrender),
        demixing_config=asap7_demixing_config(settings),
        frame_batch_size=batch,
        device=device,
    )
    t0 = time.time()
    run = Path(
        pipeline.run(
            glutamate_channel=registered, calcium_channel=None, exclude_initial_frames=0
        )
    )
    timings["compression and demixing"] = time.time() - t0

    registration.export(str(run / ASAP7_REGISTRATION_FILE))
    shifts = registration.shifts.cpu().numpy()
    provenance = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": {"path": str(src), "reader_kwargs": reader_kwargs},
        "n_frames": int(raw.shape[0]),
        "fs": fs,
        "shift_range_px": {
            "y": [float(shifts[:, 0].min()), float(shifts[:, 0].max())],
            "x": [float(shifts[:, 1].min()), float(shifts[:, 1].max())],
        },
        "settings": settings.to_dict(),
        "timing": timings,
        "masknmf_version": getattr(masknmf, "__version__", None),
    }
    (run / ASAP7_SETTINGS_FILE).write_text(
        json.dumps(provenance, indent=2, default=str)
    )
    logger.info(f"asap7: wrote {run}")
    return run


def is_asap7_run(path: Path | str) -> bool:
    """Whether ``path`` is a run folder ``run_asap7`` wrote: ``asap7.json`` beside the spine results."""
    p = Path(path)
    return (p / ASAP7_SETTINGS_FILE).is_file() and (p / ASAP7_RESULTS_FILE).is_file()


def run_provenance(run: Path | str) -> dict:
    """A run folder's ``asap7.json``."""
    return json.loads((Path(run) / ASAP7_SETTINGS_FILE).read_text(encoding="utf-8"))


def run_registration(run: Path | str) -> Path | None:
    """The run's ``registration.hdf5``, the shifts estimated on the compressed raw movie."""
    p = Path(run) / ASAP7_REGISTRATION_FILE
    return p if p.is_file() else None


def run_frame_times(run: Path | str) -> np.ndarray:
    """Each frame of the run in seconds from the start of the recording, the time axis RTMC is on."""
    prov = run_provenance(run)
    first = prov["settings"]["first_frame"]
    return (first + np.arange(prov["n_frames"])) / prov["fs"]


def run_raw_movie(run: Path | str) -> np.ndarray:
    """The inverted raw movie the run registered, re-read from its source: ``(T, Y, X)`` float32."""
    from mbo_utilities.reader import imread

    prov = run_provenance(run)
    settings = Asap7Settings.from_dict(prov["settings"])
    arr = imread(prov["source"]["path"], **prov["source"]["reader_kwargs"])
    raw = _source_movie(arr, settings)
    return 2 * raw.mean(axis=0) - raw


def run_rtmc(run: Path | str):
    """The RTMC the scan went through (a ``MotionCorrection`` in µm), None when the source has none."""
    from mbo_utilities.arrays.mesc import rtmc_motion, unit_rtmc

    source = run_provenance(run)["source"]
    unit = source["reader_kwargs"].get("unit")
    if unit is None:
        return None
    return rtmc_motion(unit_rtmc(source["path"], unit))
