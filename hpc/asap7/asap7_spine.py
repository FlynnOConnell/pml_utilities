#!/usr/bin/env python
"""
ASAP7 spine pipeline on one unit of a Femtonics .mesc file, headless: the batch form of
asap7_spine_pipeline_detrend.ipynb.

The movie is inverted (ASAP7 dims on depolarisation), registered with shifts estimated on a PMD-denoised copy,
detrended with a maximin spline and denoised, then handed to GlutamateCalciumSpinePipeline as the glutamate channel.

    python asap7_spine.py --path_mesc scan.mesc --unit MSession_0/MUnit_14 --dir_save out/MUnit_14_detrend

dir_save gets asap7.json (these arguments), logs/run.log, and the pipeline's <timestamp>_glutamate-calcium-spine
run folder.
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

import masknmf
import mbo_utilities as mbo
import torch
from masknmf.compression.preprocessing import MaximinSplineDetrend
from mbo_utilities.arrays.mesc import list_mesc_units


@dataclasses.dataclass
class Asap7Config:
    path_mesc: str
    unit: str
    dir_save: str
    channel: int = 0
    exclude_initial_frames: int = 200
    mad_correlation_threshold: float = 0.6
    detrend_window_s: float = 0.3
    device: str = "cuda"


def parse_args() -> Asap7Config:
    defaults = Asap7Config(path_mesc="", unit="", dir_save="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path_mesc", required=True, help="the .mesc file")
    parser.add_argument("--unit", required=True, help="the unit's full key, e.g. MSession_0/MUnit_14")
    parser.add_argument("--dir_save", required=True, help="run folder; made when missing")
    parser.add_argument("--channel", type=int, default=defaults.channel)
    parser.add_argument("--exclude_initial_frames", type=int, default=defaults.exclude_initial_frames)
    parser.add_argument("--mad_correlation_threshold", type=float, default=defaults.mad_correlation_threshold)
    parser.add_argument("--detrend_window_s", type=float, default=defaults.detrend_window_s, help="maximin spline window, seconds")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=defaults.device)
    return Asap7Config(**vars(parser.parse_args()))


def setup_dir_save(config: Asap7Config) -> Path:
    dir_save = Path(config.dir_save)
    (dir_save / "logs").mkdir(parents=True, exist_ok=True)
    return dir_save


def setup_logging(dir_save: Path) -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.FileHandler(dir_save / "logs" / "run.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return logging.getLogger("asap7_spine")


def save_config(dir_save: Path, config: Asap7Config) -> None:
    filepath_config = dir_save / "asap7.json"
    filepath_tmp = filepath_config.with_suffix(".json.tmp")
    filepath_tmp.write_text(json.dumps(dataclasses.asdict(config), indent=2))
    os.replace(filepath_tmp, filepath_config)


def check_inputs(config: Asap7Config) -> dict:
    """
    The unit's description from the file's headers, once everything the run needs is known to be there.

    Raises
    ------
    ValueError
        If the unit, channel or frame rate is missing, or too few frames are left after the excluded ones.
    """
    if not Path(config.path_mesc).is_file():
        raise ValueError(f"{config.path_mesc} is not a file")
    units = {u["key"]: u for u in list_mesc_units(config.path_mesc)}
    if config.unit not in units:
        raise ValueError(f"{config.path_mesc} has no unit {config.unit}; it has {', '.join(units)}")
    unit = units[config.unit]
    if not 0 <= config.channel < unit["nchannels"]:
        raise ValueError(f"{config.unit} has {unit['nchannels']} channel(s); channel {config.channel} is not one")
    if not unit["fs"]:
        raise ValueError(f"{config.unit} declares no frame rate (TStepInMs)")
    if unit["nframes"] <= config.exclude_initial_frames:
        raise ValueError(f"{config.unit} has {unit['nframes']} frames, not more than the {config.exclude_initial_frames} excluded")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda but torch sees no GPU; request one from SLURM or pass --device cpu")
    return unit


def build_demixing_config(mad_correlation_threshold: float) -> masknmf.MultipassDemixingConfig:
    passes = []
    for _ in range(2):
        init_config = masknmf.SuperpixelInitConfig(mad_correlation_threshold=mad_correlation_threshold, residual_threshold=0.1, sign="positive")
        nmf_config = masknmf.NMFConfig(
            support_threshold=(0.95, 0.7),
            ring_model_start_pt=41,
            min_brightness=0.0,
            merge_threshold=0.7,
            reassign_background=False,
        )
        passes.append(masknmf.SinglepassDemixingConfig(init_config, nmf_config))
    return masknmf.MultipassDemixingConfig(passes)


def run_pipeline(config: Asap7Config, dir_save: Path, logger: logging.Logger) -> Path:
    unit = check_inputs(config)
    logger.info(f"{config.unit}: {unit['nframes']} frames at {unit['fs']:.1f} Hz, {unit['nchannels']} channel(s), role {unit['role'] or 'unset'}")

    arr = mbo.imread(config.path_mesc, unit=config.unit)
    fs = arr.metadata["fs"]
    raw = arr[config.exclude_initial_frames :, config.channel, 0]
    movie = masknmf.OphysArray(raw, negative_indicator=True, include_mean=True, device=config.device)
    logger.info(f"movie {tuple(movie.shape)}, first {config.exclude_initial_frames} frames dropped, inverted")

    compressor = masknmf.CompressStrategy(
        block_sizes=[4, 4],
        max_components=20,
        max_consecutive_failures=1,
        temporal_avg_factor=2,
        spatial_avg_factor=4,
        frame_batch_size=300,
    )
    raw_compressed = compressor.compress(movie)
    raw_compressed.to(config.device)
    logger.info("compressed the raw movie for registration")

    n_frames = movie.shape[0]
    detrender = MaximinSplineDetrend(
        num_frames=n_frames,
        num_knots=max(4, int(n_frames / fs / config.detrend_window_s)),
        window=int(config.detrend_window_s * fs),
        sigma=max(2.0, 0.2 * config.detrend_window_s * fs),
        device=config.device,
    )

    corrector = masknmf.RigidMotionCorrector(max_shifts=[40, 40])
    corrector.compute_template(raw_compressed)
    registered = corrector.motion_correct(reference_movie=raw_compressed, target_movie=movie)
    logger.info("registered the raw movie with shifts estimated on its compression")

    denoiser = masknmf.CompressDenoiseStrategy(
        block_sizes=[4, 4],
        max_components=20,
        max_consecutive_failures=1,
        temporal_avg_factor=2,
        spatial_avg_factor=4,
        frame_batch_size=300,
        detrender=detrender,
    )
    denoised = denoiser.compress(registered)
    denoised.to(config.device)
    logger.info(f"detrended ({config.detrend_window_s} s window) and denoised the registered movie")

    compress_config = masknmf.CompressDenoiseConfig(block_sizes=(10, 10), temporal_avg_factor=10, detrender=detrender)
    pipeline = masknmf.GlutamateCalciumSpinePipeline(
        output_folder=dir_save,
        compress_config=compress_config,
        demixing_config=build_demixing_config(mad_correlation_threshold=config.mad_correlation_threshold),
        device=config.device,
    )
    dir_run = Path(pipeline.run(glutamate_channel=denoised, calcium_channel=None, exclude_initial_frames=0))
    logger.info(f"run folder {dir_run}")
    logger.info(f"open it with: masknmf view {dir_run / 'results.glutamate.hdf5'}")
    return dir_run


def main() -> None:
    config = parse_args()
    dir_save = setup_dir_save(config)
    logger = setup_logging(dir_save)
    save_config(dir_save, config)
    logger.info(f"asap7_spine {json.dumps(dataclasses.asdict(config))}")
    run_pipeline(config, dir_save, logger)


if __name__ == "__main__":
    main()
