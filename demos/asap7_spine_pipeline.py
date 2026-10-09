import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import subprocess
    import sys
    import time
    from dataclasses import asdict
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile

    import mbo_utilities as mbo
    import masknmf
    from masknmf.pipelines.scraper import config_json_value
    from masknmf.pipelines.subcellular.glutamate_calcium_spines import NMF_JUST_HALS, get_std_based_mask

    return (
        NMF_JUST_HALS,
        Path,
        asdict,
        config_json_value,
        get_std_based_mask,
        masknmf,
        mbo,
        mo,
        np,
        plt,
        subprocess,
        sys,
        tifffile,
        time,
    )


@app.cell
def _(mo):
    mo.md("""
    # ASAP7 spine pipeline, single channel

    Each stage runs here and writes its group into one `results.hdf5` in a run folder, the same layout a pipeline writes, so `masknmf view` and the other masknmf tools open it.

    1. Denoise the movie (PMD, no neural network) and estimate the registration on the denoised movie; the shifts are applied to the raw frames.
    2. Compress and denoise the registered movie (PMD with the neural network denoiser) with no temporal or spatial averaging, so dim and fast voltage signals are kept.
    3. Demix that compressed and denoised movie: two passes for the spines, then the whole-dendrite signal, then each signal's raw trace from the registered movie.
    """)
    return


@app.cell
def _(Path):
    MESC_PATH = Path("D:/demo/scan.mesc")
    UNIT = "MSession_0/MUnit_0"
    # channel 0 is UG, the green (ASAP7) channel
    CHANNEL = 0
    EXCLUDE_INITIAL_FRAMES = 200
    OUTPUT_FOLDER = MESC_PATH.parent / "masknmf_asap7" / UNIT.replace("/", "_")
    RUN_FOLDER = None
    DEVICE = "cuda"
    MAX_SHIFTS = (10, 10)
    MAD_CORRELATION_THRESHOLD = 0.6
    return (
        CHANNEL,
        DEVICE,
        EXCLUDE_INITIAL_FRAMES,
        MAD_CORRELATION_THRESHOLD,
        MAX_SHIFTS,
        MESC_PATH,
        OUTPUT_FOLDER,
        RUN_FOLDER,
        UNIT,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Load and flip

    ASAP7 dims with depolarisation. `OphysArray` flips the movie about its mean image, so everything downstream is positive when the cell depolarises.
    """)
    return


@app.cell
def _(CHANNEL, EXCLUDE_INITIAL_FRAMES, MESC_PATH, UNIT, masknmf, mbo, mo):
    with mo.status.spinner(title=f"Loading {UNIT}"):
        arr = mbo.imread(MESC_PATH, unit=UNIT)
        fs = arr.metadata["fs"]
        raw = arr[EXCLUDE_INITIAL_FRAMES:, CHANNEL, 0]
        movie = masknmf.OphysArray(raw, negative_indicator=True, include_mean=True, device="cpu")[:].numpy()
    mo.md(f"`{UNIT}` channel {CHANNEL}: {movie.shape[0]} frames of {movie.shape[1]}×{movie.shape[2]} px at {fs:.1f} Hz, first {EXCLUDE_INITIAL_FRAMES} frames dropped.")
    return fs, movie


@app.cell
def _(mo):
    mo.md("""
    ## Denoise before registration
    """)
    return


@app.cell
def _(DEVICE, asdict, masknmf, mo, movie):
    moco_compress_config = masknmf.CompressConfig(block_sizes=(4, 4),
                                                  max_components=20,
                                                  max_consecutive_failures=1,
                                                  temporal_avg_factor=2,
                                                  spatial_avg_factor=4)
    with mo.status.spinner(title="Denoising (PMD, no neural network)"):
        pmd_moco = masknmf.CompressStrategy(**asdict(moco_compress_config), frame_batch_size=300, device=DEVICE).compress(movie)
        pmd_moco.to(DEVICE)
    mo.md(f"PMD rank {pmd_moco.temporal_compressed.shape[0]} with {moco_compress_config.block_sizes[0]}×{moco_compress_config.block_sizes[1]} blocks.")
    return moco_compress_config, pmd_moco


@app.cell
def _(mo):
    mo.md("""
    ## Register

    Shifts are estimated on the denoised movie and applied to the raw frames. The same registration is also run on the raw movie alone, for comparison.
    """)
    return


@app.cell
def _(DEVICE, MAX_SHIFTS, asdict, masknmf, mo, movie, pmd_moco):
    with mo.status.spinner(title="Registering"):
        moco_config = masknmf.RigidMotionCorrectionConfig(max_shifts=MAX_SHIFTS)
        corrector = masknmf.RigidMotionCorrector(**asdict(moco_config), device=DEVICE)
        corrector.compute_template(pmd_moco)
        moco_array = corrector.motion_correct(reference_movie=pmd_moco, target_movie=movie)

        registered = moco_array[:].cpu().numpy()
        pmd_registered = corrector.motion_correct(reference_movie=pmd_moco)[:].cpu().numpy()
        shifts = moco_array.shifts.cpu().numpy()

        corrector_raw = masknmf.RigidMotionCorrector(**asdict(moco_config), device=DEVICE)
        corrector_raw.compute_template(movie)
        shifts_raw = corrector_raw.motion_correct(movie).shifts.cpu().numpy()
    return moco_array, moco_config, pmd_registered, registered, shifts, shifts_raw


@app.cell
def _(EXCLUDE_INITIAL_FRAMES, fs, mo, movie, np, plt, shifts, shifts_raw):
    t = (EXCLUDE_INITIAL_FRAMES + np.arange(movie.shape[0])) / fs

    def _row(name, s):
        jitter = np.diff(s, axis=0).std(axis=0)
        return (f"| {name} | {s[:, 0].min():.2f} to {s[:, 0].max():.2f} | {s[:, 1].min():.2f} to {s[:, 1].max():.2f} "
                f"| {jitter[0]:.2f} / {jitter[1]:.2f} |")

    _table = mo.md("\n".join([
        "| shifts estimated on | y range (px) | x range (px) | frame-to-frame jitter y / x (px) |",
        "|---|---|---|---|",
        _row("denoised", shifts),
        _row("raw", shifts_raw),
    ]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, layout="constrained")
    for _ax, _k, _label in zip(axes, (0, 1), ("y shift (px)", "x shift (px)")):
        _ax.plot(t, shifts_raw[:, _k], lw=0.3, color="0.7", label="estimated on raw")
        _ax.plot(t, shifts[:, _k], lw=0.3, color="C0", label="estimated on denoised")
        _ax.set_ylabel(_label)
    axes[0].legend(loc="upper right", ncols=2, frameon=False)
    axes[1].set_xlabel("time (s)")
    mo.vstack([_table, fig])
    return (t,)


@app.cell
def _(mo):
    mo.md("""
    ## Compare

    One frame of each movie. Each panel keeps a fixed grey scale (1st to 99th percentile of its own movie), so the denoising shows as the drop in frame-to-frame speckle.
    """)
    return


@app.cell
def _(mo, movie):
    frame = mo.ui.slider(0, movie.shape[0] - 1, value=movie.shape[0] // 2, step=1, label="frame", full_width=True)
    frame
    return (frame,)


@app.cell
def _(frame, movie, np, plt, pmd_moco, pmd_registered, registered, t):
    panels = {
        "raw": movie,
        "pmd(raw)": pmd_moco,
        "registered": registered,
        "pmd(registered)": pmd_registered,
    }
    limits = {name: np.percentile(data[:1000], [1, 99]) for name, data in panels.items()}

    fig_frames, axes_frames = plt.subplots(2, 2, figsize=(10, 5), layout="constrained")
    for _ax, (_name, _data) in zip(axes_frames.ravel(), panels.items()):
        _ax.imshow(np.asarray(_data[frame.value]).squeeze(), cmap="gray", vmin=limits[_name][0], vmax=limits[_name][1])
        _ax.set_title(_name)
        _ax.set_axis_off()
    fig_frames.suptitle(f"frame {frame.value}, t = {t[frame.value]:.3f} s")
    fig_frames
    return


@app.cell
def _(mo):
    mo.md("""
    ## Compress, denoise and demix

    The registered movie is masked to the dendrite (Otsu threshold of its standard deviation image), then compressed with the neural network denoiser at full frame rate (`temporal_avg_factor=1`, `spatial_avg_factor=1`). Every NMF pass runs on that compressed and denoised movie, read back from the results file.

    The demixing passes use `mad_correlation_threshold` 0.6 instead of the spine pipeline's 0.8, which found no signals in ASAP7 units.

    Everything goes to `<OUTPUT_FOLDER>/<timestamp>_asap7-spine/`: `results.hdf5` with one group per stage (`RigidRegistrationArray`, `CompressionArray`, `DemixingResults`, `global/DemixingResults`), the log, `config.json` and `movie.tif`, the flipped movie the run started from, which the viewer shows as the raw panel. To reopen an earlier run instead of running again, set `RUN_FOLDER` to that run folder.
    """)
    return


@app.cell
def _(MAD_CORRELATION_THRESHOLD, masknmf):
    passes = []
    for _ in range(2):
        _init_config = masknmf.SuperpixelInitConfig(mad_correlation_threshold=MAD_CORRELATION_THRESHOLD,
                                                    residual_threshold=0.1,
                                                    sign="positive")
        _nmf_config = masknmf.NMFConfig(support_threshold=(0.95, 0.7),
                                        ring_model_start_pt=41,
                                        min_brightness=0.0,
                                        merge_threshold=0.7,
                                        reassign_background=False)
        passes.append(masknmf.SinglepassDemixingConfig(_init_config, _nmf_config))
    demixing_config = masknmf.MultipassDemixingConfig(passes)
    compress_config = masknmf.CompressDenoiseConfig(block_sizes=(10, 10), temporal_avg_factor=1, spatial_avg_factor=1)
    return compress_config, demixing_config


@app.cell
def _(
    CHANNEL,
    DEVICE,
    EXCLUDE_INITIAL_FRAMES,
    MESC_PATH,
    NMF_JUST_HALS,
    OUTPUT_FOLDER,
    Path,
    RUN_FOLDER,
    UNIT,
    asdict,
    compress_config,
    config_json_value,
    demixing_config,
    fs,
    get_std_based_mask,
    masknmf,
    mo,
    moco_array,
    moco_compress_config,
    moco_config,
    movie,
    registered,
    tifffile,
    time,
):
    if RUN_FOLDER is None:
        run_folder = masknmf.io.create_run_folder(OUTPUT_FOLDER, "asap7-spine")
        masknmf.io.log_to(run_folder)
        results_path = run_folder / "results.hdf5"
        tifffile.imwrite(run_folder / "movie.tif", movie)
        timings = {}
        with mo.status.spinner(title="Running", subtitle=f"log in {run_folder}") as _spinner:
            _start = time.monotonic()
            moco_array.export(results_path)
            timings["registration export"] = round(time.monotonic() - _start, 1)

            _spinner.update(title="Compressing and denoising")
            _start = time.monotonic()
            mask = get_std_based_mask(registered)
            video = registered * mask.astype("float32")[None]
            pmd_denoised = masknmf.CompressDenoiseStrategy(**asdict(compress_config), frame_batch_size=300, device=DEVICE).compress(video)
            pmd_denoised.export(results_path)
            timings["compression"] = round(time.monotonic() - _start, 1)

            # the nmf passes take the compressed and denoised movie as the file holds it
            pmd_denoised = masknmf.CompressionArray.from_hdf5(results_path)
            pmd_denoised.to(DEVICE)
            demixer = masknmf.demixing.signal_demixer.SignalDemixer(pmd_denoised, device=DEVICE)
            spines = None
            for _i, _singlepass in enumerate(demixing_config.DemixingConfigs):
                _spinner.update(title=f"Spine demixing pass {_i + 1} of {len(demixing_config.DemixingConfigs)}")
                _start = time.monotonic()
                try:
                    demixer.initialize_signals(**asdict(_singlepass.InitConfig))
                except masknmf.demixing.NoSignalsDetectedError:
                    if spines is None:
                        raise ValueError("no signals found; lower MAD_CORRELATION_THRESHOLD")
                    break
                demixer.demix(**asdict(_singlepass.NMFConfig))
                spines = demixer.results
                timings[f"spine demixing pass {_i + 1}"] = round(time.monotonic() - _start, 1)

            _spinner.update(title="Whole-dendrite demixing")
            _start = time.monotonic()
            demixer_global = masknmf.demixing.signal_demixer.SignalDemixer(pmd_denoised, device=DEVICE)
            demixer_global.initialize_signals(is_custom=True, spatial_footprints=mask[:, :, None].astype("float"))
            demixer_global.demix(**NMF_JUST_HALS)
            dendrite = demixer_global.results
            timings["whole-dendrite demixing"] = round(time.monotonic() - _start, 1)

            _spinner.update(title="Raw traces")
            _start = time.monotonic()
            for _results in (spines, dendrite):
                _results.temporal_demixed_raw = masknmf.demixing.estimate_temporal_demixed_raw(
                    _results, video, device=DEVICE, nonneg=True, frame_batch_size=300)
            spines.export(results_path)
            dendrite.export(results_path, prefix="global")
            timings["raw traces"] = round(time.monotonic() - _start, 1)

        masknmf.io.write_run_config(
            run_folder,
            "asap7_spine_pipeline",
            {"exclude_initial_frames": EXCLUDE_INITIAL_FRAMES, "moco_compress_config": moco_compress_config,
             "motion_correct_config": moco_config, "compress_config": compress_config,
             "demixing_config": demixing_config, "frame_rate": fs},
            inputs={"movie": {"path": str(MESC_PATH), "name": MESC_PATH.name, "unit": UNIT, "channel": CHANNEL},
                    "raw": {"path": str(run_folder / "movie.tif"), "name": "movie.tif"}},
            timings=timings,
            default=config_json_value,
        )
    else:
        run_folder = Path(RUN_FOLDER).expanduser()
        results_path = run_folder / "results.hdf5"
    mo.md(f"Run folder: `{run_folder}`; stages in `results.hdf5`: {', '.join(masknmf.io.stage_groups(results_path))}")
    return results_path, run_folder


@app.cell
def _(mo):
    mo.md("""
    ## Results

    `OpenedResults` reads the file the same way the viewer does: the spine results, `movie.tif` in the same folder as the raw movie, the registration applied to it, and the shifts and template.
    """)
    return


@app.cell
def _(DEVICE, masknmf, mo, results_path):
    opened = masknmf.io.OpenedResults.open(results_path, device=DEVICE)
    spines_opened = opened.results
    dendrite_opened = masknmf.DemixingResults.from_hdf5(results_path, prefix="global", device=DEVICE)
    footprints = spines_opened.signals_array.export_spatial_demixed()
    mo.md("\n".join([
        f"{footprints.shape[-1]} spines and the whole-dendrite component.",
        "",
        f"- raw movie: `{opened.raw_source}`",
        f"- registered movie: {type(opened.registered).__name__}",
        f"- shifts: {None if opened.shifts is None else opened.shifts.shape}, template: {None if opened.template is None else opened.template.shape}",
        *[f"- skipped: {note}" for note in opened.skipped],
    ]))
    return dendrite_opened, footprints, opened, spines_opened


@app.cell
def _(dendrite_opened, footprints, np, opened, plt, spines_opened, t):
    _colors = plt.get_cmap("tab10")(np.arange(footprints.shape[-1]) % 10)[:, :3]
    _rgb = np.zeros((*footprints.shape[:2], 3))
    for _k in range(footprints.shape[-1]):
        _weight = footprints[..., _k] / max(footprints[..., _k].max(), 1e-12)
        _rgb += _weight[..., None] * _colors[_k]

    fig_results, (ax_map, ax_traces) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained")
    ax_map.imshow(opened.template if opened.template is not None else np.asarray(opened.raw[:]).mean(axis=0), cmap="gray")
    ax_map.imshow(np.clip(_rgb, 0, 1), alpha=0.7)
    ax_map.set_axis_off()
    ax_map.set_title("spine footprints")

    _traces = spines_opened.temporal_demixed.cpu().numpy()
    _dendrite = dendrite_opened.temporal_demixed.cpu().numpy()[:, 0]
    _step = 1.1 * max(_traces.max(), _dendrite.max())
    for _k in range(_traces.shape[1]):
        ax_traces.plot(t, _traces[:, _k] + _k * _step, lw=0.3, color=_colors[_k])
    ax_traces.plot(t, _dendrite + _traces.shape[1] * _step, lw=0.3, color="k")
    ax_traces.set_yticks(np.arange(_traces.shape[1] + 1) * _step, [*map(str, range(_traces.shape[1])), "dendrite"])
    ax_traces.set_xlabel("time (s)")
    ax_traces.set_title("demixed traces (flipped units, positive = depolarisation)")
    fig_results
    return


@app.cell
def _(mo):
    mo.md("""
    The masknmf viewer (`masknmf view`) does not run inside marimo; the button opens it in its own window. It uses `movie.tif` in the results folder as the raw panel, and the registration in the file for the registered movie and shift traces; `--prefix global` opens the whole-dendrite result instead.
    """)
    return


@app.cell
def _(mo):
    open_viewer = mo.ui.run_button(label="Open the masknmf viewer")
    open_viewer
    return (open_viewer,)


@app.cell
def _(fs, mo, open_viewer, results_path, subprocess, sys):
    _command = [sys.executable, "-m", "masknmf.cli", "view", str(results_path), "--fs", str(fs)]
    if open_viewer.value:
        subprocess.Popen(_command)
    mo.md(f"```\n{' '.join(_command[1:]).replace('-m masknmf.cli', 'masknmf')}\n```")
    return


if __name__ == "__main__":
    app.run()
