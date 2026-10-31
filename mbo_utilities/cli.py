"""
CLI entry point for mbo_utilities.

This module handles command-line operations with minimal imports.
GUI-related imports are deferred until actually needed.

Usage patterns:
  mbo                           # Open GUI with file dialog
  mbo /path/to/data             # Open GUI with specific file
  mbo run/demixing_results.hdf5 # masknmf demixing results in masknmf's viewer
  mbo /path/to/data --metadata  # Show only metadata
  mbo convert INPUT OUTPUT      # Convert with CLI args
  mbo info INPUT                # Show array info (CLI only)
"""

import logging
import sys
import threading
import time
from pathlib import Path

if sys.platform == "win32":
    import ctypes

    # the taskbar groups windows by this id, so it is set before any window exists
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "mbo.utilities.gui.1.0"
    )

import click

from mbo_utilities import log


class PathAwareGroup(click.Group):
    """A click group where a first argument that is not a command but looks like a path (separator, suffix, or exists) runs ``view`` on it."""

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands:
            first_arg = args[0]
            if (
                "/" in first_arg
                or "\\" in first_arg
                or "." in first_arg
                or Path(first_arg).exists()
            ):
                return "view", self.commands["view"], args
        return super().resolve_command(ctx, args)


def _get_marker_path() -> Path:
    from mbo_utilities import get_mbo_dirs

    return get_mbo_dirs()["base"] / ".initialized"


def _is_first_run() -> bool:
    return not _get_marker_path().exists()


def _mark_initialized() -> None:
    marker = _get_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


class LoadingSpinner:
    """A terminal spinner on stdout while heavy imports run."""

    def __init__(self, message: str = "Loading"):
        self.message = message
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def _spin(self):
        chars = "|/-\\"  # ascii: a cp1252 console cannot encode the unicode spinners
        i = 0
        while self._running:
            sys.stdout.write(f"\r{chars[i % len(chars)]} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


def _get_version_info() -> str:
    """The version, interpreter and install kind (uv tool, environment, conda, system), the kind from the cache when it has one."""
    from mbo_utilities import __version__
    from mbo_utilities.env_cache import get_cached_install_type

    install_type = get_cached_install_type()
    if not install_type:
        exe_str = sys.executable.lower()
        if ".local" in exe_str or ("uv" in exe_str and "tools" in exe_str):
            install_type = "uv tool"
        elif "envs" in exe_str or "venv" in exe_str or ".venv" in exe_str:
            install_type = "environment"
        elif "conda" in exe_str or "miniconda" in exe_str or "anaconda" in exe_str:
            install_type = "conda"
        else:
            install_type = "system"

    return f"mbo_utilities {__version__}\nPython: {sys.executable}\nInstall: {install_type}"


def _version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(_get_version_info())
    ctx.exit()


@click.group(cls=PathAwareGroup, invoke_without_command=True)
@click.option(
    "-V",
    "--version",
    is_flag=True,
    callback=_version_callback,
    expose_value=False,
    is_eager=True,
    help="Show version and installation info.",
)
@click.option(
    "--check-install",
    is_flag=True,
    help="Verify the installation of mbo_utilities and dependencies.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Bypass environment cache (forces fresh checks).",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear environment cache and exit.",
)
@click.option(
    "--gpu",
    "gpu_index_arg",
    type=int,
    default=None,
    help="Set and persist GPU adapter index. `mbo --gpu N` saves + exits; "
    "`mbo --gpu N <path>` saves then opens. See `mbo view --list-gpus`.",
)
@click.option(
    "--debug/--no-debug",
    default=None,
    help="Debug logging for this run (sets MBO_DEBUG): verbose logs, the "
    "Widgets > ImGui Debug tools, and every worker spawned after. "
    "Omitted: the persisted Options setting.",
)
@click.pass_context
def main(
    ctx,
    check_install=False,
    no_cache=False,
    clear_cache=False,
    gpu_index_arg=None,
    debug=None,
):
    r"""
    MBO Utilities - preview and process imaging data.

    \b
    Open the viewer:
      mbo                   file selection dialog
      mbo PATH              open a file or folder
      mbo PATH --metadata   print metadata only

    \b
    Common commands:
      mbo init [PATH]       create starter notebooks
      mbo convert IN OUT    convert between formats
      mbo info PATH         print array info
      mbo formats           list supported formats
      mbo shortcut          create a desktop icon

    \b
    Verify install:
      mbo --check-install
    """
    if debug is not None:
        from mbo_utilities import log as _log

        _log.set_debug(debug)

    if clear_cache:
        from mbo_utilities.env_cache import clear_cache as do_clear
        from mbo_utilities.env_cache import get_cache_path

        path = get_cache_path()
        if do_clear():
            click.secho(f"Cache cleared: {path}", fg="green")
        else:
            click.secho("No cache to clear.", fg="yellow")
        return

    if check_install:
        from mbo_utilities.env_cache import (
            build_full_cache_with_install_status,
            save_cache,
        )

        cache = build_full_cache_with_install_status()
        save_cache(cache)
        from mbo_utilities.gui.run_gui import _check_installation

        _check_installation()
        return

    if gpu_index_arg is not None:
        import fastplotlib as fpl

        adapters = fpl.enumerate_adapters()
        if not 0 <= gpu_index_arg < len(adapters):
            raise click.BadParameter(
                f"--gpu {gpu_index_arg} out of range; found {len(adapters)} "
                f"adapter(s). Run `mbo view --list-gpus` to see them.",
                param_hint="--gpu",
            )
        from mbo_utilities.preferences import set_gpu_index

        set_gpu_index(gpu_index_arg)
        info = getattr(adapters[gpu_index_arg], "info", {}) or {}
        click.echo(
            f"Saved GPU {gpu_index_arg}: "
            f"{info.get('device', info.get('description', '?'))} "
            f"(persisted to ~/.mbo/settings/preferences.json)"
        )
        # run_gui reads the persisted index on launch, so `view` picks it up
        if ctx.invoked_subcommand is None:
            return

    if ctx.invoked_subcommand is not None:
        return

    first_run = _is_first_run()
    if first_run:
        click.secho(
            "First run detected - initial startup may take longer while caches are built.",
            fg="yellow",
        )
    if not no_cache:
        from mbo_utilities.env_cache import ensure_cache

        ensure_cache()

    spinner = LoadingSpinner("Loading GUI")
    spinner.start()
    try:
        from mbo_utilities.gui.run_gui import run_gui
    finally:
        spinner.stop()
    if first_run:
        _mark_initialized()

    run_gui(data_in=None, roi=None, widget="preview", metadata_only=False)


@main.command()
@click.argument("data_in", required=False, type=click.Path())
@click.option(
    "--roi",
    multiple=True,
    type=int,
    help="ROI index (can pass multiple: --roi 0 --roi 2).",
)
@click.option(
    "--widget",
    is_flag=False,
    flag_value="preview",
    default="preview",
    help="Side widget to attach: 'preview' (default), 'manualroi' (preview with the ROIs widget on) or 'none'.",
)
@click.option(
    "--no-widget",
    is_flag=True,
    help="Open the viewer with no side widget.",
)
@click.option(
    "--metadata",
    is_flag=True,
    help="Show only metadata (no image viewer).",
)
@click.option(
    "--unit",
    default=None,
    help=(
        "For .mesc files: which measurement unit to open, as an index (0, 1, ...) "
        "or a 'MSession_0/MUnit_3' key. Omit to get a picker. `mbo info` lists them."
    ),
)
@click.option(
    "--dims",
    default=None,
    help=(
        "For a plain TIFF: which axis its pages are, as 'TYX' (time, the default "
        "reading), 'ZYX' (a z-stack) or 'CYX' (channels). The Image tab's Page Axis "
        "panel changes it in the viewer."
    ),
)
@click.option(
    "--gpu",
    "gpu_index",
    type=int,
    default=None,
    help="0-based GPU adapter index (see --list-gpus). Default: wgpu auto-pick.",
)
@click.option(
    "--list-gpus",
    is_flag=True,
    help="List available GPU adapters and exit.",
)
@click.option(
    "--vis",
    type=click.Choice(["demixing", "classification"]),
    default=None,
    help="For a masknmf demixing result: which of masknmf's viewers to open. "
    "Omitted: a prompt in the terminal, or the demixing viewer when there is none.",
)
@click.option(
    "--raw",
    "raw_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="For a masknmf demixing result: the raw movie, shown as a panel in the demixing viewer. "
    "Omitted: the recording a run folder's config.json names, else data_raw.bin beside the result, "
    "if any. For a pre-registration denoising result: the recording to read in place of the one "
    "its provenance names.",
)
@click.option(
    "--motion-correction",
    "motion_correction_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="For a masknmf demixing result: a motion correction hdf5 whose shifts plot above the traces. "
    "Omitted: the registration in a run folder's results.hdf5, else motion_correction.hdf5 beside the result, if any.",
)
@click.option(
    "--qc",
    is_flag=True,
    help="For a MaskNMF run folder (or its results.hdf5): open Registration-Denoising "
    "Quality Control, raw | registered over both as PMD with RTMC, masknmf shifts, frame "
    "means and a box you drag. Omitted: a demixed run opens masknmf's demixing viewer, "
    "any other run the Studio.",
)
@click.option(
    "--debug/--no-debug",
    default=None,
    help="Debug logging for this run (sets MBO_DEBUG): verbose logs, the "
    "Widgets > ImGui Debug tools, and every worker spawned after. "
    "Omitted: the persisted Options setting.",
)
def view(
    data_in=None,
    roi=None,
    widget="preview",
    no_widget=False,
    metadata=False,
    unit=None,
    dims=None,
    gpu_index=None,
    list_gpus=False,
    vis=None,
    raw_path=None,
    motion_correction_path=None,
    qc=False,
    debug=None,
):
    r"""
    Open imaging data in the GUI viewer.

    \b
    Examples:
      mbo view                       Open file selection dialog
      mbo view /data/raw.tiff        Open specific file
      mbo view /data/raw --metadata  Show only metadata
      mbo view /data --roi 0 --roi 2 View specific ROIs
      mbo view /data --widget manualroi  Open with the ROIs widget on (draw + label by hand)
      mbo view /data/scan.mesc --unit 2   Open one MESc measurement unit
      mbo view /data/scan.mesc --unit MUnit_35   A line-scan unit opens the line-scan viewer
      mbo view /data/stack.tif --dims ZYX   A plain TIFF whose pages are z-planes
      mbo view run/demixing_results.hdf5 --vis classification   masknmf's classification viewer
      mbo out/20261008T143623_masknmf_zplane01 --qc   raw, registered, PMD, RTMC and shifts
      mbo view --list-gpus           Show available GPU adapters
      mbo view /data/raw --gpu 0     Force GPU index 0
    """
    if debug is not None:
        from mbo_utilities import log as _log

        _log.set_debug(debug)

    if list_gpus:
        import fastplotlib as fpl

        adapters = fpl.enumerate_adapters()
        click.echo(f"{'idx':<4} {'type':<14} {'vendor':<10} {'name'}")
        for i, a in enumerate(adapters):
            info = getattr(a, "info", {}) or {}
            type_ = info.get("adapter_type", info.get("device_type", "?"))
            vendor = info.get("vendor", "?")
            name = info.get("device", info.get("description", "?"))
            click.echo(f"{i:<4} {str(type_):<14} {str(vendor):<10} {name}")
        return

    if gpu_index is not None:
        import fastplotlib as fpl

        adapters = fpl.enumerate_adapters()
        if not 0 <= gpu_index < len(adapters):
            raise click.BadParameter(
                f"--gpu {gpu_index} out of range; found {len(adapters)} adapter(s). "
                f"Run `mbo view --list-gpus` to see them.",
                param_hint="--gpu",
            )
        fpl.select_adapter(adapters[gpu_index])
        from mbo_utilities.preferences import set_gpu_index

        set_gpu_index(gpu_index)
        info = getattr(adapters[gpu_index], "info", {}) or {}
        click.echo(
            f"Using GPU {gpu_index}: {info.get('device', info.get('description', '?'))}"
        )

    # a masknmf demixing result opens in one of masknmf's viewers: ask which before anything heavy loads
    if vis is None and data_in and not metadata and not qc:
        from mbo_utilities.arrays.demixing import has_demixing_results
        from mbo_utilities.arrays.masknmf_run import run_demixing

        if has_demixing_results(run_demixing(data_in) or data_in):
            vis = "demixing"
            if sys.stdin.isatty():
                vis = click.prompt(
                    "masknmf viewer",
                    type=click.Choice(["demixing", "classification"]),
                    default="demixing",
                    show_choices=True,
                )
    vis = vis or "demixing"

    first_run = _is_first_run()
    if first_run:
        click.secho(
            "First run detected - initial startup may take longer while caches are built.",
            fg="yellow",
        )
    spinner = LoadingSpinner("Loading GUI")
    spinner.start()
    try:
        from mbo_utilities.gui.run_gui import run_gui
    finally:
        spinner.stop()
    if first_run:
        _mark_initialized()

    run_gui(
        data_in=data_in,
        roi=roi if roi else None,
        widget="none" if no_widget else widget,
        metadata_only=metadata,
        unit=_parse_unit(unit),
        dims=dims,
        vis=vis,
        raw_path=raw_path,
        motion_correction_path=motion_correction_path,
        qc=qc,
    )


@main.command()
@click.argument("input_path", required=False, type=click.Path())
@click.argument("output_path", required=False, type=click.Path())
@click.option(
    "-e",
    "--ext",
    type=click.Choice(
        [".tiff", ".tif", ".zarr", ".bin", ".h5", ".npy"], case_sensitive=False
    ),
    default=None,
    help="Output format extension.",
)
@click.option(
    "-p",
    "--planes",
    multiple=True,
    type=int,
    help="Z-planes to export (1-based): -p 1 -p 7 -p 14",
)
@click.option(
    "-t",
    "--timepoints",
    multiple=True,
    type=int,
    help="Timepoints to export (1-based): -t 1 -t 50 -t 100",
)
@click.option(
    "--num-timepoints",
    type=int,
    default=None,
    help="Number of timepoints to export (first N).",
)
@click.option(
    "--num-zplanes",
    type=int,
    default=None,
    help="Number of z-planes to export (first N).",
)
@click.option(
    "-n",
    "--num-frames",
    type=int,
    default=None,
    help="Deprecated, use --num-timepoints.",
)
@click.option(
    "--roi",
    type=str,
    default=None,
    help="ROI: None=stitch, 0=split, N=specific, '1,3'=multiple.",
)
@click.option(
    "--register-z/--no-register-z",
    default=False,
    help="Z-plane registration using Suite3D.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing output files.",
)
@click.option(
    "--chunk-mb",
    type=int,
    default=100,
    help="Chunk size in MB for streaming writes.",
)
@click.option(
    "--fix-phase/--no-fix-phase",
    default=None,
    help="Bidirectional phase correction (ScanImageArray).",
)
@click.option(
    "--phasecorr-method",
    type=click.Choice(["mean", "median", "max"]),
    default=None,
    help="Phase correction method.",
)
@click.option(
    "--ome/--no-ome",
    default=True,
    help="Write OME-Zarr metadata (zarr only).",
)
@click.option(
    "--output-name",
    type=str,
    default=None,
    help="Output filename for binary format.",
)
@click.option(
    "-c",
    "--channels",
    multiple=True,
    type=int,
    help="Color channels to export (1-based): -c 1 -c 2",
)
@click.option(
    "--order",
    multiple=True,
    type=int,
    help="Reorder planes before writing (0-based indices into --planes).",
)
@click.option(
    "--output-suffix",
    type=str,
    default=None,
    help="Suffix appended to output filenames.",
)
@click.option(
    "--dataset-name",
    type=str,
    default=None,
    help="HDF5 dataset name (.h5 only). Default: mov.",
)
@click.option(
    "--roi-mode",
    type=click.Choice(["concat_y", "separate"]),
    default=None,
    help="Multi-ROI handling: concat_y=stitch, separate=one file per ROI.",
)
@click.option(
    "--compressor",
    type=click.Choice(["none", "gzip", "zstd", "blosc-lz4", "blosc-zstd"]),
    default=None,
    help="Zarr compressor (.zarr only).",
)
@click.option(
    "--compression-level",
    type=int,
    default=None,
    help="Zarr compression level (.zarr only).",
)
@click.option(
    "--sharded/--no-sharded",
    default=None,
    help="Zarr v3 sharding (.zarr only).",
)
@click.option(
    "--pyramid/--no-pyramid",
    default=None,
    help="Write multiscale pyramid (.zarr only).",
)
@click.option(
    "--pyramid-max-layers",
    type=int,
    default=None,
    help="Max pyramid resolution levels (.zarr only).",
)
@click.option(
    "--pyramid-method",
    type=click.Choice(["mean", "median", "mode", "gaussian", "nearest", "local_mean"]),
    default=None,
    help="Pyramid downsampling method (.zarr only).",
)
@click.option(
    "--border",
    type=int,
    default=None,
    help="Phase correction: border pixels excluded from estimation.",
)
@click.option(
    "--max-offset",
    type=int,
    default=None,
    help="Phase correction: max pixel offset to search.",
)
@click.option(
    "--use-fft/--no-use-fft",
    default=None,
    help="Phase correction: FFT-based 2D correction.",
)
@click.option(
    "--frame-average",
    type=int,
    default=None,
    help="Average every N consecutive timepoints into one (temporal binning).",
)
@click.option(
    "--reg-max-frames",
    type=int,
    default=None,
    help="register-z: subsample frame count (default 200).",
)
@click.option(
    "--reg-chunk-frames",
    type=int,
    default=None,
    help="register-z: streaming batch size (default 10).",
)
@click.option(
    "--reg-max-xy",
    type=int,
    default=None,
    help="register-z: search radius in pixels (default 30).",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Verbose debug logging.",
)
def convert(
    input_path,
    output_path,
    ext,
    planes,
    timepoints,
    num_timepoints,
    num_zplanes,
    num_frames,
    roi,
    register_z,
    overwrite,
    chunk_mb,
    fix_phase,
    phasecorr_method,
    ome,
    output_name,
    channels,
    order,
    output_suffix,
    dataset_name,
    roi_mode,
    compressor,
    compression_level,
    sharded,
    pyramid,
    pyramid_max_layers,
    pyramid_method,
    border,
    max_offset,
    use_fft,
    frame_average,
    reg_max_frames,
    reg_chunk_frames,
    reg_max_xy,
    debug,
):
    r"""
    Convert imaging data between formats.

    If INPUT_PATH and OUTPUT_PATH are provided, runs conversion directly.
    If omitted, opens a GUI for interactive conversion.

    \b
    Examples:
      mbo convert                                    # Open conversion GUI
      mbo convert /data/raw output/ -e .zarr        # Convert to Zarr
      mbo convert /data/raw output/ -e .npy -p 1 -p 7   # Export planes as NPY
      mbo convert /data/raw output/ --fix-phase     # With phase correction
    """
    if input_path is None:
        click.echo(
            "Conversion GUI not yet implemented. Please provide INPUT_PATH and OUTPUT_PATH."
        )
        click.echo("\nUsage: mbo convert INPUT_PATH OUTPUT_PATH [OPTIONS]")
        click.echo("\nRun 'mbo convert --help' for all options.")
        return

    if output_path is None:
        click.secho(
            "Error: OUTPUT_PATH is required when INPUT_PATH is provided.", fg="red"
        )
        return

    from mbo_utilities import imread, imwrite

    parsed_roi = None
    if roi is not None:
        roi = roi.strip()
        if roi.lower() == "none":
            parsed_roi = None
        elif "," in roi:
            parsed_roi = [int(x.strip()) for x in roi.split(",")]
        else:
            parsed_roi = int(roi)

    parsed_planes = list(planes) if planes else None
    parsed_timepoints = list(timepoints) if timepoints else None
    parsed_channels = list(channels) if channels else None
    parsed_order = list(order) if order else None

    if num_frames and not num_timepoints:
        click.echo("--num-frames is deprecated, use --num-timepoints.")
        num_timepoints = num_frames

    click.echo(f"Reading: {input_path}")

    imread_kwargs = {}
    if fix_phase is not None:
        imread_kwargs["fix_phase"] = fix_phase
    if phasecorr_method:
        imread_kwargs["phasecorr_method"] = phasecorr_method
    if border is not None:
        imread_kwargs["border"] = border
    if max_offset is not None:
        imread_kwargs["max_offset"] = max_offset
    if use_fft is not None:
        imread_kwargs["use_fft"] = use_fft
    if frame_average is not None and frame_average > 1:
        imread_kwargs["frame_average"] = frame_average
    if parsed_roi is not None and parsed_roi != 0:
        imread_kwargs["roi"] = parsed_roi

    data = imread(input_path, **imread_kwargs)
    click.echo(f"  Shape: {data.shape}, dtype: {data.dtype}")

    output_ext = ext or ".tiff"
    click.echo(f"Writing: {output_path} (format: {output_ext})")

    imwrite_kwargs = {
        "ext": output_ext,
        "overwrite": overwrite,
        "target_chunk_mb": chunk_mb,
        "debug": debug,
    }

    if parsed_planes:
        imwrite_kwargs["planes"] = parsed_planes
    if parsed_timepoints:
        imwrite_kwargs["timepoints"] = parsed_timepoints
    if parsed_channels:
        imwrite_kwargs["channels"] = parsed_channels
    if num_timepoints:
        imwrite_kwargs["num_timepoints"] = num_timepoints
    if num_zplanes:
        imwrite_kwargs["num_zplanes"] = num_zplanes

    # an explicit --roi implies separate outputs (concat_y ignores roi); --roi-mode overrides
    roi_mode_kw = roi_mode
    if roi_mode_kw is None and parsed_roi is not None:
        roi_mode_kw = "separate"
    if parsed_roi is not None:
        imwrite_kwargs["roi"] = parsed_roi
    if roi_mode_kw is not None:
        imwrite_kwargs["roi_mode"] = roi_mode_kw

    if parsed_order:
        if not parsed_planes:
            click.secho("--order requires --planes; ignoring --order.", fg="yellow")
        else:
            imwrite_kwargs["order"] = parsed_order

    if register_z:
        imwrite_kwargs["register_z"] = True
        if reg_max_frames is not None:
            imwrite_kwargs["max_frames"] = reg_max_frames
        if reg_chunk_frames is not None:
            imwrite_kwargs["chunk_frames"] = reg_chunk_frames
        if reg_max_xy is not None:
            imwrite_kwargs["max_reg_xy"] = reg_max_xy

    if output_ext.lower() == ".zarr":
        imwrite_kwargs["ome"] = ome
        if compressor is not None:
            imwrite_kwargs["compressor"] = compressor
        if compression_level is not None:
            imwrite_kwargs["compression_level"] = compression_level
        if sharded is not None:
            imwrite_kwargs["sharded"] = sharded
        if pyramid is not None:
            imwrite_kwargs["pyramid"] = pyramid
        if pyramid_max_layers is not None:
            imwrite_kwargs["pyramid_max_layers"] = pyramid_max_layers
        if pyramid_method is not None:
            imwrite_kwargs["pyramid_method"] = pyramid_method

    if dataset_name and output_ext.lower() in (".h5", ".hdf5"):
        imwrite_kwargs["dataset_name"] = dataset_name
    if output_name:
        imwrite_kwargs["output_name"] = output_name
    if output_suffix:
        imwrite_kwargs["output_suffix"] = output_suffix

    result = imwrite(data, output_path, **imwrite_kwargs)
    click.secho(f"\nDone! Output saved to: {result}", fg="green")


def _info_num(v):
    """Compact number formatting: 30.0 -> '30', 0.5 -> '0.5'. None -> None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:  # nan
            return None
        return f"{v:g}"
    return str(v)


def _info_md(md, name):
    """get_param value, or None when the key is truly absent.

    get_param falls back to a parameter's registered default (e.g. dx=1.0),
    which would mask missing values; a distinct sentinel detects real absence.
    """
    from mbo_utilities.metadata import get_param

    sentinel = object()
    val = get_param(md, name, default=sentinel)
    return None if val is sentinel else val


def _info_dim_sizes(data):
    """Map of dim label (T/C/Z/Y/X) -> size, from shape + dims labels."""
    dims = getattr(data, "dims", None)
    shape = getattr(data, "shape", ()) or ()
    out = {}
    if dims and len(dims) == len(shape):
        for d, s in zip(dims, shape):
            out[str(d).upper()] = int(s)
    return out


_INFO_SKIP_SUFFIXES = {".zarr", ".corrected", ".fused", ".registered"}


def _info_find_ops_dirs(root, max_depth=2):
    """Directories at or under ``root`` (inclusive, ``max_depth`` levels) that contain ``ops.npy``, in walk order."""
    root = Path(root)
    found = [root] if (root / "ops.npy").exists() else []
    pending = [(root, 1)]
    while pending:
        d, depth = pending.pop(0)
        if depth > max_depth:
            continue
        for child in sorted(d.iterdir()):
            if not child.is_dir() or child.suffix.lower() in _INFO_SKIP_SUFFIXES:
                continue
            if (child / "ops.npy").exists() and child not in found:
                found.append(child)
            pending.append((child, depth + 1))
    return found


def _info_segmentation_counts(ops_dir):
    """(n_rois, n_cells) from stat.npy / iscell.npy, or (None, None)."""
    import numpy as np

    from mbo_utilities.file_io import load_npy

    stat_path = ops_dir / "stat.npy"
    if not stat_path.exists():
        return None, None
    n_rois = len(load_npy(stat_path))
    n_cells = None
    iscell_path = ops_dir / "iscell.npy"
    if iscell_path.exists():
        iscell = np.asarray(load_npy(iscell_path))
        n_cells = int(iscell[:, 0].astype(bool).sum())
    return n_rois, n_cells


def _parse_unit(unit):
    """Coerce a --unit value to an int index when it looks like one."""
    if unit is None:
        return None
    text = str(unit).strip()
    return int(text) if text.isdigit() else text


def _echo_mesc_units(input_path, unit):
    """Print the measurement units of a .mesc; return imread kwargs for `unit`.

    A .mesc holds one MUnit per scan, so `mbo info` shows the whole table
    before describing the single unit it opens.
    """
    from pathlib import Path

    path = Path(input_path)
    if not (path.is_file() and path.suffix.lower() == ".mesc"):
        return {}

    from mbo_utilities.arrays.mesc import list_mesc_units

    units = list_mesc_units(path)
    click.echo("")
    click.secho(f"Measurement units ({len(units)})", fg="cyan")
    click.echo(
        f"  {'idx':<5}{'unit':<12}{'type':<20}{'role':<18}{'shape (T,C,Z,Y,X)':<26}comment"
    )
    for u in units:
        shape = ",".join(str(v) for v in u["shape"])
        click.echo(
            f"  {u['index']:<5}{u['munit']:<12}{u['modality_name']:<20}{(u.get('role') or '-'):<18}"
            f"({shape}){'':<{max(0, 24 - len(shape))}}{u['comment']}"
        )
    if unit is None and len(units) > 1:
        click.secho(f"  -> no --unit given; describing {units[0]['key']}", fg="yellow")
    return {"unit": _parse_unit(unit)} if unit is not None else {}


def _info_kv(label, value, unit=""):
    """Print an aligned label/value line; skip when value is None/empty."""
    if value is None or value == "":
        return
    line = f"  {label:<15} {value}"
    if unit:
        line += f" {unit}"
    click.echo(line)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--metadata/--no-metadata",
    default=True,
    help="Show imaging/acquisition metadata sections.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Also dump the full raw metadata dict.",
)
@click.option(
    "--unit",
    default=None,
    help="For .mesc files: which measurement unit to describe (index or key).",
)
def info(input_path, metadata, show_all, unit):
    r"""
    Show information about an imaging dataset.

    \b
    Examples:
      mbo info /data/raw.tiff
      mbo info /data/volume.zarr
      mbo info /data/suite2p/plane0
      mbo info /data/raw --all
      mbo info /data/scan.mesc --unit 2
    """
    from pathlib import Path

    from mbo_utilities import imread

    reader_kwargs = _echo_mesc_units(input_path, unit)

    click.echo(f"Loading: {input_path}")
    data = imread(input_path, **reader_kwargs)
    md = getattr(data, "metadata", None) or {}
    sizes = _info_dim_sizes(data)

    is_isoview = hasattr(data, "is_tiled")
    is_tiled = bool(getattr(data, "is_tiled", False))

    dims = getattr(data, "dims", None)
    if dims and is_isoview:
        dims = tuple("V" if d == "C" else d for d in dims)
    shape_str = str(tuple(data.shape))
    if dims and len(dims) == len(data.shape):
        shape_str += f"  [{', '.join(dims)}]"

    click.echo("")
    click.secho(str(input_path), bold=True)
    _info_kv("Type", type(data).__name__)
    _info_kv("Shape", shape_str)
    _info_kv("Dtype", str(data.dtype))

    if metadata:
        fs = _info_md(md, "fs")
        dx = _info_md(md, "dx")
        dy = _info_md(md, "dy")
        dz = _info_md(md, "dz")
        ny = sizes.get("Y", data.shape[-2] if data.ndim >= 2 else None)
        nx = sizes.get("X", data.shape[-1] if data.ndim >= 1 else None)
        if is_isoview and is_tiled:
            n_timepoints = 1
        else:
            n_timepoints = sizes.get("T") or _info_md(md, "num_timepoints")

        click.secho("\nImaging", fg="cyan")
        if n_timepoints != 1:
            _info_kv("Frame rate", _info_num(fs), "Hz")
        if dx is not None or dy is not None:
            _info_kv("Pixel size", f"{_info_num(dx)} x {_info_num(dy)}", "um")
        _info_kv("Z step", _info_num(dz), "um")
        if ny is not None and nx is not None:
            _info_kv("Frame size", f"{ny} x {nx}", "px (Y x X)")
            if dx is not None and dy is not None:
                _info_kv("FOV", f"{_info_num(dy * ny)} x {_info_num(dx * nx)}", "um")

        zplanes = sizes.get("Z") or _info_md(md, "num_zplanes")

        click.secho("\nAcquisition", fg="cyan")
        _info_kv("Stack type", _info_md(md, "stack_type"))

        if is_isoview:
            # view_names are the VW labels (VW00, VW90, ...); `views` holds raw camera keys
            view_labels = (
                getattr(data, "view_names", None) or getattr(data, "views", None) or []
            )
            view_str = ", ".join(str(v) for v in view_labels) or _info_num(
                getattr(data, "num_views", None) or sizes.get("C")
            )
            if is_tiled:
                _info_kv("Tiles", _info_num(sizes.get("T")))
                _info_kv("Timepoints", "1")
            else:
                _info_kv("Timepoints", _info_num(sizes.get("T")))
            _info_kv("Z-planes", _info_num(zplanes))
            _info_kv("Views", view_str)
            _info_kv(
                "Color channels", _info_num(getattr(data, "num_color_channels", None))
            )
        else:
            timepoints = n_timepoints
            channels = sizes.get("C") or _info_md(md, "num_color_channels")
            mrois = _info_md(md, "num_mrois")

            avg = None
            if isinstance(md.get("si"), dict):
                from mbo_utilities.metadata.scanimage import get_log_average_factor

                avg = get_log_average_factor(md)

            _info_kv("Timepoints", _info_num(timepoints))
            _info_kv("Z-planes", _info_num(zplanes))
            _info_kv("Color channels", _info_num(channels))
            _info_kv("mROIs", _info_num(mrois))
            if avg and avg > 1:
                _info_kv("Averaging", f"{avg}x")
            if fs and timepoints:
                _info_kv("Duration", f"{timepoints / fs:.1f}", "s")

    if data.vmin is not None and data.vmax is not None:
        _info_kv(
            "Value range",
            f"[{_info_num(float(data.vmin))}, {_info_num(float(data.vmax))}]",
        )

    filenames = getattr(data, "filenames", None)
    if filenames:
        paths = [Path(f) for f in filenames]
        collide = len({p.name for p in paths}) < len(paths)
        click.secho(f"\nFiles ({len(filenames)})", fg="cyan")
        shown = paths if len(paths) <= 5 else paths[:3]
        for p in shown:
            click.echo(f"  - {f'{p.parent.name}/{p.name}' if collide else p.name}")
        if len(filenames) > 5:
            click.echo(f"  ... and {len(filenames) - 3} more")

    if is_isoview:
        click.secho("\nIsoview", fg="cyan")
        _info_kv("Pipeline stage", _info_md(md, "stack_type"))
        _info_kv("Layout", "tiled" if is_tiled else "timelapse")
    else:
        input_obj = Path(input_path)
        scan_root = input_obj if input_obj.is_dir() else input_obj.parent
        ops_dirs = _info_find_ops_dirs(scan_root)

        click.secho("\nResults", fg="cyan")
        if ops_dirs:
            total_rois = total_cells = seg_planes = 0
            have_seg = False
            for d in ops_dirs:
                n_rois, n_cells = _info_segmentation_counts(d)
                if n_rois is not None:
                    have_seg = True
                    seg_planes += 1
                    total_rois += n_rois
                    if n_cells is not None:
                        total_cells += n_cells
            have_reg = any((d / "data.bin").exists() for d in ops_dirs)
            have_raw = any((d / "data_raw.bin").exists() for d in ops_dirs)

            _info_kv("Suite2p", f"{len(ops_dirs)} folder(s)")
            if have_reg:
                _info_kv("Registration", "data.bin present")
            elif have_raw:
                _info_kv("Registration", "data_raw.bin only (not registered)")
            if have_seg:
                _info_kv(
                    "Segmentation",
                    f"{total_rois} ROIs ({total_cells} cells), {seg_planes} plane(s)",
                )
            else:
                _info_kv("Segmentation", "not run (no stat.npy)")
        else:
            click.echo("  none found")

    if show_all and md:
        click.secho("\nAll metadata", fg="cyan")
        for k in sorted(md, key=str):
            v = md[k]
            if isinstance(v, dict):
                click.echo(f"  {k}: {{...{len(v)} keys}}")
            elif isinstance(v, (list, tuple)) and len(v) > 8:
                click.echo(f"  {k}: [{len(v)} items]")
            else:
                s = str(v)
                if len(s) > 80:
                    s = s[:77] + "..."
                click.echo(f"  {k}: {s}")


@main.command("formats")
def list_formats():
    """List supported file formats."""
    click.echo("Supported input formats:")
    click.echo("  .tif, .tiff  - TIFF files (BigTIFF, OME-TIFF, ScanImage)")
    click.echo("  .zarr        - Zarr v3 arrays")
    click.echo("  .bin         - Suite2p binary format (with ops.npy)")
    click.echo("  .h5, .hdf5   - HDF5 files")
    click.echo("  .mesc        - Femtonics MESc (one array per measurement unit)")
    click.echo("  .npy         - NumPy arrays")
    click.echo("  .json        - Zarr array metadata (loads parent .zarr)")

    click.echo("\nSupported output formats:")
    click.echo("  .tiff        - Multi-page BigTIFF")
    click.echo("  .zarr        - Zarr v3 with optional OME-NGFF metadata")
    click.echo("  .bin         - Suite2p binary format")
    click.echo("  .h5          - HDF5 format")
    click.echo("  .npy         - NumPy array")


@main.command("scanphase")
@click.argument("input_path", required=False, type=click.Path())
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Output directory for results. Default: <input>_scanphase_analysis/",
)
@click.option(
    "-n",
    "--num-tifs",
    "num_tifs",
    type=int,
    default=None,
    help="If input is a folder, only use the first N tiff files.",
)
@click.option(
    "--format",
    "image_format",
    type=click.Choice(["png", "pdf", "svg", "tiff"]),
    default="png",
    help="Output image format.",
)
@click.option(
    "--show/--no-show",
    default=False,
    help="Display plots interactively after analysis.",
)
@click.option(
    "--patch",
    "patch_size",
    type=int,
    default=32,
    help="Spatial patch size in px for the distribution map. Default: 32.",
)
@click.option(
    "--docs",
    is_flag=True,
    default=False,
    help="Save output to docs/_images/scanphase/ for documentation.",
)
def scanphase(input_path, output_dir, num_tifs, image_format, show, docs, patch_size):
    r"""
    Scan-phase analysis for bidirectional scanning data.

    Analyzes phase offset to determine optimal correction parameters.

    \b
    OUTPUT:
      window_lengths.png        - offset vs frames averaged, with/without FFT
      spatial_distribution.png  - offset across FOV, with/without FFT
      scanphase_results.npz     - all numerical data

    \b
    Examples:
      mbo scanphase                          # open file dialog
      mbo scanphase /path/to/data.tiff       # analyze specific file
      mbo scanphase ./folder/ -n 5           # use first 5 tiffs in folder
      mbo scanphase data.tiff -o ./results/  # custom output directory
      mbo scanphase data.tiff --patch 16     # finer spatial map
      mbo scanphase data.tiff --show         # show plots interactively
    """
    from pathlib import Path

    from mbo_utilities import get_files
    from mbo_utilities.analysis.scanphase import run_scanphase_analysis

    try:
        if docs:
            repo_root = Path(__file__).parent.parent
            output_dir = str(repo_root / "docs" / "_images" / "scanphase")
            click.echo(f"--docs flag: saving to {output_dir}")

        actual_input = input_path
        if input_path is not None:
            input_path_obj = Path(input_path)
            if input_path_obj.is_dir() and num_tifs is not None:
                tiffs = get_files(input_path, str_contains=".tif", max_depth=1)
                if not tiffs:
                    click.secho(f"No tiff files found in {input_path}", fg="red")
                    raise click.Abort
                tiffs = tiffs[:num_tifs]
                click.echo(f"Using {len(tiffs)} tiff files from {input_path}")
                actual_input = tiffs

        if output_dir is not None:
            actual_output_dir = Path(output_dir)
        elif input_path is not None:
            input_path_obj = Path(input_path)
            actual_output_dir = (
                input_path_obj.parent / f"{input_path_obj.stem}_scanphase_analysis"
            )
        else:
            actual_output_dir = None

        results = run_scanphase_analysis(
            data_path=actual_input,
            output_dir=output_dir,
            image_format=image_format,
            show_plots=show,
            patch_size=patch_size,
        )

        if results is None:
            return
        summary = results.get_summary()
        meta = summary.get("metadata", {})

        click.echo("")
        click.secho("scan-phase analysis complete", fg="cyan", bold=True)
        click.echo("")
        click.echo(
            f"data: {meta.get('num_timepoints', meta.get('num_frames', 0))} timepoints, "
            f"{meta.get('num_rois', 1)} ROIs, "
            f"{meta.get('frame_shape', (0, 0))[1]}x{meta.get('frame_shape', (0, 0))[0]} px"
        )
        click.echo(f"analysis time: {meta.get('analysis_time', 0):.1f}s")
        click.echo(f"output: {actual_output_dir}")

        if "fft" in summary:
            stats = summary["fft"]
            click.echo("")
            click.secho("offset (FFT)", fg="yellow", bold=True)
            click.echo(f"  mean:   {stats.get('mean', 0):+.3f} px")
            click.echo(f"  median: {stats.get('median', 0):+.3f} px")
            click.echo(f"  std:    {stats.get('std', 0):.3f} px")
            click.echo(
                f"  range:  [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}] px"
            )

        if "int" in summary:
            stats = summary["int"]
            click.echo("")
            click.secho("offset (integer)", fg="yellow", bold=True)
            click.echo(f"  mean:   {stats.get('mean', 0):+.3f} px")
            click.echo(f"  std:    {stats.get('std', 0):.3f} px")

        click.echo("")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        raise click.Abort


@main.command("processes")
@click.option(
    "--kill-all",
    is_flag=True,
    help="Kill all tracked background processes.",
)
@click.option(
    "--kill",
    type=int,
    default=None,
    help="Kill a specific process by PID.",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Remove entries for finished processes.",
)
def processes(kill_all, kill, cleanup):
    r"""
    Manage background processes (suite2p, save operations, etc.).

    \b
    Examples:
      mbo processes                  # List all tracked processes
      mbo processes --cleanup        # Remove finished process entries
      mbo processes --kill 12345     # Kill specific process
      mbo processes --kill-all       # Kill all background processes
    """
    from mbo_utilities.gui.widgets.process_manager import get_process_manager

    pm = get_process_manager()

    if cleanup:
        count = pm.cleanup_finished()
        click.echo(f"Cleaned up {count} finished processes.")
        return

    if kill_all:
        running = pm.get_running()
        if not running:
            click.echo("No running processes to kill.")
            return
        count = pm.kill_all()
        click.secho(f"Killed {count} processes.", fg="yellow")
        return

    if kill is not None:
        if pm.kill(kill):
            click.secho(f"Killed process {kill}.", fg="yellow")
        else:
            click.secho(f"Process {kill} not found or could not be killed.", fg="red")
        return

    all_procs = pm.get_all()
    if not all_procs:
        click.echo("No tracked background processes.")
        return

    click.echo(f"\nTracked processes ({len(all_procs)}):")
    click.echo("-" * 60)

    for p in all_procs:
        alive = p.is_alive()
        status = (
            click.style("RUNNING", fg="green")
            if alive
            else click.style("FINISHED", fg="bright_black")
        )
        click.echo(f"  PID {p.pid:>6}  {status}  {p.description}")
        click.echo(f"           Started: {p.elapsed_str()}")
        if p.output_path:
            click.echo(f"           Output:  {p.output_path}")

    click.echo("-" * 60)
    running = [p for p in all_procs if p.is_alive()]
    click.echo(f"Running: {len(running)}, Finished: {len(all_procs) - len(running)}")

    if len(all_procs) > len(running):
        click.echo("\nTip: Run 'mbo processes --cleanup' to remove finished entries.")


@main.command("gpu")
@click.option(
    "--watch",
    type=float,
    default=None,
    help="Refresh every N seconds until Ctrl-C.",
)
@click.option(
    "--processes",
    "show_processes",
    is_flag=True,
    help="Also list per-process VRAM usage.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit render/compute GPU + devices + processes as JSON.",
)
def gpu(watch, show_processes, as_json):
    r"""
    Show which GPU renders (fastplotlib) and which computes (suite2p /
    cellpose / cupy), plus device memory.

    \b
    Examples:
      mbo gpu               # render GPU, compute GPU, device memory
      mbo gpu --processes   # also per-process VRAM
      mbo gpu --watch 2     # refresh every 2s
      mbo gpu --json        # machine-readable
    """
    from mbo_utilities import gpu as gpu_mod

    # reflect the GPU picked in the GUI's File > Options (persisted pref)
    gpu_mod.apply_persisted_compute_gpu()

    if as_json:
        import json as _json

        click.echo(
            _json.dumps(
                {
                    "render_gpu": gpu_mod.render_gpu(),
                    "compute_gpu": gpu_mod.compute_gpu(),
                    "devices": gpu_mod.gpu_devices(),
                    "processes": gpu_mod.gpu_processes() if show_processes else [],
                    "compute_disabled": gpu_mod.gpu_compute_disabled(),
                },
                indent=2,
            )
        )
        return

    if watch:
        try:
            while True:
                click.clear()
                click.echo(gpu_mod.format_gpu_report(show_processes=show_processes))
                click.echo("\n(Ctrl-C to stop)")
                time.sleep(watch)
        except KeyboardInterrupt:
            return
    else:
        click.echo(gpu_mod.format_gpu_report(show_processes=show_processes))


@main.command("init")
@click.argument("data_path", required=False, type=click.Path())
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Destination directory (overrides default location).",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing notebooks.",
)
@click.option(
    "--unit",
    default="",
    help="MESc measurement unit for the spine pipeline, e.g. MSession_0/MUnit_3.",
)
def init(data_path, output_dir, overwrite, unit):
    r"""
    Create starter notebooks: the user guides, and the ASAP7 spine pipeline for a .mesc.

    With DATA_PATH, notebooks go in <DATA_PATH>/../scripts and the data path
    is filled in. Without it, notebooks go in the current directory. The
    Studio's File > Open Notebook writes the same notebooks to the same place.

    \b
    Examples:
      mbo init                       # notebooks in current directory
      mbo init /data/raw             # notebooks in /data/scripts, path filled in
      mbo init /data/raw -o ./nb     # notebooks in ./nb
      mbo init /data/scan.mesc       # the guides and the spine pipeline
      mbo init /data/scan.mesc --unit MSession_0/MUnit_3
    """
    from mbo_utilities.notebook_templates import (
        notebook_path,
        scripts_dir,
        template_dir,
        templates_for,
        write_notebook,
    )

    if output_dir is not None:
        dest = Path(output_dir).expanduser()
    elif data_path is not None:
        dest = scripts_dir(data_path)
    else:
        dest = Path.cwd()

    written = []
    for template in templates_for(data_path):
        if not (template_dir() / template.filename).exists():
            click.secho(f"Missing: {template.filename}", fg="red")
            continue
        out = write_notebook(template, dest, data_path, unit, overwrite=overwrite)
        if out is None:
            click.secho(
                f"Exists: {notebook_path(template, dest, unit=unit)}  (--overwrite to replace)",
                fg="yellow",
            )
            continue
        written.append(out)
        click.secho(f"Created: {out}", fg="green")

    if not written:
        return
    if data_path:
        click.echo(f"Data path: {Path(data_path).expanduser().resolve().as_posix()}")
    click.echo(f"\nOpen with:\n  jupyter lab {dest}")


@main.command("download-models")
@click.option(
    "--force",
    is_flag=True,
    help="Re-download even if already present.",
)
def download_models(force):
    r"""
    Download/repair the cellpose model used by suite2p detection.

    \b
    Examples:
      mbo download-models           # download if missing or corrupt
      mbo download-models --force   # re-download
    """
    from mbo_utilities._cellpose_model import ensure_cellpose_model

    path = ensure_cellpose_model(force=force)
    if path:
        click.secho(f"cellpose model ready: {path}", fg="green")
    else:
        click.secho("cellpose model unavailable (pip install cellpose)", fg="red")
        raise click.Abort


@main.command("shortcut")
@click.option(
    "--name",
    default="The Studio",
    help="Shortcut name.",
)
def shortcut(name):
    r"""
    Create a desktop icon that opens the GUI.

    Windows: a .lnk that launches the viewer with no console window.
    Linux: a .desktop entry on the Desktop.

    \b
    Examples:
      mbo shortcut
      mbo shortcut --name "MBO"
    """
    from mbo_utilities._shortcut import create_desktop_shortcut

    try:
        path = create_desktop_shortcut(name=name)
    except Exception as e:
        raise click.ClickException(str(e))
    click.secho(f"Created: {path}", fg="green")


@main.command("curate")
@click.argument("path", type=click.Path(exists=True), required=False)
@click.option(
    "--serve",
    is_flag=True,
    default=False,
    help="Serve the dashboard to browsers over HTTP instead of opening a window.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="With --serve: interface to listen on (0.0.0.0 for the network).",
)
@click.option(
    "--port", type=int, default=60649, show_default=True, help="With --serve: port."
)
@click.option(
    "--channel",
    type=int,
    default=0,
    show_default=True,
    help="Channel averaged for a raw line scan's traces.",
)
def curate(path, serve, host, port, channel):
    """Vnoiser event curation of PATH: a PF folder the vnoiser pipeline
    wrote (or the experiment folder holding it), or a line-scan .mesc, with
    a PF folder beside it or raw (default: the last data path).

    Opens the desktop window, the one the viewer's Curate button opens
    (the vnoiser pipeline, or the MESc tab once a run is beside the file). With --serve the
    dashboard is rendered here and streamed to any browser that opens the
    printed URL: run it on the machine that holds the data and a GPU, and
    curate from a laptop. No login: keep --host on localhost and tunnel
    (ssh -L 60649:localhost:60649 server), or put a proxy in front.

    
      mbo curate X:/data/asako/stan112/stan112_expt12
      mbo curate /data/stan112/stan112_expt12 --serve
      mbo curate /data/stan112 --serve --host 0.0.0.0 --port 8080
    """
    if serve:
        from mbo_utilities.gui.curation_server import serve_curation

        serve_curation(path, channel=channel, host=host, port=port)
        return
    from mbo_utilities.gui.curation_viewer import open_curation_viewer

    open_curation_viewer(path, channel=channel)


@main.command("serve")
@click.argument("path", type=click.Path(exists=True), required=False)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to listen on: 'tailscale' for this machine's tailnet address, "
    "0.0.0.0 for every network.",
)
@click.option("--port", type=int, default=8800, show_default=True, help="Port.")
@click.option(
    "--max-sessions",
    type=int,
    default=1,
    show_default=True,
    help="Studio processes allowed at once.",
)
@click.option(
    "--idle-timeout",
    type=float,
    default=10.0,
    show_default=True,
    help="Minutes a session lives with no browser connected; 0 keeps it.",
)
@click.option(
    "--widget",
    type=click.Choice(["preview", "manualroi", "none"]),
    default="preview",
    show_default=True,
    help="Side widget of the session PATH opens.",
)
@click.option(
    "--quality",
    type=click.IntRange(1, 99),
    default=80,
    show_default=True,
    help="JPEG quality of frames while the view changes; a still view is sent "
    "lossless. 50 halves the bytes, for a slow or relayed link.",
)
@click.option(
    "--token",
    default=None,
    help="Token the link carries (default: a new random one each start).",
)
@click.option(
    "--no-token",
    is_flag=True,
    default=False,
    help="Serve without a token, for a localhost-only port behind a tunnel.",
)
def serve(
    path, host, port, max_sessions, idle_timeout, widget, quality, token, no_token
):
    r"""Serve The Studio to browsers from this machine.

    Each browser session runs the Studio in a process of its own on this
    machine's GPU and data, streamed to the browser over a websocket. The
    page at the printed link lists the sessions and opens a file or folder
    in a new one, in place of the desktop file dialog. With PATH a session
    on it starts right away.

    The link carries a token; anyone with it can read what you can, so keep
    the port on localhost or the tailnet.

    \b
      mbo serve --host tailscale
      mbo serve /data/stan112/stan112_expt12.mesc --host tailscale
      mbo serve --port 8800 --no-token     # then: ssh -L 8800:localhost:8800 server
    """
    import secrets
    import subprocess

    from mbo_utilities.gui.gateway import Gateway

    if host == "tailscale":
        try:
            out = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as e:
            raise click.ClickException(f"no tailnet address: {e}") from e
        host = out.split()[0]
    if no_token:
        token = None
    elif token is None:
        token = secrets.token_urlsafe(16)
    gateway = Gateway(
        host=host,
        port=port,
        max_sessions=max_sessions,
        idle_timeout=idle_timeout * 60,
        token=token,
        quality=quality,
    )
    click.echo(f"The Studio: {gateway.link()}")
    if path:
        session = gateway.open(path, widget)
        click.echo(f"  {Path(path).name}: {gateway.link(session.url)}")
    gateway.serve()


@main.command("spike-average")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--unit",
    default=None,
    help="The results unit, e.g. zplane01 or scan35 (default: the first).",
)
@click.option(
    "--roi", default=None, help="The ROI, by its results name (default: the first)."
)
@click.option(
    "--channel",
    type=int,
    default=None,
    help="The movie's channel, 0-based (default: the run's own).",
)
@click.option(
    "--trace",
    "kind",
    default=None,
    help="The trace spikes are found on: denoised, dff, raw... (default: the most processed).",
)
@click.option(
    "--spikes",
    "spikes_from",
    default=None,
    help="Where spikes come from: events, 'curated <mode>' or threshold "
    "(default: events, else a curation, else threshold).",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Peak height that counts as a spike, in the trace's units (default: 3 SD over its mean).",
)
@click.option(
    "--distance",
    type=int,
    default=5,
    show_default=True,
    help="Fewest frames between two thresholded spikes.",
)
@click.option(
    "--window",
    type=int,
    nargs=2,
    default=(10, 10),
    show_default=True,
    help="Frames averaged before and after each spike.",
)
def spike_average(
    path, unit, roi, channel, kind, spikes_from, threshold, distance, window
):
    """Spike-triggered averages of one ROI of a run: the movie it was measured
    on (and a MaskNMF run's compressed movie), its trace and every motion
    correction (masknmf's shifts, the AOD's RTMC), averaged around its spikes,
    in one window. ROI, trace and spikes are picked there; the threshold drags.

    PATH is any run with results: a MaskNMF run folder that demixed, a vnoiser
    or other results file, a PF folder, a suite2p plane dir.

    
      mbo spike-average D:/data/20261007T183740_masknmf_zplane03_tp00001-20000
      mbo spike-average session1.2026-09-21-14-30-22.vnoiser.zarr --unit scan35 --roi soma
    """
    from mbo_utilities.gui.spike_average_viewer import open_spike_average_viewer

    open_spike_average_viewer(
        path,
        unit,
        roi,
        channel,
        kind=kind,
        spikes=spikes_from,
        threshold=threshold,
        distance=distance,
        before=window[0],
        after=window[1],
    )


@main.command("vnoiser")
@click.argument("mesc_path", type=click.Path(exists=True))
@click.option(
    "--domains",
    "domains_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="domains.json (or an archive scanIDs_ROIs.pkl) naming which ROIs make each domain. "
    "Default: domains.json beside the file.",
)
@click.option(
    "--unit",
    "units",
    multiple=True,
    help="Unit(s) to process, in scan order (MUnit_35 or MSession_0/MUnit_35): line scans, "
    "chessboard or ribbon patches. Default: the domains file's scans, else every such unit.",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(file_okay=False),
    default=None,
    help="The PF folder to write. Default: <animal>/<expt>/PF for the archive layout "
    "(<expt>/<expt>/<expt>.mesc), else PF beside the file.",
)
@click.option(
    "--channel",
    type=int,
    default=0,
    show_default=True,
    help="Channel averaged per line.",
)
@click.option(
    "--convert",
    is_flag=True,
    default=False,
    help="Apply the file's linear conversion so zero means no photons. Off reproduces "
    "the archive, which worked on raw counts.",
)
@click.option(
    "--events",
    default=None,
    metavar="LO,HI,BP_SD,AMP_SD,DUR_MS",
    help="Peak detector thresholds: band-pass Hz, SD on the band-passed trace, SD on the "
    "trace, minimum duration ms. Default 2,400,3.5,4,5 (the archive's).",
)
@click.option(
    "--save-cwt",
    is_flag=True,
    default=False,
    help="Also write cwts.h5, the wavelet coefficients (about 20 bytes per sample per domain).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Replace an existing output of the same name.",
)
@click.option(
    "--pkl",
    "as_pkl",
    is_flag=True,
    default=False,
    help="Write the archive's PF folder of pickles instead of the results zarr; the curation "
    "window opens either.",
)
@click.option(
    "-p",
    "--planes",
    type=int,
    multiple=True,
    help="ROI to process (1-based; the unit's Z axis), repeat for several: -p 1 -p 3. Only these "
    "are read and every domain is cut down to them. Default: every ROI.",
)
@click.option(
    "--init",
    is_flag=True,
    default=False,
    help="Write a domains.json template beside the file (one domain per ROI) and exit.",
)
@click.option(
    "--recommended",
    is_flag=True,
    default=False,
    help="Settings for new recordings: the conversion offset applied and a robust (MAD) "
    "z-score. Without it the defaults are the archive's.",
)
@click.option(
    "--pixels",
    type=click.Choice(["all", "bright", "active", "correlated"]),
    default=None,
    help="Pixels of each ROI a domain given as a plain ROI list averages: every pixel "
    "(default), or the top --fraction by brightness, fast fluctuation or correlation "
    "with the ROI's mean. Domains that name their own pixels keep them.",
)
@click.option(
    "--fraction",
    type=float,
    default=0.3,
    show_default=True,
    help="With --pixels: the share of each ROI's pixels kept.",
)
@click.option(
    "--background",
    type=click.Choice(["none", "dim", "outside", "domain"]),
    default=None,
    help="Subtracted from each domain: the dimmest pixels of its ROIs per frame, the "
    "ROI pixels outside it, or the domain named by --background-domain. Default none.",
)
@click.option(
    "--background-domain",
    default="bg",
    show_default=True,
    help="With --background domain: the domain that is the background.",
)
@click.option(
    "--background-percentile",
    type=float,
    default=10.0,
    show_default=True,
    help="With --background dim: the percentile of the ROI's pixels per frame.",
)
@click.option(
    "--reference",
    type=click.Choice(["none", "ratio", "regress"]),
    default=None,
    help="A structural channel recorded beside the indicator (mScarlet beside ASAP7): "
    "divide by it, or remove the part of the trace it explains. Default none.",
)
@click.option(
    "--reference-channel",
    type=int,
    default=None,
    help="With --reference: its channel, 0-based. Default the other of two.",
)
@click.option(
    "--polarity",
    type=click.Choice(["flip_z", "invert_f", "none"]),
    default=None,
    help="Turning a dimming indicator's events upright: flip the z-score (vnoiser, the "
    "archive's default), invert the trace about its mean before dF/F (pml_utilities' "
    "Invert Deflection, so dF/F rises too), or neither (a brightening indicator).",
)
@click.option(
    "--dfof",
    "dfof_method",
    type=click.Choice(
        ["gaussian", "percentile", "maxmin", "exponential", "window", "none"]
    ),
    default=None,
    help="dF/F baseline: the archive's Gaussian (default), a rolling percentile or max-min "
    "over --dfof-window seconds, an exponential bleaching fit, the mean over --f0, or none.",
)
@click.option(
    "--dfof-window",
    type=float,
    default=None,
    help="Rolling baseline window, seconds (percentile, maxmin). Default 5.",
)
@click.option(
    "--f0",
    default=None,
    metavar="START,STOP",
    help="With --dfof window: the seconds F0 is the mean over. Default 0,1.",
)
@click.option(
    "--noise",
    type=click.Choice(["sd", "mad"]),
    default=None,
    help="What the z-score divides by: the SD of dF/F (the archive's) or the robust SD.",
)
@click.option(
    "--no-denoise",
    is_flag=True,
    default=False,
    help="Stop after dF/F and the z-score: no wavelet denoising, no peaks (zarr only).",
)
@click.option(
    "--timepoints",
    default=None,
    metavar="START:STOP",
    help="Frames to read, 1-based and inclusive, one contiguous window. Default every frame.",
)
def vnoiser(
    mesc_path,
    domains_path,
    units,
    out,
    channel,
    convert,
    events,
    save_cwt,
    overwrite,
    as_pkl,
    planes,
    init,
    recommended,
    pixels,
    fraction,
    background,
    background_domain,
    background_percentile,
    reference,
    reference_channel,
    polarity,
    dfof_method,
    dfof_window,
    f0,
    noise,
    no_denoise,
    timepoints,
):
    r"""The vnoiser pipeline on a .mesc: domain traces, dF/F and z-score,
    wavelet denoising, peaks, written as one
    `<input>.<timestamp>.vnoiser.zarr` beside the input that `mbo curate`
    opens.

    Each recording (a line scan, chessboard or ribbon scan, a single patch
    or a raster) is one scan; its MUnit number is the scan id. A domain is
    the pixels of one or more ROIs averaged into a trace: the lines of a
    soma or branch, a patch, the brightest or most correlated part of a
    patch, a mask drawn in the viewer. Without a domains file every ROI is
    its own domain; --init writes one to edit. The trace options (pixels,
    background, reference channel, polarity) and the dF/F options apply to
    every scan. Defaults are the archive's, written for its 1075 Hz and
    scaled to each scan's frame rate; at the archive's rate the traces
    match its PF folders. --recommended is the setting for new recordings.

    \b
      mbo vnoiser scan.mesc --recommended --unit MUnit_5
      mbo vnoiser scan.mesc --recommended --pixels correlated --fraction 0.2 --background outside
      mbo vnoiser scan.mesc --polarity invert_f --reference regress --dfof percentile --no-denoise
      mbo vnoiser stan112_expt12.mesc --init
      mbo vnoiser stan112_expt12.mesc --unit MUnit_35 --unit MUnit_38
      mbo vnoiser stan112_expt12.mesc --domains PF/scanIDs_ROIs.pkl -o PF_new
      mbo curate X:/data/asako/stan112/stan112_expt12
    """
    from mbo_utilities.gui._availability import HAS_VNOISER
    from mbo_utilities.install import VNOISER_HINT

    if not HAS_VNOISER:
        raise SystemExit(f"vnoiser is not installed: {VNOISER_HINT}")
    from mbo_utilities.vnoiser.params import VnoiserSettings
    from mbo_utilities.vnoiser.pipeline import (
        DOMAINS_FILE,
        read_domains,
        run_vnoiser_pipeline,
        scan_units,
        write_domains_template,
    )

    mesc_path = Path(mesc_path)
    if mesc_path.is_dir():
        from mbo_utilities.analysis.linescan import experiment_linescan_mesc

        found = experiment_linescan_mesc(mesc_path)
        if found is None:
            raise click.BadParameter(
                f"{mesc_path} is a folder with no <name>/<name>.mesc line scan in it",
                param_hint="MESC_PATH",
            )
        mesc_path = Path(found)
    if init:
        path = write_domains_template(mesc_path, units=list(units) or None)
        click.echo(
            f"wrote {path}; name the domains and group the ROIs, then run `mbo vnoiser` again"
        )
        return
    if domains_path is not None:
        spec = read_domains(domains_path)
    elif (mesc_path.parent / DOMAINS_FILE).exists():
        spec = read_domains(mesc_path.parent / DOMAINS_FILE)
        click.echo(f"domains: {mesc_path.parent / DOMAINS_FILE}")
    else:
        try:
            first = scan_units(mesc_path, list(units) or None)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--unit")
        if not first:
            raise click.BadParameter(
                f"no recording with traces in {mesc_path.name}", param_hint="MESC_PATH"
            )
        n_rois = int(first[0]["nrois"])
        spec = {
            "domains": {f"roi{r}": [r] for r in range(n_rois)},
            "scan_ids": [],
            "first_env": [],
        }
        click.echo(
            f"no {DOMAINS_FILE}: one domain per ROI ({n_rois}); --init writes one to edit"
        )
    settings = VnoiserSettings.recommended() if recommended else VnoiserSettings()
    if convert:
        settings.traces.convert = True
    if background is not None:
        settings.traces.background = background
    settings.traces.background_domain = background_domain
    settings.traces.background_percentile = background_percentile
    if reference is not None:
        settings.traces.reference = reference
    settings.traces.reference_channel = reference_channel
    if polarity is not None:
        settings.polarity = polarity
    if dfof_method is not None:
        settings.dfof.method = dfof_method
    if dfof_window is not None:
        settings.dfof.window_s = dfof_window
    if f0:
        try:
            settings.dfof.f0_start_s, settings.dfof.f0_stop_s = (
                float(v) for v in f0.split(",")
            )
        except ValueError:
            raise click.BadParameter("expected START,STOP seconds", param_hint="--f0")
    if noise is not None:
        settings.dfof.noise = noise
    if no_denoise:
        settings.runtime.denoise = False
    frames = None
    if timepoints:
        try:
            start, stop = (int(v) for v in timepoints.split(":"))
        except ValueError:
            raise click.BadParameter("expected START:STOP", param_hint="--timepoints")
        if not 1 <= start <= stop:
            raise click.BadParameter(
                "expected 1 <= START <= STOP", param_hint="--timepoints"
            )
        frames = (start - 1, stop)
    domains = spec["domains"]
    if pixels is not None:
        domains = {
            name: {"rois": list(value), "pixels": pixels, "fraction": fraction}
            if isinstance(value, list)
            else value
            for name, value in domains.items()
        }
    if events:
        try:
            lo, hi, bp_sd, amp_sd, dur = (float(v) for v in events.split(","))
        except ValueError:
            raise click.BadParameter(
                "expected LO,HI,BP_SD,AMP_SD,DUR_MS", param_hint="--events"
            )
        settings.events.bp_low, settings.events.bp_high = lo, hi
        settings.events.thres_bp_sd, settings.events.thres_amp_sd = bp_sd, amp_sd
        settings.events.duration_thres_ms = dur
    if as_pkl:
        if no_denoise:
            raise click.BadParameter(
                "a run without denoising writes the results zarr", param_hint="--pkl"
            )
        settings.runtime.output_format = "pkl"
    as_zarr = settings.runtime.output_format == "zarr"
    chosen = list(units) or [f"MUnit_{s}" for s in spec["scan_ids"]] or None
    # the runner narrates every step through the mbo logger; give its console lines a clock
    logger = log.get()
    for handler in logger.handlers:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        )
    try:
        paths = run_vnoiser_pipeline(
            mesc_path,
            domains=domains,
            units=chosen,
            first_env=spec["first_env"],
            out=out,
            channel=channel,
            frames=frames,
            planes=list(planes) or None,
            save_cwt=save_cwt,
            settings=settings,
            overwrite=overwrite,
            logger=logger,
            provenance={"settings": settings.to_dict()},
        )
    except (ValueError, KeyError, FileExistsError) as e:
        click.echo(f"error: {e}", err=True)
        raise click.Abort
    pf_dir = next(iter(paths.values())).parent
    click.echo(f"wrote {len(paths)} files to {pf_dir}")
    if as_zarr:
        click.echo(f"results: {next(p for p in paths.values() if p.suffix == '.zarr')}")
    else:
        click.echo(f"curate with: mbo curate {pf_dir.parent}")


@main.command("results")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="The .zarr to write. Default: <name>.<stamp>.<pipeline>.zarr inside PATH, named after PATH.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Replace an existing results file."
)
def results(path, out, overwrite):
    r"""Write a pipeline's output folder as one results zarr.

    PATH is a suite2p or masknmf output folder (one plane dir, or a folder
    of zplaneNN dirs) or the vnoiser pipeline's PF folder. The file holds
    every plane or scan as a group of (roi, t) traces, ROI membership,
    events and summary images; `mbo_utilities.results.Results.read` reads
    it back.

    \b
      mbo results run/zplane01_tp00001-01574
      mbo results stan112_expt12/PF -o stan112_expt12/stan112_expt12.vnoiser.zarr
    """
    from mbo_utilities.results import TRACES_PKL, Results, results_name

    path = Path(path)
    pf_dir = next((d for d in (path, path / "PF") if (d / TRACES_PKL).is_file()), None)
    if pf_dir is not None:
        found = Results.from_pf(pf_dir)
        source = found.source.get("mesc") or pf_dir
    else:
        try:
            found = Results.from_suite2p(path)
        except FileNotFoundError as e:
            raise click.BadParameter(str(e), param_hint="PATH")
        source = path
    target = Path(out) if out else path / results_name(source, pipeline=found.pipeline)
    try:
        written = found.write(target, overwrite=overwrite)
    except (FileExistsError, ValueError) as e:
        click.echo(f"error: {e}", err=True)
        raise click.Abort
    click.echo(f"wrote {len(found.units)} unit(s) to {written}")


@main.command("reg-denoise")
@click.argument("path", type=click.Path(exists=True))
@click.option("--unit", default=None, help="The .mesc unit, e.g. MSession_0/MUnit_3.")
@click.option(
    "--channel", type=int, default=1, show_default=True, help="Channel, 1-based."
)
@click.option(
    "--first-frame",
    type=int,
    default=200,
    show_default=True,
    help="0-based frame the run starts at; earlier frames are left out.",
)
@click.option(
    "--max-shift",
    type=int,
    default=40,
    show_default=True,
    help="Largest rigid shift allowed, px.",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(file_okay=False),
    default=None,
    help="Folder for results.hdf5. Default: <stem>[_<unit>].reg_denoise beside PATH.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Replace an existing results.hdf5."
)
def reg_denoise(path, unit, channel, first_frame, max_shift, out, overwrite):
    r"""Pre-registration denoising: compress, register to the compression, compress again.

    Writes one results.hdf5 holding the compression of the raw movie, the
    registration estimated on it and the compression of the registered
    movie. `mbo <folder>/results.hdf5` opens it in the Registration-Denoising
    Quality Control viewer.

    \b
      mbo reg-denoise scan.mesc --unit MSession_0/MUnit_3
    """
    from mbo_utilities.masknmf.params import RegDenoiseSettings
    from mbo_utilities.masknmf.reg_denoise import run_reg_denoise

    settings = RegDenoiseSettings(channel=channel, first_frame=first_frame)
    settings.registration.max_shifts = (max_shift, max_shift)
    try:
        written = run_reg_denoise(path, out, settings, unit=unit, overwrite=overwrite)
    except (FileExistsError, ValueError) as e:
        click.echo(f"error: {e}", err=True)
        raise click.Abort
    click.echo(f"wrote {written}\nview with: mbo {written}")


@main.command("asap7")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--unit",
    default=None,
    help="The .mesc unit, e.g. MUnit_10 or MSession_0/MUnit_10; left out, the units are listed.",
)
@click.option(
    "--channel", type=int, default=1, show_default=True, help="Channel, 1-based."
)
@click.option(
    "--first-frame",
    type=int,
    default=200,
    show_default=True,
    help="0-based frame the run starts at; earlier frames are left out.",
)
@click.option(
    "--max-shift",
    type=int,
    default=40,
    show_default=True,
    help="Largest rigid shift allowed, px.",
)
@click.option(
    "--detrend-window",
    type=float,
    default=0.3,
    show_default=True,
    help="Maximin spline detrend window, s.",
)
@click.option(
    "--mad-threshold",
    type=float,
    default=0.6,
    show_default=True,
    help="Superpixel MAD correlation threshold for the spine demixing.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    show_default=True,
)
@click.option(
    "-o",
    "--out",
    type=click.Path(file_okay=False),
    default=None,
    help="Folder the run folder goes in. Default: masknmf_asap7/<unit>_detrend beside PATH.",
)
def asap7(
    path,
    unit,
    channel,
    first_frame,
    max_shift,
    detrend_window,
    mad_threshold,
    device,
    out,
):
    r"""ASAP7 spine run: compress, register to the compression, compress and denoise
    with a detrender, demix spines and the dendrite.

    The raw movie is inverted (ASAP7 dims on depolarisation), compressed, and
    registered with shifts estimated on that compression; the registered raw
    movie is compressed and denoised with a maximin spline detrender and demixed
    by masknmf's glutamate spine pipeline. Defaults are
    asap7_spine_pipeline_detrend.ipynb's.

    \b
      mbo asap7 scan.mesc                      # list the units
      mbo asap7 scan.mesc --unit MUnit_10
      masknmf view <run folder>/results.glutamate.hdf5
    """
    from mbo_utilities.masknmf.params import Asap7Settings

    is_mesc = Path(path).suffix.lower() == ".mesc"
    if is_mesc and unit is None:
        _echo_mesc_units(path, None)
        click.echo("\nchoose one with --unit, e.g. --unit MUnit_10", err=True)
        raise click.Abort
    from mbo_utilities.masknmf.asap7 import run_asap7

    settings = Asap7Settings(
        channel=channel,
        first_frame=first_frame,
        detrend_window_seconds=detrend_window,
        mad_correlation_threshold=mad_threshold,
    )
    settings.registration.max_shifts = (max_shift, max_shift)
    settings.runtime.device = device
    try:
        run = run_asap7(path, unit=_parse_unit(unit), save_path=out, settings=settings)
    except ValueError as e:
        click.echo(f"error: {e}", err=True)
        raise click.Abort
    click.echo(f"wrote {run}\nview with: masknmf view {run / 'results.glutamate.hdf5'}")


if __name__ == "__main__":
    main()
