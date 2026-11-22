"""Run masknmf's own pipelines on any array mbo_utilities opens, one z-plane at a time.

pml_utilities prepares the data: the plane, channel and frames picked, the
read features applied, squeezed to the ``(T, Y, X)`` view masknmf takes
(``plane_view``). masknmf does the rest, configured by the json its Parameters
popup writes (``masknmf.visualization.imgui.parameters.Parameters.config``).
Each plane gets masknmf's own run folder under ``<save_path>/zplaneNN/``, and
a results zarr beside its ``results.hdf5`` for the viewers and ``mbo curate``.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

from mbo_utilities import log
from mbo_utilities.arrays.plane import PlaneMovie

__all__ = ["pipeline_class", "plane_view", "run_planes"]

PIPELINE_DEFAULT = "two-photon-calcium"


def pipeline_class(config: dict) -> type:
    """
    The masknmf pipeline class a config names.

    Parameters
    ----------
    config : dict
        ``{"pipeline": class name, "configs": {...}}``, as ``Parameters.config()`` returns it.

    Returns
    -------
    type
        A ``masknmf.pipelines._base.BasePipeline`` subclass.

    Raises
    ------
    ValueError
        If no masknmf pipeline has that class name.
    """
    from masknmf.pipelines import scraper

    by_name = {cls.__name__: cls for cls in scraper.pipeline_registry().values()}
    if config.get("pipeline") not in by_name:
        raise ValueError(f"{config.get('pipeline')!r} is not a masknmf pipeline; one of {sorted(by_name)}")
    return by_name[config["pipeline"]]


def plane_view(
    arr,
    plane: int,
    channel: int | None = None,
    frame_indices: list[int] | None = None,
    read_features: dict | None = None,
    metadata: dict | None = None,
) -> tuple[PlaneMovie, dict, bool]:
    """
    The ``(T, Y, X)`` view of one z-plane that masknmf reads, and the record of it ``config.json`` keeps.

    Parameters
    ----------
    arr : LazyArray
        A recording as ``mbo_utilities.imread`` opens it.
    plane : int
        1-based z-plane.
    channel : int, optional
        1-based channel; None is the first.
    frame_indices : list[int], optional
        0-based frames to read; None reads every frame.
    read_features : dict, optional
        Read features to apply (``fix_phase``, ``use_fft``, ``frame_average``, ``invert_deflection``).
    metadata : dict, optional
        Overrides of the recording's metadata; ``fs`` is read from here first.

    Returns
    -------
    tuple[PlaneMovie, dict, bool]
        The view, its record (path, reader kwargs, read features, plane, z, c, frames, fs), and whether the
        recording was inverted, in which case the view is the uninverted source.
    """
    from mbo_utilities.arrays._inverted_view import InvertedDeflectionView
    from mbo_utilities.arrays.features import apply_read_features
    from mbo_utilities.arrays.features._slicing import index_window
    from mbo_utilities.metadata import get_param
    from mbo_utilities.reader import source_reader_kwargs

    source = getattr(arr, "source_path", None)
    reader_kwargs = source_reader_kwargs(arr)
    read_features = {k: v for k, v in (read_features or {}).items() if v is not None}
    arr, _ = apply_read_features(arr, read_features)
    z = plane - 1 if arr.nz > 1 else 0
    c = int(channel) - 1 if channel is not None else 0
    inverted = isinstance(arr, InvertedDeflectionView)
    ## NOTE: 2 * mean - x holds a still mean against a moving frame, so masknmf registers the source and inverts after
    view = PlaneMovie(arr.source if inverted else arr, z=z, c=c).select(frame_indices)
    window = index_window(frame_indices)
    fs = get_param(dict(metadata or {}), "fs") or get_param(dict(getattr(arr, "metadata", None) or {}), "fs")
    fs = float(fs) / (window[2] if window else 1) if fs else None
    record = {
        "path": None if source is None else str(source),
        "name": None if source is None else Path(source).name,
        "reader_kwargs": reader_kwargs,
        "read_features": read_features,
        "plane": plane,
        "z": z,
        "c": c,
        "frames": None if window is None else list(window),
        "tp_indices": None if frame_indices is None or window is not None else [int(t) for t in frame_indices],
        "fs": fs,
    }
    return view, record, inverted


def run_planes(
    input_data,
    save_path,
    config: dict,
    runtime: dict | None = None,
    run_args: dict | None = None,
    planes: list[int] | None = None,
    frame_indices: list[int] | None = None,
    channel: int | None = None,
    read_features: dict | None = None,
    metadata: dict | None = None,
    logger=None,
    progress_callback=None,
) -> list[Path]:
    """
    Run one masknmf pipeline on each selected z-plane, each into its own run folder.

    Parameters
    ----------
    input_data : LazyArray or str
        A recording, or a path ``mbo_utilities.imread`` opens.
    save_path : str or Path
        Each plane's run folder goes in ``<save_path>/zplaneNN/``, or ``<save_path>/roiNN/zplaneNN/`` per ROI when the
        recording's ROI selection splits or picks ROIs.
    config : dict
        ``Parameters.config()``: the pipeline and its stage configs that differ from the defaults.
    runtime : dict, optional
        ``Parameters.runtime()``: device, frame batch size and the like.
    run_args : dict, optional
        ``Parameters.run_args()``: stop_after, indicator_polarity and the like; ``frame_rate`` there wins
        over the recording's.
    planes : list[int], optional
        1-based z-planes; None runs every plane.
    frame_indices, channel, read_features, metadata
        As ``plane_view`` takes them.
    logger : logging.Logger, optional
        Where progress is logged.
    progress_callback : callable, optional
        Called as ``progress_callback(plane=i, total_planes=n, message=...)`` before each plane.

    Returns
    -------
    list[Path]
        The run folders, ROI by ROI and in plane order within each.

    Raises
    ------
    ValueError
        If the recording has no frame rate and ``run_args`` gives none.
    """
    from mbo_utilities import imread
    from mbo_utilities.masknmf.outputs import write_results

    logger = logger or log.get("masknmf.pipeline")
    cls = pipeline_class(config)
    arr = input_data if hasattr(input_data, "shape") else imread(input_data)
    if planes is None:
        planes = list(range(1, arr.nz + 1))
    takes = inspect.signature(cls.run).parameters
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    filepath_config = save_path / "masknmf_config.json"
    filepath_config.write_text(json.dumps(config, indent=2))
    rois = list(arr.iter_rois()) if hasattr(arr, "iter_rois") else [None]
    jobs = [(roi, plane) for roi in rois for plane in planes]
    runs = []
    for i, (roi, plane) in enumerate(jobs):
        roi_arr, roi_path = arr, save_path
        if roi is not None:
            roi_arr = copy.copy(arr)
            roi_arr.roi = roi
            roi_path = save_path / f"roi{roi:02d}"
        if progress_callback is not None:
            progress_callback(plane=i, total_planes=len(jobs), message=f"plane {plane}" + (f", ROI {roi}" if roi else ""))
        view, record, inverted = plane_view(
            roi_arr,
            plane=plane,
            channel=channel,
            frame_indices=frame_indices,
            read_features=read_features,
            metadata=metadata,
        )
        kwargs = {"frame_rate": record["fs"], **(run_args or {})}
        if kwargs["frame_rate"] is None:
            raise ValueError("the recording has no frame rate; set frame_rate in the masknmf parameters")
        if inverted and "indicator_polarity" in takes:
            kwargs.setdefault("indicator_polarity", "negative")
        pipeline = cls.from_config(filepath_config, output_folder=roi_path / f"zplane{plane:02d}", **(runtime or {}))
        pipeline.inputs = {"data": record}
        logger.info(f"masknmf {cls.__name__}: plane {plane} of {len(planes)}" + (f", ROI {roi}" if roi else ""))
        run = Path(pipeline.run(view, **{k: v for k, v in kwargs.items() if k in takes}))
        run = run if run.is_dir() else run.parent
        write_results(run, view, plane, record, config, device=str(pipeline.torch_device))
        runs.append(run)
    return runs
