"""The results zarr of a masknmf run: footprints and calibrated traces as one pixel unit per plane.

``F`` is calibrated back to movie units from the PMD standardisation rather
than being a raw pixel average, and ``dff`` is dF/F against masknmf's static
baseline rather than a z-score or a rolling percentile. See ``roi_calibration``.
"""

from pathlib import Path

import numpy as np

from mbo_utilities import log

logger = log.get("masknmf.outputs")


def split_sparse_footprints(
    indices: np.ndarray, values: np.ndarray, n_rois: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split COO (pixel, roi) indices/values into per-ROI (pixel_idx, lam).

    ``indices`` is (2, nnz) with row 0 = flat pixel index (C-order over H,W)
    and row 1 = roi index; ``values`` is (nnz,).
    """
    order = np.argsort(indices[1], kind="stable")
    pix = indices[0][order]
    roi = indices[1][order]
    lam = values[order]
    bounds = np.searchsorted(roi, np.arange(n_rois + 1))
    return [
        (pix[bounds[k] : bounds[k + 1]], lam[bounds[k] : bounds[k + 1]])
        for k in range(n_rois)
    ]


def roi_calibration(
    footprints: list[tuple[np.ndarray, np.ndarray]],
    *,
    var_img: np.ndarray | None = None,
    mean_img: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROI ``(gain, F0)`` that undo the PMD standardisation.

    masknmf demixes ``(movie - mean_img) / var_img`` — and ``var_img`` is the
    per-pixel noise *std*, not a variance — so ``c`` is in noise-SD units.
    Worse, HALS never renormalises ``a`` or ``c``, so only the product
    ``a_k c_k`` is identified: the amplitude split between the two factors is
    whatever initialisation handed the solver. Raw ``c`` amplitudes are
    therefore not comparable across ROIs and carry no physical unit.

    Calibration fixes that gauge by moving the amplitude into ``a``'s physical
    units and reading F0 off masknmf's own static baseline ``b`` (the term the
    demixer already factored out of the traces)::

        gain_k = mean_support(a_k * var_img)
        F0_k   = mean_support(b * var_img + mean_img)
        F_k    = c_k * gain_k + F0_k              # movie units
        dFF_k  = c_k * gain_k / F0_k              # unitless

    This is masknmf's own ``produce_baseline_corrected_outputs`` convention:
    an *unweighted* mean over each footprint's support, not a lam-weighted
    one. Keeping it identical means traces here and traces from a direct
    masknmf script agree numerically.

    Missing inputs degrade gracefully: no ``var_img`` leaves ``gain`` as the
    mean lam (standardised units), and no ``mean_img``/``baseline`` leaves
    ``F0`` at 0, which callers must treat as "not calibrated".
    """
    n = len(footprints)
    gain = np.ones(n, dtype=np.float32)
    f0 = np.zeros(n, dtype=np.float32)

    var = None if var_img is None else np.asarray(var_img, dtype=np.float64).ravel()
    mean = None if mean_img is None else np.asarray(mean_img, dtype=np.float64).ravel()
    base = None if baseline is None else np.asarray(baseline, dtype=np.float64).ravel()

    for k, (pix, lam) in enumerate(footprints):
        if pix.size == 0:
            continue
        idx = pix.astype(np.int64)
        scale = lam.astype(np.float64)
        if var is not None:
            scale = scale * var[idx]
        gain[k] = float(scale.mean())

        level = np.zeros(idx.size, dtype=np.float64)
        if base is not None:
            level += base[idx] * (var[idx] if var is not None else 1.0)
        if mean is not None:
            level += mean[idx]
        f0[k] = float(level.mean())

    return gain, f0


def calibrated_traces(
    c: np.ndarray,
    gain: np.ndarray,
    f0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(F, dff)`` from demixed ``c`` (T, K) and a :func:`roi_calibration`.

    ``F`` is (K, T) in movie units; ``dff`` is (K, T) in percent, zeroed for
    any ROI whose F0 is non-positive (uncalibrated, or a footprint sitting on
    dead pixels) so a bogus baseline never manufactures a huge transient.
    """
    amp = np.ascontiguousarray(c.T, dtype=np.float32) * gain[:, None]
    F = amp + f0[:, None]
    dff = np.zeros_like(amp)
    ok = f0 > 0
    if ok.any():
        dff[ok] = amp[ok] / f0[ok, None] * 100.0
    return F.astype(np.float32), dff.astype(np.float32)


def _to_np(x) -> np.ndarray:
    """A torch tensor or array-like as numpy, on the host."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _movie_images(movie) -> dict[str, np.ndarray]:
    """``mean`` and ``max`` of up to 500 evenly sampled frames of a ``(T, Y, X)`` movie."""
    nframes = int(movie.shape[0])
    idx = np.unique(np.linspace(0, nframes - 1, min(nframes, 500)).astype(int))
    frames = _to_np(movie[idx.tolist()]).astype(np.float32)
    return {"mean": frames.mean(axis=0), "max": frames.max(axis=0)}


def results_unit(results, pmd, plane: int, run: Path, fs: float | None, images: dict):
    """
    The demixing of one plane as a pixel unit of a results file, with calibrated ``raw`` and ``dff`` traces.

    Parameters
    ----------
    results : masknmf.DemixingResults
        The plane's demixing.
    pmd : masknmf.CompressionArray
        The compression it demixed; its standardisation images calibrate the traces.
    plane : int
        1-based z-plane.
    run : Path
        The masknmf run folder, recorded as the unit's ``plane_dir``.
    fs : float, optional
        Frame rate in Hz.
    images : dict
        Summary images of the plane, by name.

    Returns
    -------
    tuple[ResultUnit, int]
        The unit and how many of its ROIs have a positive F0.
    """
    from mbo_utilities.results import ResultUnit, unit_name

    coo = results.spatial_demixed.coalesce()
    indices, values = _to_np(coo.indices()), _to_np(coo.values())
    c = np.asarray(results.signals_array.export_temporal_demixed(), dtype=np.float32)
    n_rois = int(c.shape[1])
    footprints = split_sparse_footprints(indices, values, n_rois)
    gain, f0 = roi_calibration(
        footprints,
        var_img=_to_np(pmd.noise_variance_image),
        mean_img=_to_np(pmd.mean_image),
        baseline=_to_np(results.static_baseline),
    )
    F, dff = calibrated_traces(c, gain, f0)
    unit = ResultUnit(
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
    )
    return unit, int((f0 > 0).sum())


def write_results(run: Path, movie, plane: int, record: dict, config: dict, device: str = "cpu") -> Path | None:
    """
    Write the results zarr of one plane's masknmf run beside its ``results.hdf5``.

    Parameters
    ----------
    run : Path
        The masknmf run folder.
    movie : PlaneMovie
        The ``(T, Y, X)`` view the run read; its frames give the mean and max images.
    plane : int
        1-based z-plane.
    record : dict
        The recording as ``config.json`` holds it under ``inputs["data"]``.
    config : dict
        The pipeline config the run took, ``{"pipeline", "configs"}``.
    device : str
        Where masknmf loads the results.

    Returns
    -------
    Path or None
        The results zarr, or None when the run stopped before demixing.
    """
    import masknmf

    from mbo_utilities.results import Results, results_name

    path_results = Path(run) / "results.hdf5"
    groups = masknmf.io.stage_groups(str(path_results))
    if masknmf.io.group_name_demixing() not in groups:
        return None
    results = masknmf.DemixingResults.from_hdf5(str(path_results), device=device)
    pmd = masknmf.CompressionArray.from_hdf5(str(path_results))
    images = _movie_images(movie)
    correlation = _to_np(results.standard_correlation_images[:])
    images["corr"] = correlation.max(axis=0) if correlation.ndim == 3 else correlation
    fs = record.get("fs")
    unit, n_calibrated = results_unit(results, pmd, plane, Path(run), fs, images)
    if n_calibrated < unit.n_rois:
        logger.warning(
            f"plane {plane}: {unit.n_rois - n_calibrated}/{unit.n_rois} ROIs have no positive F0; "
            "their dF/F is written as zeros"
        )
    source = record.get("path")
    path_zarr = Path(run) / results_name(source or Path(run).name, pipeline="masknmf")
    Results(
        pipeline="masknmf",
        units={unit.name: unit},
        source={"path": source, "planes": [plane], "reader_kwargs": record.get("reader_kwargs")},
        settings=config,
        metadata={"fs": fs} if fs else {},
    ).write(path_zarr)
    logger.info(f"plane {plane}: {unit.n_rois} ROIs -> {path_zarr.name}")
    return path_zarr
