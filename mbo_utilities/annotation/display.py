"""How a pipeline's traces are shown (AGENTS.md §7.6, Trace display).

One :class:`TraceProfile` per pipeline says which kinds its rows can show
(``DISPLAY_KINDS``: the trace kinds of the results zarr, plus ``SUBTRACTED``,
the raw trace minus its neuropil, for a pipeline that measured a real
``Fneu``), which one to show first, how a dF/F and a z-score are computed
from raw when the row carries none (:class:`DffSettings`), and the unit its
raw traces are in. The Traces panel and
the trace table read every row through :func:`display_trace`; nothing else
decides what a row's numbers mean. A pipeline outside this package registers its profile with
:func:`register_trace_profile` when it registers its ``PipelineInfo``;
an engine with no profile gets ``DEFAULT_TRACE_PROFILE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scipy.ndimage import gaussian_filter1d

from mbo_utilities.analysis.dff import (
    NOISE_ESTIMATES,
    baseline,
    dfof_maxmin,
    dfof_percentile,
    robust_sd,
)

__all__ = [
    "DEFAULT_TRACE_PROFILE",
    "DFF_METHODS",
    "DISPLAY_KINDS",
    "DffSettings",
    "SUBTRACTED",
    "TRACE_PROFILES",
    "TraceProfile",
    "available_kinds",
    "computed_dff",
    "computed_zscore",
    "deflect",
    "display_trace",
    "displayed_kind",
    "register_trace_profile",
    "trace_profile",
    "y_label",
]

# the one kind no file stores: the raw trace minus its neuropil, computed here
SUBTRACTED = "raw - neuropil"

# display kind -> y axis label, in selector order: the names of
# results.TRACE_KINDS and SUBTRACTED. y_label adds the profile's unit (and
# coefficient) to raw, neuropil and SUBTRACTED
DISPLAY_KINDS: dict[str, str] = {
    "dff": "dF/F (%)",
    "denoised": "denoised",
    "zscore": "z-score",
    "raw": "F",
    "neuropil": "Fneu",
    SUBTRACTED: "F - Fneu",
    "spikes": "spikes (a.u.)",
}

# what was measured from the pixels, in the profile's unit
MEASURED = ("raw", "neuropil", SUBTRACTED)

# kinds the panel computes from what was measured when a row carries none
COMPUTED = ("dff", "zscore")

# how a dF/F computed here finds F0 (DffSettings.method) -> its name in the
# panel. Every one but the static percentile is sized in seconds and needs
# the row's fs; a row without one gets the static percentile
DFF_METHODS: dict[str, str] = {
    "maxmin": "rolling max-min",
    "rolling": "rolling percentile",
    "gaussian": "Gaussian low-pass",
    "exponential": "bleaching fit",
    "window": "fixed window (MESc F0)",
    "percentile": "static percentile",
}


@dataclass
class DffSettings:
    """How a dF/F and a z-score are computed from a measured trace.

    ``method`` (:data:`DFF_METHODS`) finds F0: ``maxmin`` smooths by
    ``sigma_s`` and takes suite2p's rolling max of the rolling min over
    ``window_s``; ``rolling`` the rolling ``percentile`` over ``window_s``;
    ``gaussian`` a Gaussian low-pass of ``gaussian_s`` seconds (the voltage
    archive's, 1500 samples at its 1075 Hz); ``exponential`` a bleaching fit;
    ``window`` the mean from ``f0_start_s`` to ``f0_stop_s`` (MESc's F0);
    ``percentile`` the row's static ``percentile``. The z-score is dF/F
    minus a Gaussian of ``slow_s`` seconds (0 keeps it), over its SD or
    robust SD (``noise``, :data:`~mbo_utilities.analysis.dff.NOISE_ESTIMATES`).
    """

    method: str = "maxmin"
    window_s: float = 5.0
    sigma_s: float = 0.05
    percentile: float = 20.0
    gaussian_s: float = 1.4
    f0_start_s: float = 0.0
    f0_stop_s: float = 1.0
    noise: str = "sd"
    slow_s: float = 5.0

    def __post_init__(self):
        if self.method not in DFF_METHODS:
            raise ValueError(
                f"dF/F method must be one of {tuple(DFF_METHODS)}, got {self.method!r}"
            )
        if self.noise not in NOISE_ESTIMATES:
            raise ValueError(
                f"noise must be one of {NOISE_ESTIMATES}, got {self.noise!r}"
            )


@dataclass(frozen=True)
class TraceProfile:
    """What one pipeline's rows can show and how. A pipeline that measured a
    real neuropil lists ``neuropil`` and ``SUBTRACTED`` among its ``kinds``;
    ``neuropil_coeff`` is what it subtracts with, ``unit`` what its raw
    traces are in.
    """

    pipeline: str
    kinds: tuple[str, ...]
    default: str
    neuropil_coeff: float = 0.7
    dff_percent: bool = True
    unit: str = "a.u."
    dff: DffSettings = field(default_factory=DffSettings)


DEFAULT_TRACE_PROFILE = TraceProfile(
    pipeline="", kinds=("dff", "zscore", "raw"), default="dff"
)

TRACE_PROFILES: dict[str, TraceProfile] = {
    # suite2p: F, Fneu, the deconvolved spks and lbm_suite2p_python's
    # norm_traces, a dF/F as a fraction or (norm_method) a z-score. A dF/F
    # computed here is its plot: F - 0.7 Fneu over a static 20th percentile
    "suite2p": TraceProfile(
        pipeline="suite2p",
        kinds=("dff", "zscore", "raw", "neuropil", SUBTRACTED, "spikes"),
        default="dff",
        dff_percent=False,
        dff=DffSettings(method="percentile"),
    ),
    # masknmf: norm_traces in percent; its Fneu is zeros, so nothing to subtract
    "masknmf": TraceProfile(
        pipeline="masknmf",
        kinds=("dff", "zscore", "raw"),
        default="dff",
        dff=DffSettings(method="percentile"),
    ),
    # vnoiser: the curated (denoised) trace first, then the pipeline's own
    # dfof_raw (a fraction), the z-score and the lines' raw means in counts
    "vnoiser": TraceProfile(
        pipeline="vnoiser",
        kinds=("denoised", "dff", "zscore", "raw"),
        default="denoised",
        dff_percent=False,
        unit="counts",
    ),
    # the ROI tool's mean engine and a MESc unit's line and patch means: a
    # mask mean (plus a neuropil ring for a drawn ROI), no pipeline; dF/F
    # over the rolling baseline sized in seconds
    "mean": TraceProfile(
        pipeline="mean",
        kinds=("dff", "zscore", "raw", "neuropil", SUBTRACTED),
        default="dff",
    ),
}


def register_trace_profile(profile: TraceProfile) -> None:
    """Make ``profile`` the one rows with ``engine == profile.pipeline`` use."""
    TRACE_PROFILES[profile.pipeline] = profile


def trace_profile(engine: str) -> TraceProfile:
    """The profile for an engine / results pipeline name, else the default."""
    return TRACE_PROFILES.get(str(engine), DEFAULT_TRACE_PROFILE)


def available_kinds(trace) -> tuple[str, ...]:
    """The kinds ``trace`` can show, in its profile's order: the arrays it
    carries, ``dff`` and ``zscore`` whenever it has a raw trace to compute
    them from, and ``SUBTRACTED`` whenever it has both a raw trace and a
    neuropil.
    """
    profile = trace_profile(trace.engine)
    out = []
    for kind in profile.kinds:
        if kind == SUBTRACTED:
            has = trace.F is not None and trace.Fneu is not None
        else:
            has = trace.array(kind) is not None or (
                kind in COMPUTED and trace.F is not None
            )
        if has:
            out.append(kind)
    return tuple(out)


def displayed_kind(trace, kind: str | None = None) -> str | None:
    """The kind :func:`display_trace` shows for ``kind``: ``kind`` when the
    row has it, else the profile's default, else the first it has.
    """
    kinds = available_kinds(trace)
    if not kinds:
        return None
    if kind in kinds:
        return kind
    default = trace_profile(trace.engine).default
    return default if default in kinds else kinds[0]


def deflect(y: np.ndarray, subtract: bool = False, invert: bool = False) -> np.ndarray:
    """``y`` transformed about its own mean over time ``m`` the way the viewer
    transforms each pixel: ``y - m`` (subtract), ``2m - y`` (invert) or
    ``m - y`` (both). Always a new float32 array.
    """
    out = np.array(y, dtype=np.float32)
    if not (subtract or invert) or out.size == 0:
        return out
    m = np.float32(np.nanmean(out))
    out -= m
    if invert:
        np.negative(out, out=out)
    if not subtract:
        out += m
    return out


def computed_dff(f: np.ndarray, fs: float | None, settings: DffSettings) -> np.ndarray:
    """dF/F of one measured trace over ``settings``' baseline, as a fraction;
    a baseline at or below zero gives zero there. A method sized in seconds
    takes the static percentile when ``fs`` is unknown.
    """
    f = np.asarray(f, np.float32)[None, :]
    method = settings.method
    if method == "percentile" or not fs:
        return dfof_percentile(f, settings.percentile)[0]
    if method == "maxmin":
        return dfof_maxmin(f, fs, settings.window_s, settings.sigma_s)[0]
    f0 = baseline(
        f,
        fs,
        "percentile" if method == "rolling" else method,
        sigma=max(0.5, settings.gaussian_s * float(fs)),
        window_s=settings.window_s,
        percentile=settings.percentile,
        f0_window_s=(settings.f0_start_s, settings.f0_stop_s),
    )
    out = np.zeros(f.shape, np.float64)
    np.divide(f - f0, f0, out=out, where=f0 > 0)
    return out[0].astype(np.float32)


def computed_zscore(
    dff: np.ndarray, fs: float | None, settings: DffSettings
) -> np.ndarray:
    """``(dff - slow) / noise``: a dF/F minus its Gaussian of ``slow_s``
    seconds (its mean when that is 0 or ``fs`` unknown), over the SD or the
    robust SD of what is left.
    """
    d = np.asarray(dff, np.float64)[None, :]
    if settings.slow_s > 0 and fs:
        residual = d - gaussian_filter1d(d, sigma=settings.slow_s * float(fs), axis=1)
    else:
        residual = d - d.mean()
    scale = residual.std() if settings.noise == "sd" else float(robust_sd(residual)[0, 0])
    if not scale > 0:
        return np.zeros(d.shape[1], np.float32)
    return (residual[0] / scale).astype(np.float32)


def _measured(trace, kind: str) -> np.ndarray:
    """The row's ``raw``, ``neuropil`` or ``SUBTRACTED`` trace as float32."""
    if kind == "neuropil":
        return np.asarray(trace.Fneu, np.float32)
    f = np.asarray(trace.F, np.float32)
    if kind == SUBTRACTED:
        coeff = trace_profile(trace.engine).neuropil_coeff
        f = f - coeff * np.asarray(trace.Fneu, np.float32)
    return f


def display_trace(
    trace,
    kind: str | None = None,
    settings: DffSettings | None = None,
    subtract: bool = False,
    invert: bool = False,
    flip: bool = False,
) -> np.ndarray | None:
    """The row's trace as the panel plots it, in the kind
    :func:`displayed_kind` picks, or None when the row carries nothing.

    ``raw`` is the raw trace alone, ``neuropil`` its neuropil alone and
    ``SUBTRACTED`` the raw trace minus the profile's ``neuropil_coeff`` times
    the neuropil. ``dff`` is the row's own dF/F when it has one (scaled to
    percent), else one computed with ``settings`` (the profile's own when
    None, :func:`computed_dff`) from the ``SUBTRACTED`` trace when the row
    has one, the raw trace otherwise; ``zscore`` likewise the row's own, else
    :func:`computed_zscore` of that dF/F. Every other kind is the array the
    row carries under that name.

    ``subtract`` and ``invert`` are the viewer's Mean Subtraction and Invert
    Deflection, applied with :func:`deflect` to what was measured from the
    pixels: the ``MEASURED`` kinds, and the trace a ``dff`` is computed from
    here (invert only; a dF/F is already relative to its baseline). A
    pipeline's own kinds are shown as it wrote them, sign included.

    ``flip`` turns whatever is shown over, the panel's own switch: a
    measured trace (and one a dF/F is computed from) is inverted about its
    mean, undoing the viewer's inversion when that is on; any other kind is
    negated.
    """
    shown = displayed_kind(trace, kind)
    if shown is None:
        return None
    if shown in MEASURED:
        return deflect(_measured(trace, shown), subtract, invert != flip)
    sign = np.float32(-1.0 if flip else 1.0)
    own = trace.array(shown)
    profile = trace_profile(trace.engine)
    if own is not None:
        own = np.asarray(own, np.float32)
        if shown == "dff" and not profile.dff_percent:
            own = own * np.float32(100.0)
        return sign * own
    base = SUBTRACTED if SUBTRACTED in available_kinds(trace) else "raw"
    settings = settings or profile.dff
    f = deflect(_measured(trace, base), invert=invert != flip)
    dff = computed_dff(f, trace.fs, settings)
    if shown == "zscore":
        return computed_zscore(dff, trace.fs, settings)
    return (dff * np.float32(100.0)).astype(np.float32)


def y_label(
    trace,
    kind: str | None = None,
    subtract: bool = False,
    invert: bool = False,
    flip: bool = False,
) -> str:
    """The y axis label of what :func:`display_trace` shows for the row. A
    ``MEASURED`` kind is named with the viewer's deflection and its unit (the
    row's ``extra["unit"]``, else its profile's): ``F (a.u.)``, ``Fneu
    (a.u.)``, ``F - 0.7 Fneu (a.u.)``, ``mean - F (photons)``; a kind
    computed here from an inverted trace adds ``, inverted``; any other kind
    ``flip`` negated is ``-z-score``.
    """
    shown = displayed_kind(trace, kind)
    if shown is None:
        return ""
    computed = shown in COMPUTED and trace.array(shown) is None
    if shown in MEASURED or computed:
        invert = invert != flip
    elif flip:
        return f"-{DISPLAY_KINDS[shown]}"
    if shown in MEASURED:
        profile = trace_profile(trace.engine)
        unit = (trace.extra or {}).get("unit") or profile.unit
        name = DISPLAY_KINDS[shown]
        if shown == SUBTRACTED:
            name = f"F - {profile.neuropil_coeff:g} Fneu"
            if subtract or invert:
                name = f"({name})"
        if subtract and invert:
            name = f"mean - {name}"
        elif subtract:
            name = f"{name} - mean"
        elif invert:
            name = f"2 mean - {name}"
        return f"{name} ({unit})"
    if computed and invert:
        return f"{DISPLAY_KINDS[shown]}, inverted"
    return DISPLAY_KINDS[shown]
