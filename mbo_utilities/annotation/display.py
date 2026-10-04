"""How a pipeline's traces are shown (AGENTS.md §7.6, Trace display).

One :class:`TraceProfile` per pipeline says which kinds its rows can show
(``DISPLAY_KINDS``: the trace kinds of the results zarr, plus ``SUBTRACTED``,
the raw trace minus its neuropil, for a pipeline that measured a real
``Fneu``), which one to show first, how a dF/F is computed from raw when the
row carries none, and the unit its raw traces are in. The Traces panel and
the trace table read every row through :func:`display_trace`; nothing else
decides what a row's numbers mean. A pipeline outside this package registers its profile with
:func:`register_trace_profile` when it registers its ``PipelineInfo``;
an engine with no profile gets ``DEFAULT_TRACE_PROFILE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mbo_utilities.analysis.dff import dfof_maxmin, dfof_percentile

__all__ = [
    "DEFAULT_TRACE_PROFILE",
    "DISPLAY_KINDS",
    "DffSettings",
    "SUBTRACTED",
    "TRACE_PROFILES",
    "TraceProfile",
    "available_kinds",
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

DFF_METHODS = ("maxmin", "percentile")


@dataclass
class DffSettings:
    """How a dF/F is computed from a raw trace: a rolling max-min baseline
    sized in seconds (``maxmin``, needs the row's ``fs``; falls back to the
    percentile without one) or a static per-row percentile.
    """

    method: str = "maxmin"
    window_s: float = 5.0
    sigma_s: float = 0.05
    percentile: float = 20.0


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


DEFAULT_TRACE_PROFILE = TraceProfile(pipeline="", kinds=("dff", "raw"), default="dff")

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
        kinds=("dff", "raw"),
        default="dff",
        dff=DffSettings(method="percentile"),
    ),
    # voltage: the curated (denoised) trace first, then the pipeline's own
    # dfof_raw (a fraction), the z-score and the lines' raw means in counts
    "voltage": TraceProfile(
        pipeline="voltage",
        kinds=("denoised", "dff", "zscore", "raw"),
        default="denoised",
        dff_percent=False,
        unit="counts",
    ),
    # the ROI tool's mean engine: a mask mean plus a neuropil ring, no
    # pipeline; dF/F over the rolling baseline sized in seconds
    "mean": TraceProfile(
        pipeline="mean",
        kinds=("dff", "raw", "neuropil", SUBTRACTED),
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
    carries, ``dff`` whenever it has a raw trace to compute one from, and
    ``SUBTRACTED`` whenever it has both a raw trace and a neuropil.
    """
    profile = trace_profile(trace.engine)
    out = []
    for kind in profile.kinds:
        if kind == SUBTRACTED:
            has = trace.F is not None and trace.Fneu is not None
        else:
            has = trace.array(kind) is not None or (
                kind == "dff" and trace.F is not None
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
) -> np.ndarray | None:
    """The row's trace as the panel plots it, in the kind
    :func:`displayed_kind` picks, or None when the row carries nothing.

    ``raw`` is the raw trace alone, ``neuropil`` its neuropil alone and
    ``SUBTRACTED`` the raw trace minus the profile's ``neuropil_coeff`` times
    the neuropil. ``dff`` is the row's own dF/F when it has one (scaled to
    percent), else one computed with ``settings`` (the profile's own when
    None) from the ``SUBTRACTED`` trace when the row has one, the raw trace
    otherwise. Every other kind is the array the row carries under that name.

    ``subtract`` and ``invert`` are the viewer's Mean Subtraction and Invert
    Deflection, applied with :func:`deflect` to what was measured from the
    pixels: the ``MEASURED`` kinds, and the trace a ``dff`` is computed from
    here (invert only; a dF/F is already relative to its baseline). A
    pipeline's own kinds are shown as it wrote them, sign included.
    """
    shown = displayed_kind(trace, kind)
    if shown is None:
        return None
    if shown in MEASURED:
        return deflect(_measured(trace, shown), subtract, invert)
    if shown != "dff":
        return np.asarray(trace.array(shown), np.float32)
    profile = trace_profile(trace.engine)
    if trace.norm is not None:
        norm = np.asarray(trace.norm, np.float32)
        return norm if profile.dff_percent else norm * 100.0
    base = SUBTRACTED if SUBTRACTED in available_kinds(trace) else "raw"
    settings = settings or profile.dff
    f = deflect(_measured(trace, base), invert=invert)
    if settings.method == "maxmin" and trace.fs:
        dff = dfof_maxmin(f[None, :], trace.fs, settings.window_s, settings.sigma_s)
    else:
        dff = dfof_percentile(f[None, :], settings.percentile)
    return (dff[0] * 100.0).astype(np.float32)


def y_label(
    trace, kind: str | None = None, subtract: bool = False, invert: bool = False
) -> str:
    """The y axis label of what :func:`display_trace` shows for the row. A
    ``MEASURED`` kind is named with the viewer's deflection and its profile's
    unit: ``F (a.u.)``, ``Fneu (a.u.)``, ``F - 0.7 Fneu (a.u.)``,
    ``mean - F (a.u.)``.
    """
    shown = displayed_kind(trace, kind)
    if shown is None:
        return ""
    if shown in MEASURED:
        profile = trace_profile(trace.engine)
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
        return f"{name} ({profile.unit})"
    if shown == "dff" and invert and trace.norm is None:
        return f"{DISPLAY_KINDS['dff']}, inverted"
    return DISPLAY_KINDS[shown]
