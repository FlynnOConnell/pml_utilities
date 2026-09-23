"""How a pipeline's traces are shown (AGENTS.md §7.6, Trace display).

One :class:`TraceProfile` per pipeline says which kinds its rows can show
(``DISPLAY_KINDS``: the trace kinds of the results zarr), which one to show
first, whether a neuropil correction applies (only a pipeline that measured a
real ``Fneu``), how a dF/F is computed from raw when the row carries none,
and what its raw trace is called. The Traces panel and the trace table read
every row through :func:`display_trace`; nothing else decides what a row's
numbers mean. A pipeline outside this package registers its profile with
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
    "TRACE_PROFILES",
    "TraceProfile",
    "available_kinds",
    "display_trace",
    "displayed_kind",
    "neuropil_overlay",
    "register_trace_profile",
    "trace_profile",
    "y_label",
]

# display kind -> y axis label; the names of results.TRACE_KINDS, in selector order
DISPLAY_KINDS: dict[str, str] = {
    "dff": "dF/F (%)",
    "denoised": "denoised",
    "zscore": "z-score",
    "raw": "F",
    "neuropil": "Fneu",
    "spikes": "spikes",
}

DFF_METHODS = ("maxmin", "percentile")


@dataclass
class DffSettings:
    """How a dF/F is computed from a raw trace: a rolling max-min baseline
    sized in seconds (``maxmin``, needs the row's ``fs``; falls back to the
    percentile without one) or a static per-row percentile."""

    method: str = "maxmin"
    window_s: float = 5.0
    sigma_s: float = 0.05
    percentile: float = 20.0


@dataclass(frozen=True)
class TraceProfile:
    """What one pipeline's rows can show and how."""

    pipeline: str
    kinds: tuple[str, ...]
    default: str
    neuropil: bool = False
    neuropil_coeff: float = 0.7
    dff_percent: bool = True
    raw_label: str = "F (a.u.)"
    dff: DffSettings = field(default_factory=DffSettings)


DEFAULT_TRACE_PROFILE = TraceProfile(
    pipeline="", kinds=("dff", "raw"), default="dff", raw_label="F (a.u.)"
)

TRACE_PROFILES: dict[str, TraceProfile] = {
    # suite2p: F, Fneu and (from lbm_suite2p_python) norm_traces in percent;
    # its plot is F - 0.7 Fneu over a static 20th percentile
    "suite2p": TraceProfile(
        pipeline="suite2p", kinds=("dff", "raw", "neuropil"), default="dff",
        neuropil=True, dff=DffSettings(method="percentile"),
    ),
    # masknmf: norm_traces in percent; its Fneu is zeros, so no correction
    "masknmf": TraceProfile(
        pipeline="masknmf", kinds=("dff", "raw"), default="dff",
        dff=DffSettings(method="percentile"),
    ),
    # voltage: the curated (denoised) trace first, then the pipeline's own
    # dfof_raw (a fraction), the z-score and the lines' raw means in counts
    "voltage": TraceProfile(
        pipeline="voltage", kinds=("denoised", "dff", "zscore", "raw"), default="denoised",
        dff_percent=False, raw_label="F (counts)",
    ),
    # the ROI tool's mean engine: a mask mean plus a neuropil ring, no
    # pipeline; dF/F over the rolling baseline sized in seconds
    "mean": TraceProfile(
        pipeline="mean", kinds=("dff", "raw", "neuropil"), default="dff", neuropil=True,
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
    carries, plus ``dff`` whenever it has a raw trace to compute one from."""
    profile = trace_profile(trace.engine)
    out = []
    for kind in profile.kinds:
        if trace.array(kind) is not None or (kind == "dff" and trace.F is not None):
            out.append(kind)
    return tuple(out)


def displayed_kind(trace, kind: str | None = None) -> str | None:
    """The kind :func:`display_trace` shows for ``kind``: ``kind`` when the
    row has it, else the profile's default, else the first it has."""
    kinds = available_kinds(trace)
    if not kinds:
        return None
    if kind in kinds:
        return kind
    default = trace_profile(trace.engine).default
    return default if default in kinds else kinds[0]


def _corrected_raw(trace, neuropil: bool) -> np.ndarray:
    profile = trace_profile(trace.engine)
    f = np.asarray(trace.F, np.float32)
    if neuropil and profile.neuropil and trace.Fneu is not None:
        f = f - profile.neuropil_coeff * np.asarray(trace.Fneu, np.float32)
    return f


def display_trace(
    trace, kind: str | None = None, settings: DffSettings | None = None, neuropil: bool = True
) -> np.ndarray | None:
    """The row's trace as the panel plots it, in the kind
    :func:`displayed_kind` picks, or None when the row carries nothing.

    ``raw`` is the raw trace, neuropil-corrected when the profile offers it
    and ``neuropil`` is on. ``dff`` is the row's own dF/F when it has one
    (scaled to percent), else one computed from the corrected raw trace with
    ``settings`` (the profile's own when None). Every other kind is the
    array the row carries under that name.
    """
    shown = displayed_kind(trace, kind)
    if shown is None:
        return None
    profile = trace_profile(trace.engine)
    if shown == "raw":
        return _corrected_raw(trace, neuropil)
    if shown != "dff":
        return np.asarray(trace.array(shown), np.float32)
    if trace.norm is not None:
        norm = np.asarray(trace.norm, np.float32)
        return norm if profile.dff_percent else norm * 100.0
    f = _corrected_raw(trace, neuropil)[None, :]
    settings = settings or profile.dff
    if settings.method == "maxmin" and trace.fs:
        dff = dfof_maxmin(f, trace.fs, settings.window_s, settings.sigma_s)
    else:
        dff = dfof_percentile(f, settings.percentile)
    return (dff[0] * 100.0).astype(np.float32)


def neuropil_overlay(trace, kind: str | None = None, settings: DffSettings | None = None) -> np.ndarray | None:
    """The neuropil trace drawn under a ``raw`` or ``dff`` row on the same
    scale (raw counts, or percent over its own baseline), for a profile that
    offers the correction; None otherwise."""
    profile = trace_profile(trace.engine)
    if not profile.neuropil or trace.Fneu is None:
        return None
    shown = displayed_kind(trace, kind)
    fneu = np.asarray(trace.Fneu, np.float32)
    if shown == "raw":
        return fneu
    if shown != "dff":
        return None
    settings = settings or profile.dff
    if settings.method == "maxmin" and trace.fs:
        dff = dfof_maxmin(fneu[None, :], trace.fs, settings.window_s, settings.sigma_s)
    else:
        dff = dfof_percentile(fneu[None, :], settings.percentile)
    return (dff[0] * 100.0).astype(np.float32)


def y_label(trace, kind: str | None = None) -> str:
    """The y axis label of what :func:`display_trace` shows for the row."""
    shown = displayed_kind(trace, kind)
    if shown is None:
        return ""
    if shown == "raw":
        return trace_profile(trace.engine).raw_label
    return DISPLAY_KINDS[shown]
