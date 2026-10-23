"""vnoiser pipeline settings: what the Run tab edits and the worker rebuilds.

GUI-free and JSON round-trippable, like ``masknmf.params``. The defaults are
the archive's (``Denoiser.upstream``), so an untouched run reproduces a PF
folder; ``from_provenance`` reads them back out of a folder's ``pipeline.json``.
The parameters counted in samples are written for the archive's frame rate
(``runtime.reference_fs``); ``VnoiserSettings.at_fs`` scales them to a scan's.
``traces`` (:class:`~mbo_utilities.analysis.roi_traces.TraceSettings`) says
what a domain's pixel mean goes through before dF/F; ``dfof`` how dF/F and
the z-score are taken; ``polarity`` which of the two sign flips, if any,
turns a dimming indicator's events upright.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from vnoiser.denoiser import ClusteringConfig, cwtReducerConfig, thresConfig

from mbo_utilities.analysis.dff import BASELINES, NOISE_ESTIMATES, dfof_zscore
from mbo_utilities.analysis.roi_traces import TraceSettings
from vnoiser import Denoiser, DfofConfig, SpikeDetectConfig

__all__ = [
    "POLARITIES",
    "VnoiserDfofSettings",
    "VnoiserDenoiserSettings",
    "VnoiserEventSettings",
    "VnoiserRuntimeSettings",
    "VnoiserSettings",
]


# which way a dimming indicator (JEDI, ASAP) is turned upright: vnoiser flips
# the z-score (the archive's), pml_utilities inverts F about its mean before
# dF/F (the viewer's Invert Deflection, so dF/F rises too), or neither
POLARITIES = ("flip_z", "invert_f", "none")


@dataclass
class VnoiserDfofSettings:
    """dF/F and z-score (:func:`mbo_utilities.analysis.dff.dfof_zscore`).

    ``method`` picks the baseline (:data:`~mbo_utilities.analysis.dff.BASELINES`):
    ``gaussian`` uses ``sigma_dfof`` samples (the archive's), ``percentile``
    and ``maxmin`` a ``window_s`` second window (``percentile`` the rank),
    ``window`` the mean over ``f0_start_s`` to ``f0_stop_s``. The z-score
    removes a ``sigma_baseline`` sample Gaussian and divides by ``noise``.
    """

    method: str = "gaussian"
    sigma_dfof: float = 1500.0
    window_s: float = 5.0
    percentile: float = 20.0
    f0_start_s: float = 0.0
    f0_stop_s: float = 1.0
    sigma_baseline: float = 5000.0
    n_startup: int = 1000
    noise: str = "sd"
    negative: bool = True

    def __post_init__(self):
        if self.method not in BASELINES:
            raise ValueError(
                f"dfof.method must be one of {BASELINES}, got {self.method!r}"
            )
        if self.noise not in NOISE_ESTIMATES:
            raise ValueError(
                f"dfof.noise must be one of {NOISE_ESTIMATES}, got {self.noise!r}"
            )

    def config(self) -> DfofConfig:
        return DfofConfig(
            sigma_dfof=float(self.sigma_dfof),
            sigma_baseline=float(self.sigma_baseline),
            n_startup=int(self.n_startup),
            negative=bool(self.negative),
        )

    def apply(self, F, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """``(dfof, z)`` of ``(D, T)`` traces at ``fs`` Hz with these settings."""
        return dfof_zscore(
            F,
            fs,
            method=self.method,
            sigma=float(self.sigma_dfof),
            window_s=float(self.window_s),
            percentile=float(self.percentile),
            f0_window_s=(float(self.f0_start_s), float(self.f0_stop_s)),
            n_startup=int(self.n_startup),
            sigma_baseline=float(self.sigma_baseline),
            noise=self.noise,
            negative=bool(self.negative),
        )


@dataclass
class VnoiserDenoiserSettings:
    thres_type: str = "soft"
    soft_levels: tuple = (0.7, 0.5, 0.2, 0.01)
    hard_floor: float = 0.05
    complex_bands: bool = True
    n_scales: int = 100
    scale_min: float = 1.0
    scale_max: float = 1000.0
    lp_cutoff_hz: float = 1.0
    fir_window_ms: float = 2000.0
    fir_odd_taps: bool = False
    n_components: int = 30
    n_comp_clu: int = 10
    n_clusters: int = 5
    n_subclusters: int = 10
    slow_upthres: float = 2.0
    fast_upthres: float = 2.5

    def factory(self, fs: float) -> Denoiser:
        """The denoiser for one domain at ``fs``; ``run_pipeline``'s ``denoiser_factory``."""
        scales = np.logspace(
            np.log10(self.scale_min), np.log10(self.scale_max), num=int(self.n_scales)
        )
        return Denoiser(
            fs,
            freq_scales=scales,
            lp_cutoff=float(self.lp_cutoff_hz),
            fir_window_ms=float(self.fir_window_ms),
            cfg_clust=ClusteringConfig(
                n_components=int(self.n_components),
                n_comp_clu=int(self.n_comp_clu),
                n_clusters=int(self.n_clusters),
                n_subclusters=int(self.n_subclusters),
            ),
            cfg_reducer=cwtReducerConfig(
                slow_upthres=float(self.slow_upthres),
                fast_upthres=float(self.fast_upthres),
                complex_bands=bool(self.complex_bands),
            ),
            cfg_thres=thresConfig(
                thres_type=self.thres_type,
                soft_levels=tuple(float(v) for v in self.soft_levels),
                hard_floor=float(self.hard_floor),
            ),
            fir_odd_taps=bool(self.fir_odd_taps),
        )


@dataclass
class VnoiserEventSettings:
    detect: bool = True
    bp_low: float = 2.0
    bp_high: float = 400.0
    thres_bp_sd: float = 3.5
    thres_amp_sd: float = 4.0
    duration_thres_ms: float = 5.0
    distance_samples: int = 3

    def config(self) -> SpikeDetectConfig:
        return SpikeDetectConfig(
            bp=(float(self.bp_low), float(self.bp_high)),
            thres_bp_sd=float(self.thres_bp_sd),
            thres_amp_sd=float(self.thres_amp_sd),
            duration_thres_ms=float(self.duration_thres_ms),
            distance_samples=int(self.distance_samples),
        )


@dataclass
class VnoiserRuntimeSettings:
    # off: stop after dF/F and the z-score, no wavelet denoising or peaks
    denoise: bool = True
    save_cwt: bool = False
    overwrite: bool = True
    reference_fs: float = 1075.2688
    # "zarr": one <stem>.<stamp>.vnoiser.zarr results file (mbo_utilities.results),
    # its own files in a vnoiser/ folder inside it; "pkl": the archive's PF folder of pickles
    output_format: str = "zarr"


OUTPUT_FORMATS = ("pkl", "zarr")


def _section(cls, d):
    """A section dataclass from a dict, unknown keys dropped, lists back to tuples."""
    d = d or {}
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        if isinstance(f.default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


@dataclass
class VnoiserSettings:
    """One worker-args payload for the vnoiser pipeline."""

    traces: TraceSettings = field(default_factory=TraceSettings)
    dfof: VnoiserDfofSettings = field(default_factory=VnoiserDfofSettings)
    denoiser: VnoiserDenoiserSettings = field(default_factory=VnoiserDenoiserSettings)
    events: VnoiserEventSettings = field(default_factory=VnoiserEventSettings)
    runtime: VnoiserRuntimeSettings = field(default_factory=VnoiserRuntimeSettings)

    @property
    def polarity(self) -> str:
        """:data:`POLARITIES`: which flip ``traces.invert`` and ``dfof.negative`` make."""
        if self.traces.invert and self.dfof.negative:
            raise ValueError(
                "the trace is inverted twice: traces.invert (pml_utilities) and "
                "dfof.negative (vnoiser) each flip it; keep one"
            )
        if self.traces.invert:
            return "invert_f"
        return "flip_z" if self.dfof.negative else "none"

    @polarity.setter
    def polarity(self, value: str) -> None:
        if value not in POLARITIES:
            raise ValueError(f"polarity must be one of {POLARITIES}, got {value!r}")
        self.traces.invert = value == "invert_f"
        self.dfof.negative = value == "flip_z"

    @classmethod
    def recommended(cls) -> VnoiserSettings:
        """For new recordings: the archive's denoiser on photon counts (the
        file's conversion offset applied, so dF/F has its real size) with a
        robust z-score. The archive's own settings stay ``VnoiserSettings()``.
        """
        out = cls()
        out.traces.convert = True
        out.dfof.noise = "mad"
        return out

    def at_fs(self, fs: float) -> VnoiserSettings:
        """A copy for scans at ``fs`` Hz: the parameters counted in samples
        (the dF/F sigmas, start-up samples, wavelet scales and peak spacing)
        are scaled by ``fs / runtime.reference_fs`` so they keep the
        archive's durations, and the peak band-pass is capped below Nyquist.
        The copy's ``reference_fs`` is ``fs``, so applying it again changes
        nothing; at the reference rate the parameters are untouched.
        """
        out = copy.deepcopy(self)
        ref = float(self.runtime.reference_fs or 0)
        k = float(fs) / ref if ref > 0 else 1.0
        # the archive's rate is 1000 / 0.93 Hz and reference_fs its rounding; within 1 ppm it is the same rate
        if not math.isclose(k, 1.0, rel_tol=1e-6):
            out.dfof.sigma_dfof = float(self.dfof.sigma_dfof) * k
            out.dfof.sigma_baseline = float(self.dfof.sigma_baseline) * k
            out.dfof.n_startup = max(1, int(round(self.dfof.n_startup * k)))
            out.denoiser.scale_min = max(1.0, float(self.denoiser.scale_min) * k)
            out.denoiser.scale_max = max(
                out.denoiser.scale_min, float(self.denoiser.scale_max) * k
            )
            out.events.distance_samples = max(
                1, int(round(self.events.distance_samples * k))
            )
            out.runtime.reference_fs = float(fs)
        out.events.bp_high = min(float(self.events.bp_high), 0.95 * float(fs) / 2)
        if out.events.bp_low >= out.events.bp_high:
            out.events.bp_low = out.events.bp_high / 2
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["denoiser"]["soft_levels"] = list(d["denoiser"]["soft_levels"])
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> VnoiserSettings:
        d = d or {}
        traces = dict(d.get("traces") or {})
        # runs before the traces section kept the conversion under runtime
        if "convert" not in traces and "convert" in (d.get("runtime") or {}):
            traces["convert"] = d["runtime"]["convert"]
        return cls(
            traces=_section(TraceSettings, traces),
            dfof=_section(VnoiserDfofSettings, d.get("dfof")),
            denoiser=_section(VnoiserDenoiserSettings, d.get("denoiser")),
            events=_section(VnoiserEventSettings, d.get("events")),
            runtime=_section(VnoiserRuntimeSettings, d.get("runtime")),
        )

    @classmethod
    def from_provenance(cls, provenance: dict | None) -> VnoiserSettings:
        """From a PF folder's ``pipeline.json`` (``vnoiser.run_pipeline`` output); defaults without one."""
        if not provenance:
            return cls()
        if isinstance(provenance.get("settings"), dict):
            return cls.from_dict(provenance["settings"])
        out = cls(dfof=_section(VnoiserDfofSettings, provenance.get("dfof")))
        den = provenance.get("denoiser") or {}
        scales = den.get("freq_scales") or {}
        clustering, reducer, threshold = (
            den.get("clustering") or {},
            den.get("reducer") or {},
            den.get("threshold") or {},
        )
        out.denoiser = _section(
            VnoiserDenoiserSettings,
            {
                "thres_type": threshold.get("thres_type"),
                "soft_levels": threshold.get("soft_levels"),
                "hard_floor": threshold.get("hard_floor"),
                "complex_bands": reducer.get("complex_bands"),
                "slow_upthres": reducer.get("slow_upthres"),
                "fast_upthres": reducer.get("fast_upthres"),
                "n_scales": scales.get("num"),
                "scale_min": scales.get("min"),
                "scale_max": scales.get("max"),
                "lp_cutoff_hz": den.get("lp_cutoff"),
                "fir_window_ms": den.get("fir_window_ms"),
                "fir_odd_taps": den.get("fir_odd_taps"),
                "n_components": clustering.get("n_components"),
                "n_comp_clu": clustering.get("n_comp_clu"),
                "n_clusters": clustering.get("n_clusters"),
                "n_subclusters": clustering.get("n_subclusters"),
            },
        )
        events = provenance.get("events")
        if events is None:
            out.events = VnoiserEventSettings(detect=False)
        else:
            bp = events.get("bp") or (2.0, 400.0)
            out.events = _section(
                VnoiserEventSettings,
                {
                    "detect": True,
                    "bp_low": bp[0],
                    "bp_high": bp[1],
                    "thres_bp_sd": events.get("thres_bp_sd"),
                    "thres_amp_sd": events.get("thres_amp_sd"),
                    "duration_thres_ms": events.get("duration_thres_ms"),
                    "distance_samples": events.get("distance_samples"),
                },
            )
        source = provenance.get("source") or {}
        if "convert" in source:
            out.traces.convert = bool(source["convert"])
        out.runtime.save_cwt = "cwts.h5" in (provenance.get("files") or [])
        return out
