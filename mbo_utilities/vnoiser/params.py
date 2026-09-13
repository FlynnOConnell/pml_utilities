"""Voltage pipeline settings: what the Run tab edits and the worker rebuilds.

GUI-free and JSON round-trippable, like ``masknmf.params``. The defaults are
the archive's (``Denoiser.upstream``), so an untouched run reproduces a PF
folder; ``from_provenance`` reads them back out of a folder's ``pipeline.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np
from vnoiser import Denoiser, DfofConfig, SpikeDetectConfig
from vnoiser.denoiser import ClusteringConfig, cwtReducerConfig, thresConfig

__all__ = [
    "VoltageDfofSettings",
    "VoltageDenoiserSettings",
    "VoltageEventSettings",
    "VoltageRuntimeSettings",
    "VoltageSettings",
]


@dataclass
class VoltageDfofSettings:
    sigma_dfof: float = 1500.0
    sigma_baseline: float = 5000.0
    n_startup: int = 1000
    negative: bool = True

    def config(self) -> DfofConfig:
        return DfofConfig(
            sigma_dfof=float(self.sigma_dfof), sigma_baseline=float(self.sigma_baseline),
            n_startup=int(self.n_startup), negative=bool(self.negative),
        )


@dataclass
class VoltageDenoiserSettings:
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
        scales = np.logspace(np.log10(self.scale_min), np.log10(self.scale_max), num=int(self.n_scales))
        return Denoiser(
            fs,
            freq_scales=scales,
            lp_cutoff=float(self.lp_cutoff_hz),
            fir_window_ms=float(self.fir_window_ms),
            cfg_clust=ClusteringConfig(
                n_components=int(self.n_components), n_comp_clu=int(self.n_comp_clu),
                n_clusters=int(self.n_clusters), n_subclusters=int(self.n_subclusters),
            ),
            cfg_reducer=cwtReducerConfig(
                slow_upthres=float(self.slow_upthres), fast_upthres=float(self.fast_upthres),
                complex_bands=bool(self.complex_bands),
            ),
            cfg_thres=thresConfig(
                thres_type=self.thres_type, soft_levels=tuple(float(v) for v in self.soft_levels),
                hard_floor=float(self.hard_floor),
            ),
            fir_odd_taps=bool(self.fir_odd_taps),
        )


@dataclass
class VoltageEventSettings:
    detect: bool = True
    bp_low: float = 2.0
    bp_high: float = 400.0
    thres_bp_sd: float = 3.5
    thres_amp_sd: float = 4.0
    duration_thres_ms: float = 5.0
    distance_samples: int = 3

    def config(self) -> SpikeDetectConfig:
        return SpikeDetectConfig(
            bp=(float(self.bp_low), float(self.bp_high)), thres_bp_sd=float(self.thres_bp_sd),
            thres_amp_sd=float(self.thres_amp_sd), duration_thres_ms=float(self.duration_thres_ms),
            distance_samples=int(self.distance_samples),
        )


@dataclass
class VoltageRuntimeSettings:
    convert: bool = False
    save_cwt: bool = False
    overwrite: bool = True


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
class VoltageSettings:
    """One worker-args payload for the voltage pipeline."""

    dfof: VoltageDfofSettings = field(default_factory=VoltageDfofSettings)
    denoiser: VoltageDenoiserSettings = field(default_factory=VoltageDenoiserSettings)
    events: VoltageEventSettings = field(default_factory=VoltageEventSettings)
    runtime: VoltageRuntimeSettings = field(default_factory=VoltageRuntimeSettings)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["denoiser"]["soft_levels"] = list(d["denoiser"]["soft_levels"])
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> VoltageSettings:
        d = d or {}
        return cls(
            dfof=_section(VoltageDfofSettings, d.get("dfof")),
            denoiser=_section(VoltageDenoiserSettings, d.get("denoiser")),
            events=_section(VoltageEventSettings, d.get("events")),
            runtime=_section(VoltageRuntimeSettings, d.get("runtime")),
        )

    @classmethod
    def from_provenance(cls, provenance: dict | None) -> VoltageSettings:
        """From a PF folder's ``pipeline.json`` (``vnoiser.run_pipeline`` output); defaults without one."""
        if not provenance:
            return cls()
        if isinstance(provenance.get("settings"), dict):
            return cls.from_dict(provenance["settings"])
        out = cls(dfof=_section(VoltageDfofSettings, provenance.get("dfof")))
        den = provenance.get("denoiser") or {}
        scales = den.get("freq_scales") or {}
        clustering, reducer, threshold = den.get("clustering") or {}, den.get("reducer") or {}, den.get("threshold") or {}
        out.denoiser = _section(VoltageDenoiserSettings, {
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
        })
        events = provenance.get("events")
        if events is None:
            out.events = VoltageEventSettings(detect=False)
        else:
            bp = events.get("bp") or (2.0, 400.0)
            out.events = _section(VoltageEventSettings, {
                "detect": True, "bp_low": bp[0], "bp_high": bp[1],
                "thres_bp_sd": events.get("thres_bp_sd"), "thres_amp_sd": events.get("thres_amp_sd"),
                "duration_thres_ms": events.get("duration_thres_ms"),
                "distance_samples": events.get("distance_samples"),
            })
        source = provenance.get("source") or {}
        if "convert" in source:
            out.runtime.convert = bool(source["convert"])
        out.runtime.save_cwt = "cwts.h5" in (provenance.get("files") or [])
        return out
