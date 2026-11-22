"""Labels, tooltips, tiers and columns of the vnoiser settings, for the vnoiser tab's Parameters popup.

A hint is keyed by the settings group (``traces``, ``dfof``, ``denoiser``,
``events``, ``runtime``) and the field. ``COLUMNS`` gives each stage's column
its sections in order; a field's tier decides whether it sits in its section
or in the column's Advanced or Stable fold. ``("traces", "polarity")`` stands
for :attr:`~mbo_utilities.vnoiser.params.VnoiserSettings.polarity`, which sets
``traces.invert`` and ``dfof.negative`` together.
"""

from __future__ import annotations

from dataclasses import dataclass

TIERS = ("important", "normal", "advanced", "stable")
POLARITY = ("traces", "polarity")
POLARITY_DEFAULT = "flip_z"
STAGE_DENOISER = "Wavelet denoiser"
STAGE_PEAKS = "Peaks"
# fields another control stands for
HIDDEN = {("traces", "invert"), ("dfof", "negative"), ("runtime", "denoise")}


@dataclass(frozen=True)
class Hint:
    """How the Parameters popup shows one field.

    ``tier`` is one of :data:`TIERS`: an important field gets a boxed label,
    advanced and stable ones sit in folds. ``choices`` maps each value to the
    word its dropdown shows. ``fmt`` is a float's printf format. ``seconds``
    marks a sample count at the reference rate, shown and edited in seconds.
    """

    label: str
    tooltip: str = ""
    tier: str = "normal"
    choices: dict | None = None
    fmt: str = "%.6g"
    lo: float | None = None
    seconds: bool = False

    def __post_init__(self):
        if self.tier not in TIERS:
            raise ValueError(
                f"tier {self.tier!r} of {self.label!r} is not one of {TIERS}"
            )


POLARITY_LABELS = {
    "flip_z": "dims: flip z (vnoiser)",
    "invert_f": "dims: invert F (pml_utilities)",
    "none": "brightens: as recorded",
}
BACKGROUND_LABELS = {
    "none": "none",
    "dim": "dimmest pixels",
    "outside": "outside the domain",
    "domain": "a background domain",
}
REFERENCE_LABELS = {"none": "none", "ratio": "divide", "regress": "regress out"}
BASELINE_LABELS = {
    "gaussian": "Gaussian",
    "percentile": "rolling percentile",
    "maxmin": "rolling max-min",
    "exponential": "bleaching fit",
    "window": "fixed window (MESc F0)",
    "none": "none (z of F)",
}
HINTS: dict[str, dict[str, Hint]] = {
    "traces": {
        "polarity": Hint(
            "Polarity",
            "Which way the indicator's events point, and which stage turns them up; exactly one "
            "inversion is ever applied.\n"
            "dims: flip z: JEDI and ASAP dim on depolarization; vnoiser negates the z-score, as "
            "the archive did, and dF/F stays negative-going.\n"
            "dims: invert F: the trace is turned over about its mean (2 x mean - F) before dF/F, "
            "the viewer's Invert Deflection on these pixels; dF/F and z both rise.\n"
            "brightens: an indicator that brightens, left as recorded.",
            "important",
            choices=POLARITY_LABELS,
        ),
        "convert": Hint(
            "Photon counts",
            "Apply the file's conversion offset so zero means no photons. MESc's raw counts sit "
            "about 1000 above it, which makes dF/F several times too small. Off reproduces the archive.",
            "important",
        ),
        "background": Hint(
            "Background",
            "Subtracted from each domain's mean before dF/F.\n"
            "dimmest pixels: a low percentile of the ROI's pixels, frame by frame.\n"
            "outside the domain: the mean of the ROI's pixels the domain leaves out.\n"
            "a background domain: the F of one domain of the table, which then gets no trace of its own.",
            choices=BACKGROUND_LABELS,
        ),
        "background_percentile": Hint(
            "Percentile",
            "The percentile of the ROI's pixels taken each frame.",
            fmt="%.0f",
            lo=1.0,
        ),
        "background_domain": Hint(
            "Domain", "The domain whose F is the background (often bg)."
        ),
        "background_coeff": Hint(
            "Coefficient",
            "The background is multiplied by this before it is subtracted.",
            fmt="%.2f",
        ),
        "reference": Hint(
            "Reference",
            "A structural channel recorded beside the indicator (mScarlet beside ASAP7), read "
            "through the same pixels.\n"
            "divide: F / R x mean(R), for brightness changes both channels share.\n"
            "regress out: removes the least-squares fit of R from F, for motion and hemodynamic artifacts.",
            choices=REFERENCE_LABELS,
        ),
        "reference_channel": Hint(
            "Channel", "The channel the structural marker is on."
        ),
        "score_seconds": Hint(
            "Scoring span (s)",
            "How much of the middle of the recording the brightest, most active and most "
            "correlated pixel rules rank pixels on.",
            "advanced",
            fmt="%.0f",
            lo=1.0,
        ),
        "score_highpass_hz": Hint(
            "Scoring high-pass (Hz)",
            "Fluctuations below this are left out of the active and correlated scores, so slow "
            "drift and motion shared by every pixel do not count.",
            "advanced",
            fmt="%.0f",
            lo=0.0,
        ),
    },
    "dfof": {
        "method": Hint(
            "Baseline",
            "F0 under dF/F = (F - F0) / F0.\n"
            "Gaussian: a Gaussian low-pass of F, the archive's.\n"
            "rolling percentile, rolling max-min: a low envelope over a window, robust to spikes.\n"
            "bleaching fit: a decaying exponential plus a constant.\n"
            "fixed window: the mean over a stretch you name, as MESc's dF/F does.\n"
            "none: no division; the z-score is taken of F itself.",
            "important",
            choices=BASELINE_LABELS,
        ),
        "sigma_dfof": Hint(
            "Gaussian sigma (s)",
            "The Gaussian baseline's sigma; the archive's 1500 samples at 1075 Hz are 1.4 s.",
            fmt="%.2f",
            lo=0.0,
            seconds=True,
        ),
        "window_s": Hint("Window (s)", "The rolling window.", fmt="%.1f", lo=0.05),
        "percentile": Hint(
            "Percentile", "The rank the rolling window takes.", fmt="%.0f", lo=1.0
        ),
        "f0_start_s": Hint(
            "F0 from (s)",
            "F0 is the mean of F from here to F0 to.",
            fmt="%.2f",
            lo=0.0,
        ),
        "f0_stop_s": Hint(
            "F0 to (s)", "F0 is the mean of F from F0 from to here.", fmt="%.2f", lo=0.0
        ),
        "noise": Hint(
            "Noise",
            "What the z-score divides by: the SD of dF/F, or 1.4826 x the median absolute "
            "deviation, which spikes do not inflate.",
            "important",
            choices={"sd": "SD", "mad": "robust SD (MAD)"},
        ),
        "sigma_baseline": Hint(
            "Slow baseline (s)",
            "The slow Gaussian removed from dF/F before the z-score; the archive's 5000 samples are 4.7 s.",
            fmt="%.2f",
            lo=0.0,
            seconds=True,
        ),
        "n_startup": Hint(
            "Start-up samples",
            "Leading samples replaced by the mean of the rest (filter start-up).",
            "advanced",
            lo=0,
        ),
    },
    "denoiser": {
        "thres_type": Hint(
            "Mask type",
            "Soft keeps a fraction of every band outside events; hard keeps one floor. The archive used soft.",
            "important",
            choices={"soft": "soft", "hard": "hard"},
        ),
        "soft_levels": Hint(
            "Soft levels",
            "Attenuation outside events for bands below 5, 5 to 30, 30 to 80 and above 80 Hz.",
            "important",
            fmt="%.2f",
        ),
        "hard_floor": Hint(
            "Hard floor", "Attenuation outside events in hard mode.", fmt="%.2f"
        ),
        "n_scales": Hint(
            "Scales",
            "Wavelet scales, log-spaced between Scale min and Scale max.",
            lo=8,
        ),
        "scale_min": Hint(
            "Scale min", "Smallest wavelet scale in samples.", fmt="%.1f"
        ),
        "scale_max": Hint("Scale max", "Largest wavelet scale in samples.", fmt="%.0f"),
        "lp_cutoff_hz": Hint(
            "Cutoff (Hz)",
            "FIR low-pass of the z-score saved as lp_FIR1Hz; not added to the curated trace.",
            fmt="%.1f",
        ),
        "fir_window_ms": Hint(
            "FIR window (ms)",
            "Hamming window length of the baseline filter.",
            fmt="%.0f",
        ),
        "slow_upthres": Hint(
            "Open level, slow (SD)",
            "An event window opens at this many SD for bands below 30 Hz.",
            "advanced",
            fmt="%.2f",
        ),
        "fast_upthres": Hint(
            "Open level, fast (SD)",
            "An event window opens at this many SD for bands at or above 30 Hz.",
            "advanced",
            fmt="%.2f",
        ),
        "n_subclusters": Hint(
            "Bands", "Frequency bands reconstructed.", "advanced", lo=1
        ),
        "n_components": Hint(
            "PCA components", "PCA components of the wavelet rows.", "advanced", lo=2
        ),
        "n_comp_clu": Hint(
            "Components clustered",
            "Leading components the clustering uses.",
            "advanced",
            lo=1,
        ),
        "complex_bands": Hint(
            "Complex bands above 75 Hz",
            "As the original code; off takes the real part first.",
            "stable",
        ),
        "fir_odd_taps": Hint(
            "Odd FIR taps",
            "Trim the tap count to odd. Off reproduces the archive.",
            "stable",
        ),
        "n_clusters": Hint(
            "Clusters",
            "Coarse cut of the frequency tree; nothing downstream reads it.",
            "stable",
            lo=1,
        ),
    },
    "events": {
        "detect": Hint(
            "Detect peaks",
            "Write the detected peaks with the run; the curation window does not need them.",
        ),
        "bp_low": Hint(
            "Low edge (Hz)", "Low edge of the Butterworth band-pass.", fmt="%.0f"
        ),
        "bp_high": Hint(
            "High edge (Hz)",
            "High edge of the Butterworth band-pass; capped below Nyquist.",
            fmt="%.0f",
        ),
        "thres_bp_sd": Hint(
            "Band-passed (SD)",
            "Peaks of the band-passed trace above its mean plus this many SD.",
            "important",
            fmt="%.2f",
        ),
        "thres_amp_sd": Hint(
            "Amplitude (SD)",
            "Peak height on the trace itself above its mean plus this many SD.",
            "important",
            fmt="%.2f",
        ),
        "duration_thres_ms": Hint(
            "Min duration (ms)",
            "Shortest stretch above the on and off level a peak may sit on.",
            fmt="%.1f",
        ),
        "distance_samples": Hint(
            "Min spacing (samples)",
            "Samples between peaks; 3 reproduces the archive.",
            "advanced",
            lo=1,
        ),
    },
    "runtime": {
        "output_format": Hint(
            "Output format",
            "zarr: one <input>.<time>.vnoiser.zarr results file beside the input, the shape every "
            "pipeline's results share. pkl: the archive's PF folder of pickles.",
            choices={"zarr": "zarr", "pkl": "pkl"},
        ),
        "save_cwt": Hint(
            "Save cwts.h5",
            "The wavelet coefficients, about 20 bytes per sample per domain.",
        ),
        "overwrite": Hint(
            "Overwrite PF",
            "Replace an existing PF folder's pipeline files; its .curation stays.",
        ),
        "reference_fs": Hint(
            "Reference rate (Hz)",
            "The frame rate the sample-count settings are written for; they scale to each scan's rate.",
            fmt="%.4f",
            lo=1.0,
        ),
    },
}

# each stage column with its sections: a title, the settings group and its fields
COLUMNS = (
    (
        "Trace",
        (
            ("Signal", "traces", ("polarity", "convert")),
            (
                "Background",
                "traces",
                (
                    "background",
                    "background_percentile",
                    "background_domain",
                    "background_coeff",
                ),
            ),
            ("Reference channel", "traces", ("reference", "reference_channel")),
            ("Pixel scoring", "traces", ("score_seconds", "score_highpass_hz")),
        ),
    ),
    (
        "dF/F and z-score",
        (
            (
                "dF/F baseline",
                "dfof",
                (
                    "method",
                    "sigma_dfof",
                    "window_s",
                    "percentile",
                    "f0_start_s",
                    "f0_stop_s",
                ),
            ),
            ("z-score", "dfof", ("noise", "sigma_baseline")),
            ("Start-up", "dfof", ("n_startup",)),
        ),
    ),
    (
        STAGE_DENOISER,
        (
            ("Event mask", "denoiser", ("thres_type", "soft_levels", "hard_floor")),
            ("Wavelet", "denoiser", ("n_scales", "scale_min", "scale_max")),
            ("Baseline filter", "denoiser", ("lp_cutoff_hz", "fir_window_ms")),
            ("Event windows", "denoiser", ("slow_upthres", "fast_upthres")),
            (
                "Clustering",
                "denoiser",
                ("n_subclusters", "n_components", "n_comp_clu", "n_clusters"),
            ),
            ("Archive", "denoiser", ("complex_bands", "fir_odd_taps")),
        ),
    ),
    (
        STAGE_PEAKS,
        (
            ("Detection", "events", ("detect",)),
            ("Band-pass", "events", ("bp_low", "bp_high")),
            ("Thresholds", "events", ("thres_bp_sd", "thres_amp_sd")),
            ("Spacing", "events", ("duration_thres_ms", "distance_samples")),
        ),
    ),
)
RUNTIME = ("output_format", "save_cwt", "overwrite", "reference_fs")


def applies(settings, group: str, name: str) -> bool:
    """Whether a field applies with the settings as they are: a baseline's own parameters, a background's, a reference's."""
    tr, dfof = settings.traces, settings.dfof
    if group == "traces":
        if name == "background_percentile":
            return tr.background == "dim"
        if name == "background_domain":
            return tr.background == "domain"
        if name == "background_coeff":
            return tr.background != "none"
        if name == "reference_channel":
            return tr.reference != "none"
    if group == "dfof":
        if name == "sigma_dfof":
            return dfof.method == "gaussian"
        if name == "window_s":
            return dfof.method in ("percentile", "maxmin")
        if name == "percentile":
            return dfof.method == "percentile"
        if name in ("f0_start_s", "f0_stop_s"):
            return dfof.method == "window"
    if group == "denoiser" and name == "hard_floor":
        return settings.denoiser.thres_type == "hard"
    if group == "denoiser" and name == "soft_levels":
        return settings.denoiser.thres_type == "soft"
    if group == "runtime" and name in ("save_cwt", "overwrite"):
        return name == "save_cwt" or settings.runtime.output_format == "pkl"
    return True
