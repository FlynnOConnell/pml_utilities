"""``annotation.display``: what a pipeline's rows show and how (AGENTS.md §7.6)."""

from __future__ import annotations

import numpy as np
import pytest
from mbo_utilities.analysis.dff import (
    baseline,
    dfof_maxmin,
    dfof_percentile,
    maxmin_baseline,
    robust_sd,
)
from mbo_utilities.annotation import RoiTrace
from mbo_utilities.annotation.display import (
    DEFAULT_TRACE_PROFILE,
    DFF_METHODS,
    DISPLAY_KINDS,
    SUBTRACTED,
    TRACE_PROFILES,
    DffSettings,
    TraceProfile,
    available_kinds,
    computed_dff,
    computed_zscore,
    deflect,
    display_trace,
    displayed_kind,
    register_trace_profile,
    trace_profile,
    y_label,
)
from mbo_utilities.results import TRACE_KINDS

F = np.array([10.0, 10.0, 30.0, 10.0], np.float32)
FNEU = np.array([2.0, 2.0, 2.0, 2.0], np.float32)


def test_display_kinds_are_the_results_trace_kinds_and_the_one_computed_here():
    assert set(DISPLAY_KINDS) == {*TRACE_KINDS, SUBTRACTED}
    for profile in TRACE_PROFILES.values():
        assert (
            set(profile.kinds) <= set(DISPLAY_KINDS)
            and profile.default in profile.kinds
        )


def test_profiles_say_who_measured_a_neuropil():
    # suite2p measured a real Fneu, the mean engine reads a neuropil ring;
    # masknmf's Fneu is zeros and the vnoiser pipeline has none
    assert [p for p, prof in TRACE_PROFILES.items() if SUBTRACTED in prof.kinds] == [
        "suite2p",
        "mean",
    ]
    assert trace_profile("vnoiser").default == "denoised"
    assert trace_profile("computed from channel 0") is DEFAULT_TRACE_PROFILE


def test_registering_a_profile_makes_it_the_engines_own():
    profile = TraceProfile(
        pipeline="plugin", kinds=("raw",), default="raw", unit="photons"
    )
    register_trace_profile(profile)
    try:
        assert trace_profile("plugin") is profile
        assert y_label(RoiTrace(uid=1, engine="plugin", F=F)) == "F (photons)"
    finally:
        del TRACE_PROFILES["plugin"]


def test_suite2p_rows_plot_the_lsp_recipe():
    row = RoiTrace(uid=1, engine="suite2p", F=F, Fneu=FNEU)
    assert available_kinds(row) == ("dff", "zscore", "raw", "neuropil", SUBTRACTED)
    assert displayed_kind(row) == "dff" and y_label(row) == "dF/F (%)"
    # a dF/F computed here is taken of the neuropil-subtracted trace
    corr = F - 0.7 * FNEU
    f0 = float(np.percentile(corr, 20))
    np.testing.assert_allclose(display_trace(row), (corr - f0) / f0 * 100.0, rtol=1e-5)
    # the run's own norm_traces win outright: lbm_suite2p_python writes a
    # fraction, masknmf percent, and both plot in percent
    norm = np.array([0.0, 0.05, 0.5, 0.0], np.float32)
    np.testing.assert_allclose(
        display_trace(RoiTrace(uid=1, engine="suite2p", F=F, norm=norm)), norm * 100.0
    )
    np.testing.assert_array_equal(
        display_trace(RoiTrace(uid=1, engine="masknmf", F=F, norm=norm)), norm
    )
    # a run that z-scored its norm_traces shows them as a z-score, not a dF/F
    scored = RoiTrace(uid=1, engine="suite2p", F=F, kinds={"zscore": norm})
    assert available_kinds(scored) == ("dff", "zscore", "raw")
    np.testing.assert_array_equal(display_trace(scored, "zscore"), norm)
    assert y_label(scored, "zscore") == "z-score"


def test_the_raw_trace_its_neuropil_and_their_difference_are_three_kinds():
    row = RoiTrace(uid=1, engine="suite2p", F=F, Fneu=FNEU)
    np.testing.assert_array_equal(display_trace(row, "raw"), F)
    np.testing.assert_array_equal(display_trace(row, "neuropil"), FNEU)
    np.testing.assert_allclose(display_trace(row, SUBTRACTED), F - 0.7 * FNEU)
    # each in the unit the pipeline measured in
    assert [y_label(row, kind) for kind in ("raw", "neuropil", SUBTRACTED)] == [
        "F (a.u.)",
        "Fneu (a.u.)",
        "F - 0.7 Fneu (a.u.)",
    ]
    # nothing to subtract without a neuropil, and masknmf's zeros are not one
    assert available_kinds(RoiTrace(uid=1, engine="suite2p", F=F)) == (
        "dff",
        "zscore",
        "raw",
    )
    masknmf = RoiTrace(uid=1, engine="masknmf", F=F, Fneu=np.zeros(4))
    assert available_kinds(masknmf) == ("dff", "zscore", "raw")
    f0 = float(np.percentile(F, 20))
    np.testing.assert_allclose(display_trace(masknmf), (F - f0) / f0 * 100.0, rtol=1e-5)
    spikes = RoiTrace(uid=1, engine="suite2p", F=F, kinds={"spikes": FNEU})
    assert y_label(spikes, "spikes") == "spikes (a.u.)"


def test_vnoiser_rows_show_the_curated_trace_first_and_dff_in_percent():
    denoised = np.array([0.0, 1.0, 2.0, 1.0], np.float32)
    dff = np.array([0.0, 0.1, 0.5, 0.0], np.float32)
    row = RoiTrace(
        uid=0, member=0, engine="vnoiser", norm=dff, kinds={"denoised": denoised}
    )
    assert available_kinds(row) == ("denoised", "dff")
    np.testing.assert_array_equal(display_trace(row), denoised)
    assert y_label(row) == "denoised"
    # the pipeline stores dfof_raw as a fraction; the panel shows percent
    np.testing.assert_allclose(display_trace(row, "dff"), dff * 100.0)
    # a kind the row lacks falls back to the default, and the label follows
    assert (
        displayed_kind(row, "zscore") == "denoised"
        and y_label(row, "zscore") == "denoised"
    )
    # a line's raw means: counts, and a dF/F computed over the rolling baseline
    line = RoiTrace(uid=0, member=1, engine="vnoiser", F=F, fs=1000.0)
    assert available_kinds(line) == ("dff", "zscore", "raw")
    assert y_label(line, "raw") == "F (counts)"
    expected = dfof_maxmin(F[None, :], 1000.0, 5.0, 0.05)[0] * 100.0
    np.testing.assert_allclose(display_trace(line, "dff"), expected)
    assert display_trace(RoiTrace(uid=0, member=2, engine="vnoiser")) is None


def test_mean_rows_use_the_rolling_baseline_when_they_know_their_rate():
    row = RoiTrace(uid=1, engine="mean", F=F, Fneu=FNEU, fs=10.0)
    corr = F - 0.7 * FNEU
    expected = dfof_maxmin(corr[None, :], 10.0, 5.0, 0.05)[0] * 100.0
    np.testing.assert_allclose(display_trace(row), expected)
    # without fs the rolling window has no size: the percentile baseline instead
    row = RoiTrace(uid=1, engine="mean", F=F, Fneu=FNEU)
    f0 = float(np.percentile(corr, 20))
    np.testing.assert_allclose(display_trace(row), (corr - f0) / f0 * 100.0, rtol=1e-5)
    # the panel's own settings override the profile's
    settings = DffSettings(method="percentile", percentile=50.0)
    f0 = float(np.percentile(corr, 50))
    np.testing.assert_allclose(
        display_trace(
            RoiTrace(uid=1, engine="mean", F=F, Fneu=FNEU, fs=10.0), settings=settings
        ),
        (corr - f0) / f0 * 100.0,
        rtol=1e-5,
    )


def test_dff_math():
    F2 = np.vstack([F, np.zeros(4, np.float32)])
    base = maxmin_baseline(F2, fs=1.0, window_s=3.0, sigma_s=0.0)
    assert base.shape == F2.shape and np.all(base[1] == 0)
    # a zero baseline gives zero, never inf
    assert np.all(np.isfinite(dfof_maxmin(F2, fs=1.0, window_s=3.0, sigma_s=0.0)))
    pct = dfof_percentile(F2, 20.0)
    np.testing.assert_allclose(
        pct[0], (F - np.percentile(F, 20)) / np.percentile(F, 20), rtol=1e-5
    )
    assert np.all(pct[1] == 0)


DEFLECTIONS = {
    (True, False): lambda y, m: y - m,
    (False, True): lambda y, m: 2 * m - y,
    (True, True): lambda y, m: m - y,
}


def test_deflect_is_the_viewers_pixel_transform_about_the_rows_mean():
    m = F.mean()
    for (subtract, invert), expected in DEFLECTIONS.items():
        np.testing.assert_allclose(deflect(F, subtract, invert), expected(F, m))
    np.testing.assert_array_equal(deflect(F), F)
    assert F[2] == 30.0  # never written in place


def test_a_mask_trace_deflected_is_the_mask_mean_of_the_deflected_movie():
    rng = np.random.default_rng(0)
    movie = rng.integers(0, 4000, (50, 8, 8)).astype(np.float64)
    weights = rng.random((8, 8))
    weights /= weights.sum()
    trace = np.einsum("tyx,yx->t", movie, weights)
    mean_img = movie.mean(axis=0)
    for (subtract, invert), expected in DEFLECTIONS.items():
        image = expected(movie, mean_img)
        np.testing.assert_allclose(
            deflect(trace, subtract, invert),
            np.einsum("tyx,yx->t", image, weights),
            rtol=1e-4,
        )


def test_what_was_measured_from_the_pixels_follows_the_deflection():
    trace = RoiTrace(uid=1, engine="mean", F=F, Fneu=FNEU)
    measured = {"raw": F, "neuropil": FNEU, SUBTRACTED: F - 0.7 * FNEU}
    for (subtract, invert), expected in DEFLECTIONS.items():
        for kind, y in measured.items():
            np.testing.assert_allclose(
                display_trace(trace, kind, None, subtract, invert),
                expected(y, y.mean()),
                rtol=1e-6,
                atol=1e-6,
            )
    assert y_label(trace, "raw", True, True) == "mean - F (a.u.)"
    assert y_label(trace, "raw", False, True) == "2 mean - F (a.u.)"
    assert y_label(trace, "neuropil", True, False) == "Fneu - mean (a.u.)"
    assert y_label(trace, SUBTRACTED, True, True) == "mean - (F - 0.7 Fneu) (a.u.)"


def test_a_dff_computed_here_is_taken_of_the_inverted_raw_trace():
    trace = RoiTrace(uid=1, engine="mean", F=F)
    settings = DffSettings(method="percentile")
    flipped = 2 * F.mean() - F
    expected = dfof_percentile(flipped[None, :], settings.percentile)[0] * 100
    for subtract in (False, True):
        np.testing.assert_allclose(
            display_trace(trace, "dff", settings, subtract, True),
            expected,
            rtol=1e-5,
        )
    # a dF/F is relative to its baseline already; subtraction leaves it be
    np.testing.assert_allclose(
        display_trace(trace, "dff", settings, True, False),
        display_trace(trace, "dff", settings),
    )
    assert y_label(trace, "dff", False, True) == "dF/F (%), inverted"


def test_a_pipelines_own_kinds_keep_the_sign_it_wrote():
    norm = np.array([0.0, 5.0, -3.0, 1.0], np.float32)
    trace = RoiTrace(uid=1, engine="masknmf", F=F, norm=norm)
    np.testing.assert_array_equal(display_trace(trace, "dff", None, True, True), norm)
    assert y_label(trace, "dff", True, True) == DISPLAY_KINDS["dff"]


def test_flip_turns_over_whatever_is_shown():
    denoised = np.array([0.0, -4.0, 1.0, 0.0], np.float32)
    norm = np.array([0.0, -0.05, 0.01, 0.0], np.float32)
    trace = RoiTrace(
        uid=0,
        member=0,
        engine="vnoiser",
        F=F,
        norm=norm,
        kinds={"denoised": denoised, "zscore": 2 * denoised},
    )
    for kind in ("denoised", "zscore", "dff"):
        np.testing.assert_allclose(
            display_trace(trace, kind, flip=True), -display_trace(trace, kind)
        )
    np.testing.assert_allclose(
        display_trace(trace, "raw", flip=True), 2 * F.mean() - F, rtol=1e-6
    )
    # the panel's flip undoes the viewer's Invert Deflection on what was measured
    np.testing.assert_allclose(
        display_trace(trace, "raw", None, False, True, True), F, rtol=1e-6
    )
    assert y_label(trace, "zscore", flip=True) == "-z-score"
    assert y_label(trace, "dff", flip=True) == "-dF/F (%)"
    assert y_label(trace, "raw", flip=True) == "2 mean - F (counts)"
    assert y_label(trace, "raw", False, True, True) == "F (counts)"


def test_flip_inverts_the_trace_a_dff_is_computed_from():
    trace = RoiTrace(uid=1, engine="mean", F=F)
    settings = DffSettings(method="percentile")
    np.testing.assert_allclose(
        display_trace(trace, "dff", settings, flip=True),
        display_trace(trace, "dff", settings, False, True),
    )
    assert y_label(trace, "dff", flip=True) == "dF/F (%), inverted"


def _bleaching(fs: float = 100.0, n: int = 3000) -> np.ndarray:
    """A bleaching trace with a dip every second, on a dark first sample."""
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    f = 200.0 + 100.0 * np.exp(-t / 10.0) + rng.normal(0, 2.0, n)
    f[(np.arange(n) % int(fs)) < 3] -= 40.0
    return f.astype(np.float32)


def test_every_baseline_is_on_offer_and_computed_from_the_measured_trace():
    fs = 100.0
    f = _bleaching(fs)
    row = RoiTrace(uid=1, engine="mean", F=f, fs=fs)
    for method in DFF_METHODS:
        settings = DffSettings(method=method)
        shown = display_trace(row, "dff", settings)
        assert shown.shape == f.shape and np.isfinite(shown).all(), method
        np.testing.assert_allclose(shown, computed_dff(f, fs, settings) * 100.0)
    # a rolling percentile is analysis.dff's, sized in seconds
    settings = DffSettings(method="rolling", window_s=2.0, percentile=10.0)
    f0 = baseline(f[None, :], fs, "percentile", window_s=2.0, percentile=10.0)[0]
    np.testing.assert_allclose(
        computed_dff(f, fs, settings), (f - f0) / f0, rtol=1e-4, atol=1e-6
    )
    # MESc's F0: the mean over a fixed stretch
    settings = DffSettings(method="window", f0_start_s=1.0, f0_stop_s=2.0)
    f0 = f[100:200].mean()
    np.testing.assert_allclose(
        computed_dff(f, fs, settings), (f - f0) / f0, rtol=1e-4, atol=1e-6
    )
    # every moving baseline follows the decay; a fixed F0 does not
    early, late = slice(200, 700), slice(2500, None)
    for method in ("exponential", "gaussian", "maxmin", "rolling"):
        dff = computed_dff(f, fs, DffSettings(method=method))
        assert abs(np.median(dff[late]) - np.median(dff[early])) < 0.03, method
    dff = computed_dff(f, fs, DffSettings(method="window"))
    assert np.median(dff[late]) - np.median(dff[early]) < -0.15
    # without a rate every method but the static percentile has no size: the static percentile
    no_rate = RoiTrace(uid=1, engine="mean", F=f)
    np.testing.assert_allclose(
        display_trace(no_rate, "dff", DffSettings(method="gaussian")),
        dfof_percentile(f[None, :], 20.0)[0] * 100.0,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="dF/F method"):
        DffSettings(method="median")
    with pytest.raises(ValueError, match="noise"):
        DffSettings(noise="iqr")


def test_a_zscore_is_computed_from_the_dff_for_any_row_that_measured():
    fs = 100.0
    f = _bleaching(fs)
    row = RoiTrace(uid=1, engine="mean", F=f, fs=fs)
    assert "zscore" in available_kinds(row)
    settings = DffSettings(method="gaussian", slow_s=0.0, noise="sd")
    dff = computed_dff(f, fs, settings)
    z = display_trace(row, "zscore", settings)
    np.testing.assert_allclose(z, (dff - dff.mean()) / dff.std(), rtol=1e-4, atol=1e-5)
    # a robust SD that the dips do not inflate makes them bigger
    robust = DffSettings(method="gaussian", slow_s=0.0, noise="mad")
    scale = robust_sd((dff - dff.mean())[None, :])[0, 0]
    np.testing.assert_allclose(
        computed_zscore(dff, fs, robust),
        (dff - dff.mean()) / scale,
        rtol=1e-4,
        atol=1e-5,
    )
    # inverted, the dips are the events, pointing up
    up = display_trace(row, "zscore", settings, False, True)
    assert up[(np.arange(f.size) % 100) < 3].mean() > 3
    assert y_label(row, "zscore") == "z-score"
    assert y_label(row, "zscore", False, True) == "z-score, inverted"
    # a pipeline's own z-score wins and keeps its sign
    own = np.array([0.0, 1.0, -2.0, 0.5], np.float32)
    scored = RoiTrace(uid=1, engine="vnoiser", F=F, kinds={"zscore": own})
    np.testing.assert_array_equal(
        display_trace(scored, "zscore", None, True, True), own
    )


def test_maxmin_is_suite2ps_and_sits_under_the_trace():
    from scipy.ndimage import gaussian_filter1d

    f = _bleaching()[None, :].astype(np.float64)
    base = maxmin_baseline(f, 100.0, window_s=2.0, sigma_s=0.05)
    # the rolling max of the rolling min: down at the dips, under the
    # smoothed trace (a max-then-min would ride its top)
    smooth = gaussian_filter1d(f[0], 5.0)
    assert np.median(smooth[200:-200] - base[0, 200:-200]) > 5
    assert np.median(dfof_maxmin(f, 100.0, 2.0, 0.05)) > 0


def test_a_rows_unit_names_its_axis():
    row = RoiTrace(uid=1, engine="mean", F=F, extra={"unit": "photons"})
    assert y_label(row, "raw") == "F (photons)"
    assert y_label(row, "raw", True, False) == "F - mean (photons)"
