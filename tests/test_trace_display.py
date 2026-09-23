"""``annotation.display``: what a pipeline's rows show and how (AGENTS.md §7.6)."""

from __future__ import annotations

import numpy as np

from mbo_utilities.analysis.dff import dfof_maxmin, dfof_percentile, maxmin_baseline
from mbo_utilities.annotation import RoiTrace
from mbo_utilities.annotation.display import (
    DEFAULT_TRACE_PROFILE,
    DISPLAY_KINDS,
    TRACE_PROFILES,
    DffSettings,
    TraceProfile,
    available_kinds,
    display_trace,
    displayed_kind,
    neuropil_overlay,
    register_trace_profile,
    trace_profile,
    y_label,
)
from mbo_utilities.results import TRACE_KINDS

F = np.array([10.0, 10.0, 30.0, 10.0], np.float32)
FNEU = np.array([2.0, 2.0, 2.0, 2.0], np.float32)


def test_display_kinds_are_the_results_trace_kinds():
    assert set(DISPLAY_KINDS) == set(TRACE_KINDS)
    for profile in TRACE_PROFILES.values():
        assert set(profile.kinds) <= set(DISPLAY_KINDS) and profile.default in profile.kinds


def test_profiles_say_who_offers_a_neuropil_correction():
    # suite2p measured a real Fneu, the mean engine reads a neuropil ring;
    # masknmf's Fneu is zeros and the voltage pipeline has none
    assert [p for p, prof in TRACE_PROFILES.items() if prof.neuropil] == ["suite2p", "mean"]
    assert trace_profile("voltage").default == "denoised"
    assert trace_profile("computed from channel 0") is DEFAULT_TRACE_PROFILE


def test_registering_a_profile_makes_it_the_engines_own():
    profile = TraceProfile(pipeline="plugin", kinds=("raw",), default="raw", raw_label="photons")
    register_trace_profile(profile)
    try:
        assert trace_profile("plugin") is profile
        assert y_label(RoiTrace(uid=1, engine="plugin", F=F)) == "photons"
    finally:
        del TRACE_PROFILES["plugin"]


def test_suite2p_rows_plot_the_lsp_recipe():
    row = RoiTrace(uid=1, engine="suite2p", F=F, Fneu=FNEU)
    assert available_kinds(row) == ("dff", "raw", "neuropil")
    assert displayed_kind(row) == "dff" and y_label(row) == "dF/F (%)"
    corr = F - 0.7 * FNEU
    f0 = float(np.percentile(corr, 20))
    np.testing.assert_allclose(display_trace(row), (corr - f0) / f0 * 100.0, rtol=1e-5)
    f0 = float(np.percentile(F, 20))
    np.testing.assert_allclose(display_trace(row, neuropil=False), (F - f0) / f0 * 100.0, rtol=1e-5)
    np.testing.assert_array_equal(display_trace(row, "raw"), corr)
    assert y_label(row, "raw") == "F (a.u.)"
    np.testing.assert_array_equal(display_trace(row, "neuropil"), FNEU)
    # the run's own norm_traces win outright, already in percent
    norm = np.array([0.0, 5.0, 50.0, 0.0], np.float32)
    np.testing.assert_array_equal(display_trace(RoiTrace(uid=1, engine="suite2p", F=F, norm=norm)), norm)
    # the neuropil rides the same scale as what is shown, and only where offered
    np.testing.assert_allclose(neuropil_overlay(row), np.zeros(4), atol=1e-3)
    np.testing.assert_array_equal(neuropil_overlay(row, "raw"), FNEU)
    assert neuropil_overlay(RoiTrace(uid=1, engine="masknmf", F=F, Fneu=np.zeros(4))) is None
    assert neuropil_overlay(RoiTrace(uid=1, engine="suite2p", F=F)) is None


def test_voltage_rows_show_the_curated_trace_first_and_dff_in_percent():
    denoised = np.array([0.0, 1.0, 2.0, 1.0], np.float32)
    dff = np.array([0.0, 0.1, 0.5, 0.0], np.float32)
    row = RoiTrace(uid=0, member=0, engine="voltage", norm=dff, kinds={"denoised": denoised})
    assert available_kinds(row) == ("denoised", "dff")
    np.testing.assert_array_equal(display_trace(row), denoised)
    assert y_label(row) == "denoised"
    # the pipeline stores dfof_raw as a fraction; the panel shows percent
    np.testing.assert_allclose(display_trace(row, "dff"), dff * 100.0)
    # a kind the row lacks falls back to the default, and the label follows
    assert displayed_kind(row, "zscore") == "denoised" and y_label(row, "zscore") == "denoised"
    # a line's raw means: counts, and a dF/F computed over the rolling baseline
    line = RoiTrace(uid=0, member=1, engine="voltage", F=F, fs=1000.0)
    assert available_kinds(line) == ("dff", "raw")
    assert y_label(line, "raw") == "F (counts)"
    expected = dfof_maxmin(F[None, :], 1000.0, 5.0, 0.05)[0] * 100.0
    np.testing.assert_allclose(display_trace(line, "dff"), expected)
    assert display_trace(RoiTrace(uid=0, member=2, engine="voltage")) is None


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
        display_trace(RoiTrace(uid=1, engine="mean", F=F, Fneu=FNEU, fs=10.0), settings=settings),
        (corr - f0) / f0 * 100.0, rtol=1e-5,
    )


def test_dff_math():
    F2 = np.vstack([F, np.zeros(4, np.float32)])
    base = maxmin_baseline(F2, fs=1.0, window_s=3.0, sigma_s=0.0)
    assert base.shape == F2.shape and np.all(base[1] == 0)
    # a zero baseline gives zero, never inf
    assert np.all(np.isfinite(dfof_maxmin(F2, fs=1.0, window_s=3.0, sigma_s=0.0)))
    pct = dfof_percentile(F2, 20.0)
    np.testing.assert_allclose(pct[0], (F - np.percentile(F, 20)) / np.percentile(F, 20), rtol=1e-5)
    assert np.all(pct[1] == 0)
