"""The vnoiser guide: every section draws on a bare imgui context, and what it
says and shows is what the pipeline and the curation window do.
"""

from __future__ import annotations

import pytest
from imgui_bundle import imgui

from mbo_utilities.gui.imgui import vnoiser_help
from mbo_utilities.gui.imgui.vnoiser_help import draw_vnoiser_help

TABLES = []
REAL_BEGIN_TABLE = imgui.begin_table


def spy_begin_table(name, *a, **k):
    TABLES.append(name)
    return REAL_BEGIN_TABLE(name, *a, **k)


def guide_frames(state, n=3):
    """Draw the guide for ``n`` frames on a bare imgui context."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 2400)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(n):
            imgui.new_frame()
            state["open"], state["keys"] = draw_vnoiser_help(
                state["open"], state["keys"]
            )
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_every_section_draws_and_the_page_stays_open(monkeypatch):
    TABLES.clear()
    monkeypatch.setattr(imgui, "begin_table", spy_begin_table)
    state = {"open": True, "keys": False}
    guide_frames(state)
    assert state == {"open": True, "keys": False}
    assert set(TABLES) == {
        "##stages",
        "##window",
        "##domains",
        "##panels",
        "##rules",
        "##modes",
        "##files",
    }


def test_a_closed_page_draws_nothing_and_keeps_the_popups_state():
    # no imgui context: a closed page must not touch imgui
    assert draw_vnoiser_help(False, True) == (False, True)
    assert draw_vnoiser_help(False, False) == (False, False)


def test_the_sketches_agree_with_their_own_counts():
    """The made-up curation sketch counts what its lines pass, as its slider card says."""
    peaks = [0.08 + 0.8 * h for _s, h in vnoiser_help.SPIKES]
    counts = {name: count for name, _color, _at, count in vnoiser_help.SLIDERS}
    n = len(peaks)
    assert counts["thr"] == f"{sum(v > vnoiser_help.THRESHOLD_AT for v in peaks)} found"
    assert (
        counts["peak"] == f"{sum(v >= vnoiser_help.AUTO_PASS_AT for v in peaks)}/{n}"
    )
    assert (
        counts["PC1"]
        == f"{sum(u >= vnoiser_help.PC1_AT for u, _v in vnoiser_help.PCA)}/{n}"
    )
    assert counts["cos"] == f"{sum(c >= 0.8 for c in vnoiser_help.COSINES)}/{n}"
    # a rule-made call is pale, and a candidate under the cosine line is never an auto yes
    for call, cosine in zip(vnoiser_help.CALLS, vnoiser_help.COSINES, strict=True):
        assert call in vnoiser_help.LABELS
        assert not (call == "auto_yes" and cosine < 0.8)
    # every ROI of the made-up scan is in exactly one domain
    rois = [r for _name, _text, members in vnoiser_help.DOMAINS for r in members]
    assert sorted(rois) == list(range(len(vnoiser_help.LINES)))


def test_the_numbers_are_the_pipelines_defaults():
    """Every number the stage table quotes is a default of ``VoltageSettings``; the colours are
    the curation window's.
    """
    pytest.importorskip("vnoiser")
    from mbo_utilities.gui import event_curation
    from mbo_utilities.vnoiser import LABEL_RGBA
    from mbo_utilities.vnoiser.params import VoltageSettings

    settings = VoltageSettings()
    dfof, den, events = settings.dfof, settings.denoiser, settings.events
    levels = [f"{v:g}" for v in den.soft_levels]
    text = " ".join(str(part) for row in vnoiser_help.STAGES for part in row)
    for want in (
        f"{dfof.sigma_dfof:.0f} samples wide",
        f"({dfof.sigma_baseline:.0f} samples)",
        f"first {dfof.n_startup} samples",
        f"{den.n_scales} scales",
        f"from {den.scale_min:.0f} to {den.scale_max:.0f} samples",
        f"{den.n_subclusters} frequency bands",
        f"{den.n_components} principal components",
        f"first {den.n_comp_clu} clustered",
        f"above {den.slow_upthres} SD",
        f"or {den.fast_upthres} SD",
        f"{', '.join(levels[:-1])} and {levels[-1]}",
        f"{den.lp_cutoff_hz:.0f} Hz baseline",
        f"{events.bp_low:.0f}-{events.bp_high:.0f} Hz",
        f"+ {events.thres_bp_sd} SD",
        f"+ {events.thres_amp_sd} SD",
        f"{events.duration_thres_ms:.0f} ms",
        f"{events.distance_samples} samples apart",
        f"{settings.runtime.reference_fs:.0f} Hz",
    ):
        assert want in text, want
    assert dfof.negative and settings.runtime.output_format == "zarr"
    assert not settings.runtime.convert
    for mine, theirs in (
        (vnoiser_help.THRESHOLD, event_curation.THRESHOLD_COLOR),
        (vnoiser_help.AUTO_PASS, event_curation.AUTO_PASS_COLOR),
        (vnoiser_help.PC1, event_curation.PC1_COLOR),
        (vnoiser_help.COSINE, event_curation.COSINE_COLOR),
        (vnoiser_help.SNIPPET, event_curation.SNIPPET_COLOR),
    ):
        assert (mine.x, mine.y, mine.z) == pytest.approx(theirs[:3], abs=2e-3)
    assert set(vnoiser_help.LABELS) == set(LABEL_RGBA)
    for name, color in vnoiser_help.LABELS.items():
        assert (color.x, color.y, color.z) == pytest.approx(
            LABEL_RGBA[name][:3], abs=2e-3
        )
    # the curation window reads the guide's keybinds table and lists the guide in it
    assert event_curation.KEYBINDS is vnoiser_help.KEYBINDS
    assert ("h", "the vnoiser guide") in vnoiser_help.KEYBINDS
