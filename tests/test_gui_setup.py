"""PyQt6 on a Wayland session runs Qt under XWayland, where rendercanvas can present to the window."""

import pytest

from mbo_utilities.gui import _setup


@pytest.mark.parametrize(
    ("qpa_before", "wayland_display", "qpa_after"),
    [
        (None, "wayland-1", "xcb"),
        ("wayland;xcb", "wayland-1", "xcb"),
        ("wayland", "wayland-1", "xcb"),
        ("offscreen", "wayland-1", "offscreen"),
        (None, None, None),
    ],
)
def test_wayland_sessions_with_xwayland_use_xcb(monkeypatch, qpa_before, wayland_display, qpa_after):
    monkeypatch.setattr(_setup.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    for name, value in (("QT_QPA_PLATFORM", qpa_before), ("WAYLAND_DISPLAY", wayland_display)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    _setup._use_xwayland_for_pyqt6()
    assert _setup.os.environ.get("QT_QPA_PLATFORM") == qpa_after


def test_wayland_without_xwayland_is_left_alone(monkeypatch):
    monkeypatch.setattr(_setup.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    _setup._use_xwayland_for_pyqt6()
    assert _setup.os.environ["QT_QPA_PLATFORM"] == "wayland"
