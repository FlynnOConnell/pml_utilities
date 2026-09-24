"""The Cloud tab: the viewer's bridge to the imgui_cloud package.

The panel itself is tested in that package. What matters here is the seam - the
tab appears when the widget is enabled, it draws without raising whether or not
imgui_cloud is installed, and it hands the loaded dataset's folder over.
"""

from __future__ import annotations


FIGURE_SIZE = (1000, 800)


def test_dir_for_resolves_files_lists_and_folders(tmp_path):
    from mbo_utilities.gui._cloud import _dir_for

    filepath = tmp_path / "a.tif"
    filepath.write_bytes(b"x")
    assert _dir_for(str(filepath)) == str(tmp_path)
    assert _dir_for([str(filepath)]) == str(tmp_path)
    assert _dir_for(str(tmp_path)) == str(tmp_path)
    assert _dir_for(None) == ""
    assert _dir_for([]) == ""


def test_missing_package_explains_itself_instead_of_raising(monkeypatch):
    """With imgui_cloud absent the tab must degrade, not take the GUI down."""
    import builtins

    from mbo_utilities.gui import _cloud

    real_import = builtins.__import__

    def refuse_imgui_cloud(name, *args, **kwargs):
        if name.startswith("imgui_cloud"):
            raise ImportError("no imgui_cloud")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_imgui_cloud)
    drawn = []
    monkeypatch.setattr(
        _cloud.imgui, "text_colored", lambda color, text: drawn.append(text)
    )
    monkeypatch.setattr(_cloud.imgui, "button", lambda label: False)

    _cloud.draw_cloud_tab(object(), "/data/raw")
    assert any("not installed" in line for line in drawn)
    assert any("imgui_cloud" in line for line in drawn)


def test_the_panel_is_created_once_and_reused(monkeypatch):
    from mbo_utilities.gui import _cloud

    calls = []

    class FakeParent:
        pass

    def fake_draw_panel(state, dir_input=""):
        calls.append((id(state), dir_input))
        state.setdefault("cloud_panel", object())

    monkeypatch.setitem(
        __import__("sys").modules,
        "imgui_cloud.gui",
        type("M", (), {"draw_cloud_tab": staticmethod(fake_draw_panel)}),
    )
    parent = FakeParent()
    _cloud.draw_cloud_tab(parent, "/data/raw")
    _cloud.draw_cloud_tab(parent, "/data/raw")
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0], "the panel state must survive between frames"
