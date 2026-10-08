"""The shipped notebook templates, how they are filled in beside a recording, and the File > Open
Notebook menu that lists them.
"""

import json
import time
from datetime import date
from pathlib import Path

from imgui_bundle import imgui
from mbo_utilities import notebook_templates
from mbo_utilities.gui.widgets import notebooks
from mbo_utilities.notebook_templates import (
    TEMPLATES,
    notebook_path,
    scripts_dir,
    template_dir,
    templates_for,
    write_notebook,
)

SPINE = next(t for t in TEMPLATES if t.filename == "asap7_spine_pipeline.ipynb")


def test_templates_for_a_mesc_include_the_spine_pipeline():
    names = [t.filename for t in templates_for("/data/scan.MESC")]
    assert "asap7_spine_pipeline.ipynb" in names
    assert "mbo_user_guide.ipynb" in names
    assert "asap7_spine_pipeline.ipynb" not in [
        t.filename for t in templates_for("/data/raw")
    ]
    assert templates_for(None) == [t for t in TEMPLATES if not t.suffix]


def test_shipped_spine_template_is_clean_and_carries_its_tokens():
    text = (template_dir() / SPINE.filename).read_text(encoding="utf-8")
    assert SPINE.path_token in text
    assert SPINE.unit_token in text
    assert "pip install" not in text
    nb = json.loads(text)
    assert all(cell.get("outputs", []) == [] for cell in nb["cells"])


def test_scripts_dir_sits_beside_the_file_or_the_raw_folder(tmp_path):
    assert scripts_dir(tmp_path / "expt" / "scan.mesc") == tmp_path / "expt" / "scripts"
    assert scripts_dir(tmp_path / "expt" / "raw") == tmp_path / "expt" / "scripts"


def test_write_notebook_fills_path_and_unit_and_keeps_an_existing_copy(
    tmp_path, monkeypatch
):
    demos = tmp_path / "demos"
    demos.mkdir()
    (demos / SPINE.filename).write_text(
        f'MESC_PATH = Path("{SPINE.path_token}")\nUNIT = "{SPINE.unit_token}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(notebook_templates, "template_dir", lambda: demos)
    mesc = tmp_path / "expt" / "scan.mesc"
    mesc.parent.mkdir()
    dest = scripts_dir(mesc)
    out = write_notebook(SPINE, dest, mesc, "MSession_0/MUnit_3")
    assert out == notebook_path(SPINE, dest, date.today(), "MSession_0/MUnit_3")
    assert out.name.endswith("_asap7_spine_pipeline_MSession_0_MUnit_3.ipynb")
    text = out.read_text(encoding="utf-8")
    assert mesc.resolve().as_posix() in text
    assert 'UNIT = "MSession_0/MUnit_3"' in text
    assert SPINE.path_token not in text
    out.write_text("edited", encoding="utf-8")
    assert write_notebook(SPINE, dest, mesc, "MSession_0/MUnit_3") is None
    assert out.read_text(encoding="utf-8") == "edited"
    other = write_notebook(SPINE, dest, mesc, "MSession_0/MUnit_16")
    assert other is not None and other != out
    assert 'UNIT = "MSession_0/MUnit_16"' in other.read_text(encoding="utf-8")
    assert (
        write_notebook(SPINE, dest, mesc, "MSession_0/MUnit_3", overwrite=True) == out
    )
    assert "edited" not in out.read_text(encoding="utf-8")


class Host:
    """What the menu is handed: the preview widget's attributes it reads."""

    def __init__(self, fpath):
        self.fpath = fpath
        self.image_widget = None


def wait_for_scan(menu, timeout=10.0):
    deadline = time.monotonic() + timeout
    while menu.scanning and time.monotonic() < deadline:
        time.sleep(0.05)


def test_template_entries_point_beside_the_data(tmp_path):
    mesc = tmp_path / "expt" / "scan.mesc"
    found = notebooks.template_notebooks(Host(str(mesc)))
    assert [n.label for n in found] == [t.title for t in templates_for(mesc)]
    assert all(n.group == notebooks.TEMPLATE_GROUP for n in found)
    assert all(Path(n.path).parent == scripts_dir(mesc) for n in found)
    assert all(n.backend == "jupyter" for n in found)
    assert notebooks.template_notebooks(Host(None)) == []
    assert notebooks.template_notebooks(Host([str(mesc)]))[0].root == str(
        scripts_dir(mesc)
    )


def test_scripts_entries_list_the_folder_beside_the_data(tmp_path):
    mesc = tmp_path / "expt" / "scan.mesc"
    assert notebooks.scripts_notebooks(Host(str(mesc))) == []
    scripts = scripts_dir(mesc)
    scripts.mkdir(parents=True)
    (scripts / "analysis.ipynb").write_text("{}", encoding="utf-8")
    found = notebooks.scripts_notebooks(Host(str(mesc)))
    assert [n.label for n in found] == ["analysis.ipynb"]
    assert found[0].group == notebooks.SCRIPTS_GROUP


def test_the_menu_lists_both_groups_and_draws(tmp_path, monkeypatch):
    monkeypatch.setattr(notebooks, "_menu", None)
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    mesc = tmp_path / "expt" / "scan.mesc"
    scripts = scripts_dir(mesc)
    scripts.mkdir(parents=True)
    (scripts / "analysis.ipynb").write_text("{}", encoding="utf-8")
    host = Host(str(mesc))
    menu = notebooks.get_notebook_menu(host)
    assert notebooks.get_notebook_menu(host) is menu
    wait_for_scan(menu)
    groups = {n.group for n in menu.notebooks()}
    assert groups == {notebooks.SCRIPTS_GROUP, notebooks.TEMPLATE_GROUP}
    assert menu.errors == []
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(2):
            imgui.new_frame()
            imgui.begin("host", None, imgui.WindowFlags_.menu_bar)
            imgui.begin_menu_bar()
            if imgui.begin_menu("File"):
                notebooks.draw_notebook_menu(host)
                imgui.end_menu()
            imgui.end_menu_bar()
            imgui.end()
            menu.show()
            notebooks.draw_notebooks_window(host)
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)
    assert menu.launcher.servers == {}
