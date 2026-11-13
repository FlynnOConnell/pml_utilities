"""The shipped notebook templates and how they are filled in beside a recording."""

import json
from datetime import date

from mbo_utilities import notebook_templates
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
