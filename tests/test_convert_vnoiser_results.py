"""``scripts/convert_vnoiser_results.py``: results files the pipeline wrote as "voltage" renamed to vnoiser."""

import json

import numpy as np
from mbo_utilities.results import (
    Results,
    ResultUnit,
    pipeline_files,
    results_pipeline,
)
from scripts.convert_vnoiser_results import main


def test_a_voltage_results_file_and_its_registry_become_vnoiser(tmp_path):
    unit = ResultUnit(
        name="scan35",
        kind="scan",
        index=35,
        roi_names=["soma"],
        traces={"denoised": np.zeros((1, 8), np.float32)},
        member_kind="line",
        members=[np.array([0])],
    )
    old = Results(pipeline="voltage", units={unit.name: unit}).write(
        tmp_path / "session1.2026-09-21-14-30-22.voltage.zarr"
    )
    (old / "voltage").mkdir()
    (old / "voltage" / "pipeline.json").write_text("{}")
    (old / ".curation").mkdir()
    (old / ".curation" / "fast_template_curation.json").write_text("{}")
    registry = tmp_path / "roi_runs_MSession_0_MUnit_35.json"
    runs = [
        {"path": str(old), "kind": "voltage"},
        {"path": str(old / "scan35"), "kind": "voltage"},
        {"path": str(tmp_path / "plane01"), "kind": "suite2p"},
    ]
    registry.write_text(json.dumps({"runs": runs}))

    assert main([str(tmp_path)]) == 0
    assert old.is_dir() and json.loads(registry.read_text())["runs"] == runs

    assert main([str(tmp_path), "--write"]) == 0
    new = tmp_path / "session1.2026-09-21-14-30-22.vnoiser.zarr"
    assert not old.exists() and results_pipeline(new) == "vnoiser"
    assert pipeline_files(new) == new / "vnoiser"
    assert (new / "vnoiser" / "pipeline.json").is_file()
    assert (new / ".curation" / "fast_template_curation.json").is_file()
    assert list(Results.read(new).units) == ["scan35"]
    assert json.loads(registry.read_text())["runs"] == [
        {"path": str(new), "kind": "vnoiser"},
        {"path": str(new / "scan35"), "kind": "vnoiser"},
        {"path": str(tmp_path / "plane01"), "kind": "suite2p"},
    ]
    # a second run finds nothing left to rename
    assert main([str(tmp_path), "--write"]) == 0
    assert results_pipeline(new) == "vnoiser"
