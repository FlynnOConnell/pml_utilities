"""mROI selections: which ROIs a viewer opens, how the selection reaches a worker, and the pipelines running once per ROI."""

import logging
import types
from pathlib import Path

import mbo_utilities as mbo
import numpy as np
import pytest

from tests.scanimage_tiff import scanimage_header, strip_value, write_scanimage_tiff

ROI_HEIGHTS = [8, 8, 8]
WIDTH = 12
TILES = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


def _scan(tmp_path, roi_centers=None, name="scan.tif") -> Path:
    return write_scanimage_tiff(
        tmp_path / name, 6, scanimage_header(), roi_heights=ROI_HEIGHTS, roi_width=WIDTH, fly_to_lines=3,
        roi_centers=roi_centers,
    )


def _lbm_tiles(tmp_path) -> Path:
    sources = {f"SI.hScan2D.virtualChannelSettings__{i}.source": "AI0" for i in range(1, 5)}
    header = scanimage_header(sources, hChannels__channelSave=[1, 2, 3, 4])
    return write_scanimage_tiff(
        tmp_path / "lbm.tif", 4 * 6, header, roi_heights=ROI_HEIGHTS, roi_width=WIDTH, fly_to_lines=3,
        roi_centers=TILES,
    )


def test_adjacent_strips_open_stitched_and_tiles_open_split(tmp_path):
    from mbo_utilities.gui.run_gui import _load_for_viewer

    assert mbo.imread(_scan(tmp_path)).viewer_roi is None
    tiles = mbo.imread(_scan(tmp_path, TILES, "tiles.tif"))
    assert tiles.roi is None and tiles.viewer_roi == 0
    assert _load_for_viewer(str(_scan(tmp_path, TILES, "tiles.tif"))).roi == 0
    assert _load_for_viewer(str(_scan(tmp_path, TILES, "tiles.tif")), roi=2).roi == 2


def test_lbm_strips_stitch_whatever_their_scanfields_say(tmp_path):
    from mbo_utilities.arrays import LBMArray

    lbm = mbo.imread(_lbm_tiles(tmp_path))
    assert isinstance(lbm, LBMArray) and lbm.viewer_roi is None


def test_reader_kwargs_reopen_the_same_rois(tmp_path):
    path = _scan(tmp_path, TILES)
    arr = mbo.imread(path, fix_phase=False)
    assert arr.reader_kwargs == {}
    for roi in (0, 2, [1, 3]):
        arr.roi = roi
        assert mbo.imread(path, **arr.reader_kwargs).roi == roi


def test_widget_reader_kwargs_names_every_roi_subplot(tmp_path):
    import copy

    from mbo_utilities.reader import widget_reader_kwargs

    arr = mbo.imread(_scan(tmp_path, TILES), roi=0)
    views = []
    for roi in arr.iter_rois():
        view = copy.copy(arr)
        view.roi = roi
        views.append(view)
    assert widget_reader_kwargs(types.SimpleNamespace(data=views)) == {"roi": [1, 2, 3]}
    assert widget_reader_kwargs(types.SimpleNamespace(data=views[1:2])) == {"roi": 2}


def test_concat_y_stitches_a_split_selection(tmp_path):
    arr = mbo.imread(_scan(tmp_path, TILES), fix_phase=False, roi=0)
    out = tmp_path / "out"
    mbo.imwrite(arr, out, ext=".tiff", roi_mode="concat_y", overwrite=True)
    assert not list(out.glob("roi*"))
    back = mbo.imread(next(out.rglob("*.tif")))
    assert back.shape[-2:] == (8, 3 * WIDTH)
    assert arr.roi == 0


def test_task_suite2p_runs_each_roi_into_its_own_folder(tmp_path, monkeypatch):
    import lbm_suite2p_python

    from mbo_utilities.gui.tasks import task_suite2p

    calls = []
    monkeypatch.setattr(lbm_suite2p_python, "pipeline", lambda *a, **kw: calls.append(kw))
    out = tmp_path / "s2p"
    task_suite2p(
        {
            "input_path": str(_scan(tmp_path, TILES)),
            "output_dir": str(out),
            "reader_kwargs": {"roi": [1, 3]},
            "planes": [1],
            "ops": {},
        },
        logging.getLogger("test"),
    )
    assert [(Path(c["save_path"]), c["reader_kwargs"]) for c in calls] == [
        (out / "roi01", {"roi": 1}),
        (out / "roi03", {"roi": 3}),
    ]


def test_task_suite2p_on_stitched_data_writes_no_roi_folder(tmp_path, monkeypatch):
    import lbm_suite2p_python

    from mbo_utilities.gui.tasks import task_suite2p

    calls = []
    monkeypatch.setattr(lbm_suite2p_python, "pipeline", lambda *a, **kw: calls.append(kw))
    out = tmp_path / "s2p"
    task_suite2p(
        {"input_path": str(_scan(tmp_path)), "output_dir": str(out), "planes": [1], "ops": {}},
        logging.getLogger("test"),
    )
    assert [(Path(c["save_path"]), c["reader_kwargs"]) for c in calls] == [(out, None)]


class _FakePipeline:
    """Stands in for a masknmf pipeline class: records the frames it is handed and makes its run folder."""

    seen: list = []

    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
        self.torch_device = "cpu"

    @classmethod
    def from_config(cls, path, output_folder, **runtime):
        return cls(output_folder)

    def run(self, data, frame_rate):
        _FakePipeline.seen.append((self.output_folder, np.asarray(data[0])))
        run = self.output_folder / "run"
        run.mkdir(parents=True)
        return run


def test_run_planes_runs_each_roi_into_its_own_folder(tmp_path, monkeypatch):
    import mbo_utilities.masknmf.outputs
    import mbo_utilities.masknmf.pipeline as masknmf_pipeline

    monkeypatch.setattr(masknmf_pipeline, "pipeline_class", lambda config: _FakePipeline)
    monkeypatch.setattr(mbo_utilities.masknmf.outputs, "write_results", lambda *a, **kw: None)
    _FakePipeline.seen = []
    arr = mbo.imread(_scan(tmp_path, TILES), fix_phase=False, roi=[2, 3])
    out = tmp_path / "masknmf"
    runs = masknmf_pipeline.run_planes(arr, out, {"pipeline": "Fake", "configs": {}}, run_args={"frame_rate": 30.0})
    assert runs == [out / "roi02" / "zplane01" / "run", out / "roi03" / "zplane01" / "run"]
    assert [(folder, frame.shape, int(frame.max())) for folder, frame in _FakePipeline.seen] == [
        (out / "roi02" / "zplane01", (8, WIDTH), strip_value(0, 1)),
        (out / "roi03" / "zplane01", (8, WIDTH), strip_value(0, 2)),
    ]
    assert arr.roi == [2, 3]


@pytest.mark.parametrize("nested", [False, True])
def test_find_masknmf_run_names_the_folder_run_writes_into(tmp_path, nested):
    import json

    from mbo_utilities.gui.widgets.pipelines.masknmf import find_masknmf_run

    out = tmp_path / "masknmf"
    run = (out / "roi02" if nested else out) / "zplane01" / "20261120T000000_two-photon-calcium"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"pipeline": "TwoPhotonCalciumPipeline", "configs": {}}))
    (run / "results.hdf5").touch()
    assert find_masknmf_run(out)[1] == str(out)
