"""
rebase_provenance_paths: stale pipeline-recorded paths are repaired against
the opened tree or dropped — never re-embedded verbatim.

Every case runs on tmp_path.
"""

from mbo_utilities.metadata.base import (
    rebase_provenance_paths,
)

DEAD = r"C:\Users\loson\data\movie_0001.tif"


def _make_anchor(tmp_path):
    anchor = tmp_path / "demo" / "run" / "plane01"
    anchor.mkdir(parents=True)
    return anchor


def test_dead_path_rebased_from_grandparent_raw(tmp_path):
    anchor = _make_anchor(tmp_path)
    raw_dir = anchor.parent.parent / "raw"
    raw_dir.mkdir()
    target = raw_dir / "movie_0001.tif"
    target.write_bytes(b"x")

    out = rebase_provenance_paths({"data_path": DEAD}, anchor)

    assert out["data_path"] == str(target)


def test_dead_path_no_match_dropped_rest_intact(tmp_path):
    anchor = _make_anchor(tmp_path)
    md = {"data_path": DEAD, "fs": 17.0, "nframes": 2000}

    out = rebase_provenance_paths(md, anchor)

    assert "data_path" not in out
    assert out["fs"] == 17.0
    assert out["nframes"] == 2000
    # input dict is not mutated
    assert md["data_path"] == DEAD


def test_list_with_live_and_dead_entries(tmp_path):
    anchor = _make_anchor(tmp_path)
    live = anchor / "session.tif"
    live.write_bytes(b"x")

    out = rebase_provenance_paths({"file_paths": [str(live), DEAD]}, anchor)

    assert out["file_paths"] == [str(live)]


def test_empty_string_save_path0_untouched(tmp_path):
    anchor = _make_anchor(tmp_path)

    out = rebase_provenance_paths({"save_path0": ""}, anchor)

    assert out["save_path0"] == ""


def test_non_str_values_untouched_never_raises(tmp_path):
    anchor = _make_anchor(tmp_path)
    md = {
        "data_path": 123,
        "file_list": None,
        "raw_file": ["ok", 42],
        "reg_file": {"nested": "dict"},
        "fast_disk": 3.14,
        "save_path": (1, 2),
    }

    out = rebase_provenance_paths(md, anchor)

    assert out["data_path"] == 123
    assert out["file_list"] is None
    assert out["raw_file"] == ["ok", 42]
    assert out["reg_file"] == {"nested": "dict"}
    assert out["fast_disk"] == 3.14
    assert out["save_path"] == (1, 2)


def test_string_data_path_scalar_shape_preserved(tmp_path):
    # real db.npy stores data_path as a STRING, not a list — the scalar
    # shape must survive the round trip
    anchor = _make_anchor(tmp_path)
    raw_dir = anchor.parent / "raw"
    raw_dir.mkdir()
    target = raw_dir / "movie_0001.tif"
    target.write_bytes(b"x")

    out = rebase_provenance_paths({"data_path": DEAD}, anchor)

    assert isinstance(out["data_path"], str)
    assert out["data_path"] == str(target)
