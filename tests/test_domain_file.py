"""The domain file: layouts, block starts, the one-table form and domain means."""

import json

import numpy as np
import pytest
from mbo_utilities.annotation.domains import (
    DomainFile,
    Layout,
    domain_means,
    domains_path,
    find_domains,
)


def test_put_moves_scans_between_layouts():
    book = DomainFile([Layout(["1", "2", "3"], {"a": [0]}), Layout(["4"], {"b": [1]})])
    book.put(Layout(["3", "4"], {"c": [2]}))
    assert [(lay.scans, list(lay.domains)) for lay in book.layouts] == [
        (["1", "2"], ["a"]),
        (["3", "4"], ["c"]),
    ]
    book.put(Layout(["1", "2"], {"a": [0, 1]}))
    assert book.layouts[0].domains == {"a": [0, 1]}


def test_a_layout_without_scans_covers_the_rest():
    book = DomainFile([Layout([], {"all": [0, 1]}), Layout(["5"], {"x": [0]})])
    assert list(book.layout_of("9").domains) == ["all"]
    assert list(book.layout_of("5").domains) == ["x"]
    book.put(Layout([], {"new": [1]}))
    assert [list(lay.domains) for lay in book.layouts] == [["new"], ["x"]]


def test_set_blocks_replaces_only_the_scans_given():
    book = DomainFile(first_env=["1", "10"], scan_labels={"10": ["Odor on"]})
    book.set_blocks(["2", "3"], ["3"], {"3": ["New env"]})
    assert book.first_env == ["1", "3", "10"]
    assert book.scan_labels == {"10": ["Odor on"], "3": ["New env"]}
    book.set_blocks(["10"], [], {})
    assert book.first_env == ["1", "3"] and "10" not in book.scan_labels


def test_the_one_table_form_reads_as_one_layout(tmp_path):
    path = tmp_path / "domains.json"
    path.write_text(
        json.dumps(
            {
                "domains": {"soma": [0, 1]},
                "scans": [35],
                "run": ["soma"],
                "first_env": [35],
            }
        )
    )
    book = DomainFile.read(path)
    assert [(lay.scans, lay.domains, lay.run) for lay in book.layouts] == [
        (["35"], {"soma": [0, 1]}, ["soma"])
    ]
    assert book.labels is None
    again = DomainFile.from_json(book.to_json())
    assert again == book
    with pytest.raises(ValueError):
        DomainFile.from_json({"domains": {}})


def test_find_domains_prefers_the_roi_folder_file(tmp_path):
    mesc = tmp_path / "data" / "x.mesc"
    mesc.parent.mkdir()
    mesc.touch()
    assert find_domains(mesc) is None
    (mesc.parent / "domains.json").write_text("{}")
    assert find_domains(mesc) == mesc.parent / "domains.json"
    DomainFile([Layout([], {"a": [0]})]).write(domains_path(mesc))
    assert find_domains(mesc) == mesc.parent / "x.domains.json"
    root = tmp_path / "rois"
    assert domains_path(mesc, root).parent.parent == root
    assert find_domains(mesc, root) == mesc.parent / "domains.json"


def test_domain_means_weigh_lines_by_their_pixels():
    lines = np.array([[1.0, 1.0], [4.0, 4.0], [10.0, 10.0]], np.float32)
    means = domain_means(
        lines,
        {"ab": [0, 1], "c": [2], "cell": {"rois": [0], "pixels": "bright"}, "far": [7]},
        {0: 10, 1: 30, 2: 5},
    )
    assert list(means) == ["ab", "c"]
    np.testing.assert_allclose(means["ab"], [3.25, 3.25])
    np.testing.assert_allclose(means["c"], [10.0, 10.0])
