"""A recording file's domain tables, block starts and scan labels in one file.

A file's scans fall into layouts: scans whose ROI numbers mean the same lines,
which share one domain table (the archive's ``scanIDs_ROIs.pkl`` holds one per
experiment). Stored line positions cannot tell layouts apart (stan110_expt1's
lines move up to 7.6 um between scans that share a table), so a layout lists its
scans. Block starts (``first_env``) and labels are per scan and sit at the top.

The file is ``<stem>.domains.json`` in the recording's ROI folder
(:func:`~mbo_utilities.roi_workflow.roi_folder`), beside the data unless the
user picked a root; a ``domains.json`` beside the data, the earlier one-table
form, still reads as one layout. The vnoiser tab writes it, ``mbo vnoiser``,
the dashboard and the curation window read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities.analysis.roi_traces import domains_from_json
from mbo_utilities.roi_workflow import roi_folder

__all__ = [
    "DOMAINS_SUFFIX",
    "LEGACY_FILE",
    "DomainFile",
    "Layout",
    "domain_means",
    "domains_path",
    "find_domains",
]

DOMAINS_SUFFIX = ".domains.json"
LEGACY_FILE = "domains.json"


@dataclass
class Layout:
    """One domain table and the scans (MUnit numbers) it applies to; no scans
    means every scan no other layout lists. ``run`` names the domains a run
    takes (None: all).
    """

    scans: list[str]
    domains: dict
    run: list[str] | None = None

    def __post_init__(self):
        self.scans = [str(s) for s in self.scans]
        self.domains = {d.name: d.to_json() for d in domains_from_json(self.domains)}
        if not self.domains:
            raise ValueError("a layout defines no domains")
        if self.run is not None:
            self.run = [str(n) for n in self.run]

    def to_json(self) -> dict:
        out = {"scans": list(self.scans), "domains": dict(self.domains)}
        if self.run is not None:
            out["run"] = list(self.run)
        return out


@dataclass
class DomainFile:
    """Every layout of a recording file, its block starts in order, the
    experiment's label list (None when never set) and what changed at each
    block start.
    """

    layouts: list[Layout] = field(default_factory=list)
    first_env: list[str] = field(default_factory=list)
    labels: list[str] | None = None
    scan_labels: dict[str, list[str]] = field(default_factory=dict)

    def layout_of(self, scan) -> Layout | None:
        """The layout listing ``scan``, else the one listing no scans."""
        scan = str(scan)
        listed = next((lay for lay in self.layouts if scan in lay.scans), None)
        return listed or next((lay for lay in self.layouts if not lay.scans), None)

    def put(self, layout: Layout) -> None:
        """Make ``layout`` the table of its scans: they leave every other
        layout, and a layout left with none is dropped. It takes the place of
        the first layout it emptied.
        """
        taken = set(layout.scans)
        kept, at = [], None
        for lay in self.layouts:
            rest = [s for s in lay.scans if s not in taken]
            if lay.scans and not rest or not lay.scans and not taken:
                at = len(kept) if at is None else at
                continue
            kept.append(
                Layout(rest, lay.domains, lay.run) if rest != lay.scans else lay
            )
        kept.insert(len(kept) if at is None else at, layout)
        self.layouts = kept

    def set_blocks(self, scans, first_env, scan_labels) -> None:
        """Replace the block starts and labels of ``scans``; other scans keep theirs."""
        scans = {str(s) for s in scans}
        starts = {s for s in self.first_env if s not in scans} | {
            str(s) for s in first_env
        }
        self.first_env = sorted(starts, key=_scan_order)
        self.scan_labels = {
            s: v for s, v in self.scan_labels.items() if s not in scans
        } | {str(s): [str(x) for x in v] for s, v in scan_labels.items() if v}

    def block_start(self, scan, scans) -> str | None:
        """The block start ``scan`` falls after among ``scans`` (in order), or None before the first."""
        scans = [str(s) for s in scans]
        upto = scans[: scans.index(str(scan)) + 1] if str(scan) in scans else []
        return next((s for s in reversed(upto) if s in self.first_env), None)

    def to_json(self) -> dict:
        out = {
            "layouts": [lay.to_json() for lay in self.layouts],
            "first_env": list(self.first_env),
            "scan_labels": {s: list(v) for s, v in self.scan_labels.items()},
        }
        if self.labels is not None:
            out["labels"] = list(self.labels)
        return out

    @classmethod
    def from_json(cls, doc: dict) -> DomainFile:
        """From the file, or the one-table form (``domains``, ``scans`` and ``run`` at the top)."""
        if "layouts" in doc:
            layouts = [
                Layout(lay.get("scans", []), lay.get("domains") or {}, lay.get("run"))
                for lay in doc["layouts"]
            ]
        else:
            layouts = [
                Layout(doc.get("scans", []), doc.get("domains") or {}, doc.get("run"))
            ]
        return cls(
            layouts,
            [str(s) for s in doc.get("first_env", [])],
            None if "labels" not in doc else [str(x) for x in doc["labels"]],
            {
                str(s): [str(x) for x in v]
                for s, v in (doc.get("scan_labels") or {}).items()
            },
        )

    @classmethod
    def read(cls, path) -> DomainFile:
        return cls.from_json(json.loads(Path(path).read_text()))

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path


def _scan_order(scan: str):
    return (0, int(scan), "") if scan.isdigit() else (1, 0, scan)


def domains_path(fpath, root=None) -> Path:
    """Where ``fpath``'s domain file is written: ``<stem>.domains.json`` in its ROI folder."""
    return roi_folder(fpath, root) / f"{Path(fpath).stem}{DOMAINS_SUFFIX}"


def find_domains(fpath, root=None) -> Path | None:
    """``fpath``'s domain file, else a ``domains.json`` beside it, else None."""
    path = domains_path(fpath, root)
    if path.is_file():
        return path
    legacy = Path(fpath).parent / LEGACY_FILE
    return legacy if legacy.is_file() else None


def domain_means(lines: np.ndarray, domains: dict, sizes: dict[int, int]) -> dict:
    """``{name: trace}`` for every whole-ROI domain: its lines' means weighted
    by their pixel counts (``sizes``), which is the mean of all its pixels.
    Domains with a pixel rule, or naming a line ``lines`` lacks, are left out.
    """
    out = {}
    for d in domains_from_json(domains):
        if d.pixels != "all" or max(d.rois) >= len(lines):
            continue
        weights = np.array([sizes.get(r, 1) for r in d.rois], np.float64)
        out[d.name] = (weights @ lines[d.rois] / weights.sum()).astype(np.float32)
    return out
