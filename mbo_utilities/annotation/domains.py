"""A recording file's domain tables, block starts and scan labels in one file.

A file's scans fall into layouts: scans whose ROI numbers mean the same lines,
which share one domain table (the archive's ``scanIDs_ROIs.pkl`` holds one per
experiment). Stored line positions cannot tell layouts apart (stan110_expt1's
lines move up to 7.6 um between scans that share a table), so a layout lists its
scans. Block starts (``first_env``) and labels are per scan and sit at the top,
with the trim: the seconds every reader leaves off the start of a scan
(``trim_s`` for the file, ``scan_trim`` for a scan with its own).

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
    "scan_of",
    "trim_frames",
    "trim_seconds",
    "trimmed_window",
    "write_trim",
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
    experiment's label list (None when never set), what changed at each
    block start, and the seconds left off the start of every scan
    (``trim_s``) or of one (``scan_trim``).
    """

    layouts: list[Layout] = field(default_factory=list)
    first_env: list[str] = field(default_factory=list)
    labels: list[str] | None = None
    scan_labels: dict[str, list[str]] = field(default_factory=dict)
    trim_s: float = 0.0
    scan_trim: dict[str, float] = field(default_factory=dict)

    def trim_of(self, scan) -> float:
        """Seconds left off the start of ``scan``: its own, else the file's."""
        return float(self.scan_trim.get(str(scan), self.trim_s))

    def set_trim(self, seconds: float | None, scan=None) -> None:
        """Set the file's trim, or ``scan``'s own; None gives a scan back the file's."""
        if seconds is not None and seconds < 0:
            raise ValueError(f"a trim is 0 s or more, got {seconds}")
        if scan is None:
            self.trim_s = float(seconds or 0.0)
        elif seconds is None:
            self.scan_trim.pop(str(scan), None)
        else:
            self.scan_trim[str(scan)] = float(seconds)

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
        if self.trim_s:
            out["trim_s"] = self.trim_s
        if self.scan_trim:
            out["scan_trim"] = dict(self.scan_trim)
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
        scan_trim = {str(s): float(v) for s, v in (doc.get("scan_trim") or {}).items()}
        return cls(
            layouts,
            [str(s) for s in doc.get("first_env", [])],
            None if "labels" not in doc else [str(x) for x in doc["labels"]],
            {
                str(s): [str(x) for x in v]
                for s, v in (doc.get("scan_labels") or {}).items()
            },
            float(doc.get("trim_s") or 0.0),
            scan_trim,
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


def scan_of(unit) -> str:
    """The scan a unit is in the domain file: ``MSession_0/MUnit_35`` is ``35``."""
    return str(unit).rsplit("_", 1)[-1]


def trim_seconds(fpath, unit, root=None) -> float:
    """Seconds every reader leaves off the start of ``unit`` of ``fpath``; 0 with no domain file."""
    found = find_domains(fpath, root)
    return 0.0 if found is None else DomainFile.read(found).trim_of(scan_of(unit))


def trim_frames(fpath, unit, fs, root=None) -> int:
    """:func:`trim_seconds` in timepoints at ``fs``."""
    return round(trim_seconds(fpath, unit, root) * float(fs or 0.0))


def trimmed_window(frames, n_timepoints: int, trim: int) -> tuple[int, int]:
    """The half-open ``(start, stop)`` window read of a scan of ``n_timepoints``:
    ``frames`` (None: all of it) starting no earlier than ``trim``. The window
    stays in the recording's frames, so what is read sits where it was recorded.
    """
    start, stop = (0, int(n_timepoints)) if frames is None else map(int, frames)
    start = max(start, int(trim))
    if start >= stop:
        raise ValueError(
            f"a trim of {trim} frames leaves nothing of frames {start} to {stop}"
        )
    return start, stop


def write_trim(fpath, seconds: float | None, unit=None, root=None) -> Path:
    """Store the trim of ``fpath`` (``unit`` None) or of one unit in its domain
    file, writing one with no layouts when there is none yet.
    """
    found = find_domains(fpath, root)
    book = DomainFile() if found is None else DomainFile.read(found)
    book.set_trim(seconds, None if unit is None else scan_of(unit))
    return book.write(domains_path(fpath, root))


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
