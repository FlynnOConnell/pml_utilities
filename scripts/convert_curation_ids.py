"""Rename the recordings in a PF folder's curation files to the names the viewer uses.

Until 2026-09-17 a recording was saved as
``<animal>/<experiment>/scan=<id>/domain=<name>``. The viewer looks for
``scan=<id>/domain=<name>``, the domain named as the traces name it (``soma``
where the ROI file says ``soma1``). Only names change: every label, sample
index and line value is carried over, and the original file is kept beside
the converted one as ``<name>.json.<its date>.orig``.

usage:
    python scripts/convert_curation_ids.py PF [PF ...]          # print what would change
    python scripts/convert_curation_ids.py PF [PF ...] --write  # convert
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from mbo_utilities.results import FINAL_DOMAIN_NAMES, TRACES_PKL, Results
from mbo_utilities.vnoiser import recording_id

SAVED_ID = re.compile(r"(?:^|/)scan=([^/]+)/domain=(.+)$")


def main(argv=None) -> int:
    """Convert every curation file under the given folders; 1 when a saved name matches no recording."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="+", type=Path, help="PF folders, or folders holding them"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="convert; without it the changes are only printed",
    )
    args = parser.parse_args(argv)

    current = {}
    unmatched = 0
    for root in args.paths:
        for label_file in sorted(root.rglob(".curation/*_template_curation.json")):
            pf_dir = label_file.parent.parent
            if not (pf_dir / TRACES_PKL).is_file():
                print(f"{label_file}: not in a PF folder, left alone")
                continue
            if pf_dir not in current:
                current[pf_dir] = {
                    recording_id(unit, roi)
                    for unit in Results.from_pf(pf_dir).units.values()
                    for roi in unit.roi_names
                }
            payload = json.loads(label_file.read_text(encoding="utf-8"))
            detection = payload.get("candidate_detection", {})
            events = payload.get("events", {})
            saved = {key.rsplit("|sample=", 1)[0] for key in events}
            for per_recording in detection.values():
                if isinstance(per_recording, dict):
                    saved.update(per_recording)

            renames = {}
            missing = []
            for old in sorted(saved - current[pf_dir]):
                found = SAVED_ID.search(old)
                new = None
                if found:
                    scan, domain = found.groups()
                    for name in (domain, FINAL_DOMAIN_NAMES.get(domain, domain)):
                        if f"scan={scan}/domain={name}" in current[pf_dir]:
                            new = f"scan={scan}/domain={name}"
                            break
                if new is None:
                    missing.append(old)
                else:
                    renames[old] = new
            if missing:
                unmatched += len(missing)
                print(f"{label_file}: left alone, no recording for")
                for old in missing:
                    print(f"    {old}")
                continue
            if not renames:
                print(f"{label_file}: nothing to rename")
                continue

            print(f"{label_file}: {len(renames)} recordings")
            for old, new in renames.items():
                print(f"    {old} -> {new}")
            for field, per_recording in detection.items():
                if not isinstance(per_recording, dict):
                    continue
                renamed = {}
                for name, value in per_recording.items():
                    new = renames.get(name, name)
                    # saved under both names: the viewer's own is the newer
                    if new != name and new in per_recording:
                        print(f"    dropped {field}[{name}] = {value}, {new} has its own")
                        continue
                    renamed[new] = value
                detection[field] = renamed
            converted = {}
            for key, event in events.items():
                old, sample = key.rsplit("|sample=", 1)
                new = renames.get(old, old)
                if new != old and f"{new}|sample={sample}" in events:
                    print(f"    dropped event {key}, {new} has its own")
                    continue
                converted[f"{new}|sample={sample}"] = (
                    event if new == old else {**event, "recording": new}
                )
            moved = sum(key not in events for key in converted)
            print(f"    {moved} of {len(events)} events renamed")
            if not args.write:
                continue
            payload["events"] = converted
            stamp = datetime.fromtimestamp(label_file.stat().st_mtime).strftime(
                "%Y%m%d-%H%M%S"
            )
            backup = label_file.with_name(f"{label_file.name}.{stamp}.orig")
            shutil.copy2(label_file, backup)
            scratch = label_file.with_name(f"{label_file.name}.tmp")
            scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(scratch, label_file)
            print(f"    written; original kept as {backup.name}")

    if unmatched:
        print(f"{unmatched} saved names match no recording; those files were not changed")
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
