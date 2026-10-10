"""Rename results files the pipeline wrote as "voltage" to its name since 2026-10-09, vnoiser.

A results file names its pipeline three ways: the file name
(``<stem>.<stamp>.voltage.zarr``), the root ``pipeline`` attr in ``zarr.json``
and the folder of the run's own files inside it (``voltage/``). The ROI
widget's ``roi_runs*.json`` sidecars point at the file by path and say
``"kind": "voltage"``. This renames all four; no array, label or curation file
is touched, and a ``.curation`` folder inside the file moves with it.

usage:
    python scripts/convert_vnoiser_results.py ROOT [ROOT ...]          # print what would change
    python scripts/convert_vnoiser_results.py ROOT [ROOT ...] --write  # convert
"""
import argparse
import json
import sys
from pathlib import Path

from mbo_utilities.results import results_pipeline

OLD, NEW = "voltage", "vnoiser"


def main(argv=None) -> int:
    """Convert every results file and run registry under the given folders; 1 when a target name is taken."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path, help="folders to search")
    parser.add_argument(
        "--write",
        action="store_true",
        help="convert; without it the changes are only printed",
    )
    args = parser.parse_args(argv)

    blocked = 0
    renamed = {}
    for root in args.roots:
        for old in sorted(root.rglob(f"*.{OLD}.zarr")):
            if results_pipeline(old) != OLD:
                print(f"{old}: not a {OLD} results file, left alone")
                continue
            new = old.with_name(old.name[: -len(f".{OLD}.zarr")] + f".{NEW}.zarr")
            if new.exists():
                blocked += 1
                print(f"{old}: left alone, {new.name} exists")
                continue
            print(f"{old} -> {new.name}")
            renamed[str(old)] = str(new)
            if not args.write:
                continue
            meta = old / "zarr.json"
            payload = json.loads(meta.read_text(encoding="utf-8"))
            payload["attributes"]["pipeline"] = NEW
            meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if (old / OLD).is_dir():
                (old / OLD).rename(old / NEW)
            old.rename(new)

    for root in args.roots:
        for registry in sorted(root.rglob("roi_runs*.json")):
            payload = json.loads(registry.read_text(encoding="utf-8"))
            changed = 0
            for run in payload.get("runs", []):
                before = (run["path"], run.get("kind"))
                # a unit inside the file is <file>.zarr/<unit>
                at = run["path"].find(f".{OLD}.zarr")
                if at >= 0:
                    end = at + len(f".{OLD}.zarr")
                    file = run["path"][:end]
                    target = run["path"][:at] + f".{NEW}.zarr"
                    # renamed now, or by an earlier run of this script
                    if file in renamed or Path(target).exists():
                        run["path"] = target + run["path"][end:]
                if run.get("kind") == OLD:
                    run["kind"] = NEW
                changed += (run["path"], run.get("kind")) != before
            if not changed:
                continue
            print(f"{registry}: {changed} run(s)")
            if args.write:
                registry.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
