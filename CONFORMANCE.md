# CONFORMANCE.md

Known gaps between the code and `AGENTS.md`. Fix one, delete its line.
`tests/test_reader_contract.py` enforces §5 and §6 for every reader it covers; a break
there goes in its `KNOWN_BREAKS` as a strict xfail, and also here.

## Readers outside the contract test

The contract test builds a small real file per format. These readers have none yet, so
§5.1 (5D shape), §6.4 (counts agree with the shape, no aliases) and a save-as round
trip are not checked for them:

- `MescArray`, `BrukerArray`, `Lab4H5Array`, `IsoviewArray`, `Suite2pArray`, `BinArray`,
  `MasknmfRunArray`, `ResultsArray`, `MP4Array`.

## Readers stamping aliases (§6.4.2)

- `arrays/suite2p.py:682` writes `nplanes`, `num_planes`.
- `arrays/isoview/array.py:2054` writes `nplanes`, `num_planes`, `nframes`, `num_frames`.
- `arrays/bin.py:224` writes `nframes` (suite2p reads it; move it to `write_ops`).
- `arrays/_average_view.py:142,159` write `num_frames`.

## Consumers bypassing the registry (§6.5)

Sites that read `fs`, `dx`, `dy`, `dz` or `frame_rate` with `md.get(...)` / `md[...]`
instead of `get_param` or `arr.<key>`:

- `vnoiser/pipeline.py` (4), `arrays/bruker.py` (3), `_writers.py` (2),
  `arrays/masknmf_run.py` (2), `gui/widgets/pipelines/settings.py`, `arrays/demixing.py`,
  `arrays/compression.py`, `arrays/_average_view.py`, `analysis/scanphase.py`.
- `_writers.py` also reads `nframes` / `num_frames` by name (lines 270, 483, 540, 575,
  1710, 1973); safe today because `OutputMetadata.to_dict` stamps them.

## Selections (§5.7)

- `imwrite(..., timepoints="1:3")` raises `ValueError: invalid literal for int()`; §5.7
  promises `"start:stop:step"` strings. Lists work; the GUI and lsp pass lists.

## Logging (§8.1)

- `logging.getLogger` outside `log.py`: `_sysmem.py`, `pipeline_registry.py`,
  `arrays/features/_slicing.py`, `gui/_worker.py`, `gui/curation_viewer.py`, `gui/tasks.py`,
  `gui/viewers/__init__.py`, `gui/viewers/pollen_calibration.py`,
  `gui/widgets/gui_logger.py`, `masknmf/runner.py`.
- `log.py:42,54` set `propagate = False`; the §8.1 target wiring is not in place.

## ScanImage

- Slow piezo stacks, fast-z flyback frames and LBM + piezo (pollen) are tested on
  synthetic files only (`tests/scanimage_tiff.py`); no real acquisition of each was
  available.
