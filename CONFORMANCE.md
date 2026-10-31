# Conformance backlog

Known places where the code does not follow `AGENTS.md`. Fix on touch; do not add new
ones. Remove an entry when its fix lands. Line numbers date from the sweep they were
recorded in and drift; grep for the symbol.

**Lazy arrays**

- Every reader still inherits `Shape5DMixin` (`arrays/bin.py:43`, `h5.py:112`,
  `isoview/array.py:1886`, `mesc.py:814`, `mp4.py:664`, `numpy.py:79`,
  `suite2p.py:573`, `tiff.py:543,970`, `zarr.py:68`). Target: subclass `LazyArray`
  directly, delete `Shape5DMixin` (`arrays/_base.py:40`).
- `ZarrArray(dims=...)` is documented (`arrays/zarr.py:108`) but ignored, and the
  reader never reads `dimension_names` or `attrs["dims"]` back
  (`arrays/zarr.py:294-311`). `H5Array` never reads the `dims` attr the writer stamps
  (`arrays/h5.py:212-233` vs `_writers.py:1285`). Both work only because the rank
  guess happens to match the writer's rank rule.
- `PiezoArray` reports T = volumes but `fs` = ScanImage `scanFrameRate`
  (`arrays/tiff.py:1795`); `num_timepoints / fs` is not the duration. Target:
  `fs = scanFrameRate / frames_per_volume`.
- `LBMPiezoArray` lays out T = piezo positions, C = 1, Z = beamlets
  (`arrays/tiff.py:2174-2176`); `docs/file_formats.md` says C = beamlets,
  Z = piezo positions. Fix the doc or the layout, and say which axis is a calibration
  sweep rather than time.
- `_imwrite_base`'s per-plane path builds `OutputMetadata` with the legacy
  `frame_indices`/`plane_indices` kwargs and then re-stamps every timepoint alias by
  hand (`arrays/_base.py:560-607`). Target: `source_shape`/`source_dims`/`selections`
  like the volumetric writers, no manual alias stamps.
- The per-plane `_write_tiff`, `_write_h5`, `_write_zarr` behind `_get_file_writer`
  (`_writers.py:442-470, 597-643, 715-843, 1852-1998`) are unreachable from
  `imwrite`; only `.bin` and `.npy` reach `_write_plane`. Delete or route.
- `metadata/base.py:154,411` still describe shapes as TZYX / TYX / YX.

**Metadata**

- Readers stamp aliases instead of canonical keys: `TiffArray` writes `num_frames`,
  `num_planes` (`arrays/tiff.py:704-710,731-738,777-784,827-833`); `Suite2pArray` volumes
  write `nplanes`/`num_planes` (`arrays/suite2p.py:762-763`); `_metadata_from_ops`
  writes `num_planes`, `frame_rate` (`metadata/io.py:53-74`); `imwrite` writes
  `num_frames`/`nframes` for a truncation (`writer.py:334-336`).
- `ScanImageArray.metadata` mutates `_metadata` on every read (`arrays/tiff.py:1221-1246`).
- IsoView deposits the camera rate as `fps` (`arrays/isoview/array.py:506`), which
  is an `fs` alias (`metadata/base.py:224`), so `arr.fs` resolves to the camera rate
  even though the reader intends `fs` to stay unset (`arrays/isoview/array.py:2454-2457`).
  Target: `fs` = T-axis rate (`vps`) when known; camera rate under
  `isoview_camera_fps`.
- `_extract_tiff_scale` resolves `finterval` → `fs` and `XResolution` → `dx` itself
  (`arrays/isoview/array.py:915-935`) instead of depositing the ImageJ keys.
- ScanImage ingest rounds `fs` and `pixel_resolution` to two decimals
  (`metadata/scanimage.py:453,458`, `metadata/io.py:613`).
- Registry labels say "Frame Rate" / "Frame Interval" (`metadata/base.py:231-232,262-263`).
  Target: "Sampling rate" / "Sampling interval" per §6.1.
- `pixel_size_um` is registered under both `dx` and `dy` (`metadata/base.py:166,186`);
  `get_canonical_name("pixel_size_um")` answers `dy`. Needs a "scalar applies to
  both" transform instead of a double alias.
- 85 direct dict reads of rate/resolution/count keys bypass the registry, most in
  `_writers.py` (23), `arrays/isoview/array.py` (13), `gui/tasks.py` (8),
  `arrays/_average_view.py` (5). Target: `arr.<key>` or `get_param`.
- `docs/development.md:364-383` documents `get_pipelines_by_category` and
  `get_readable_extensions`, which do not exist; `docs/canonical_metadata.md:271`
  links `dim_metadata_refactor.md`, which does not exist.

**Logging**

- `log.get` children carry `propagate=False` and no handler (`log.py:22`): outside
  the GUI their INFO is silent and only WARNING+ leaks bare. Only `isoview` and
  `mbo.arrays.isoview.consolidate` are bridged into the worker log
  (`gui/tasks.py:858,1208`); `mbo.writers`, `mbo.reader` and `mbo.arrays.*` never
  reach the Process Console. `_sysmem.py:147-152` works around it by logging to the
  root. Target: the wiring in §8.1.
- Second logger tree: `logging.getLogger("mbo_utilities")` in
  `metadata/output.py:147`, `metadata/params.py:247`,
  `arrays/features/_slicing.py:164`, `gui/viewers/pollen_calibration.py:164`;
  `logging.getLogger(__name__)` in `pipeline_registry.py:11`.
- `hpc/pipeline.py` narrates compute with 23 `print` calls: no level, timestamp or
  job id, and no progress during input staging, the suite2p run, or copy-back.
- `imwrite(debug=)` toggles the `mbo.writer` logger only, while `_writers.py` logs to
  `mbo.writers` (`writer.py:33,233-239`, `_writers.py:21`), and `debug=True` hides
  the tqdm bars (`_writers.py:319`). `mbo convert --debug` (`cli.py:592`) inherits
  both.
- No `--log-level` or `-v` on `mbo` or `mbo hpc`: `mbo --debug` is DEBUG-or-nothing,
  and `mbo hpc` has neither.
- `docs/development.md:98-99,117-118` document `log.enable` and `log.disable`, which
  do not exist.

**Pipelines**

- Built-in widgets are hardcoded in `gui/widgets/pipelines/__init__.py:51-73` and
  built-in tasks in `gui/tasks.py:1608-1619`; the `mbo_utilities.pipelines`
  entry-point group is empty (`ROIs`, the one widget that declares `info`, is off
  the Process tab for now); none declares `task_type` or `task_func`. Target: §7.2 for every built-in, entry points as the only
  registration path.
- Suite2p's `PipelineInfo` is registered from the reader module with category
  `segmentation` (`arrays/suite2p.py:31-54`); MaskNMF, ROI workflow, and the IsoView
  processing modes have no `PipelineInfo` at all (the four `isoview-*` infos are
  readers).
- `hpc/` is suite2p-only and routes `[parameters]` keys by a hardcoded suite2p list
  (`hpc/config.py:66-73`, `hpc/pipeline.py:65-100`).
- Legacy `hpc/*.sh` and `hpc/run_pipeline.py` duplicate `mbo hpc`; remove once the
  submitit path is validated on the cluster.

**Style**

Counts from `ruff check` on `manual-roi-model` (2026-09-23), after the banner and
dead-code sweep. The workflow autofixes what it can; the rest is fix-on-touch.
When a family reaches zero, move its rule into `select`.

- Selected, not autofixable: `PTH` 43, `D301` 15, `ERA001` 9 (all false positives
  on prose that reads like code), `E402`/`E721`/`E741` 10, `D200` 5, `UP` 4,
  `F403`/`F405` star imports 4, `D404` 2.
- Two real defects ruff finds and nobody has fixed: `cli.py:2405` uses an undefined
  `as_zarr`, and `arrays/zarr.py:31` rebinds `logger`.
- Ignored until swept: `D205` 671, `E501` 521, `D400` 35.
- Not yet selected: `PLC0415` 1688 function-local imports (most are the sanctioned
  heavy packages; needs per-import `noqa` before enabling), `G004` 441 f-strings in
  log calls, `BLE001` 414 blind excepts, `N` 350, `T20` 181 `print` calls, `SIM`
  131, `S110` 126 `try`/`except`/`pass`, `B` 82.
- Not ruff-checkable: 0 banner or section-header comments (swept 2026-09-23; do not
  add more), 34 `logging.getLogger` calls, 172 nested `def`s in the library and 96
  in tests.
