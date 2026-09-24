# AGENTS.md

Single source of truth for any agent (human or AI) contributing to pml_utilities.
The importable package is `mbo_utilities`; `pml_utilities` is only the distribution
name. `CLAUDE.md` imports this file and `STYLE.md`.

Where code and this file disagree, this file wins: fix the code when you touch it,
and record anything you cannot fix in [§15](#15-conformance-backlog).

---

## 1. Project layout

```
pml_utilities/
├── mbo_utilities/            # importable package
│   ├── lazy_array.py         # LazyArray base + imread dispatch registry (stdlib only)
│   ├── reader.py             # imread()
│   ├── writer.py             # imwrite()
│   ├── _writers.py           # per-format writers, ops.npy, processing_history
│   ├── squeeze.py            # opt-in SqueezedView
│   ├── arrays/               # one LazyArray subclass per format + read-time views
│   │   ├── _base.py          # _imwrite_base, ReductionMixin, TiffReaderMixin, DIMS
│   │   ├── features/         # dims, tags, slicing, selection, roi, phase, frame average, stats
│   │   ├── tiff.py zarr.py h5.py bruker.py numpy.py bin.py mesc.py suite2p.py mp4.py
│   │   └── isoview/          # IsoView light-sheet trees (four layouts, one class)
│   ├── metadata/             # canonical vocabulary, alias resolution, OutputMetadata
│   ├── pipeline_registry.py  # PipelineInfo + entry-point loading
│   ├── results.py            # the results zarr every pipeline molds into + ResultsArray (§7.5)
│   ├── masknmf/  vnoiser/    # pipeline packages: params / runner / outputs / qc
│   ├── roi_workflow.py       # register -> ROI subset -> extract | demix | discover
│   ├── hpc/                  # submitit/SLURM runner for the suite2p pipeline (`mbo hpc`)
│   ├── gui/                  # Miller Brain Studio (imgui + fastplotlib)
│   │   ├── app/              # `mbo app`: the viewer and apps docked or windowed around it (§17.1)
│   │   ├── playhead.py       # one time in seconds shared by every view (§7.6)
│   │   ├── widgets/pipelines # Run tab: one PipelineWidget per pipeline
│   │   ├── tasks.py          # worker task table: task_<name>(args, logger)
│   │   └── _worker.py        # python -m mbo_utilities.gui._worker <task_type> <args_json>
│   ├── analysis/             # scan-phase, linescan, phasecorr math
│   ├── annotation/           # GUI-free manual-ROI model: label store, trace table, RoiModel, NGFF labels zarr
│   ├── cli.py                # `mbo` (click)
│   └── assets/docs/          # in-app help pages
├── pollen/                   # pollen calibration (console script `pollen`)
├── hpc/                      # legacy sbatch scripts; fallback for `mbo hpc`
├── tests/                    # pytest; tests/local/ needs real data
├── docs/                     # Sphinx book theme + MyST
├── scripts/ demos/ examples/ docker/
└── pyproject.toml            # pins, entry points, ruff
```

`pip install -e .`, Python 3.12.7 to 3.13, `uv` for environments. Console scripts:
`mbo`, `pollen`. Entry-point groups: `mbo_utilities.lazy_arrays` (readers) and
`mbo_utilities.pipelines` (pipelines).

## 2. Layer responsibilities

| Layer | Responsibility | Must not |
|-------|----------------|----------|
| `lazy_array` | `LazyArray` base, 5D accessors, `register_array_class`, `_dispatch` | Import numpy, tifffile, zarr, or any `arrays` module |
| `arrays/*` | Open one format lazily; deposit the source's metadata; implement the 5D contract (§5) | Normalize metadata; write files; import `gui` |
| `arrays/features` | Dims, tags, selection, ROI, phase, frame average, stats; format-agnostic | Know about any one file format |
| `metadata` | Canonical vocabulary, alias resolution, `OutputMetadata`, ScanImage parsing | Read pixels |
| `writer` + `_writers` | `imwrite`; emit canonical values under each format's keys; `ops.npy`; provenance | Hand-roll alias fan-out; every emitted key comes from the registry or `OutputMetadata` |
| `masknmf` `vnoiser` `roi_workflow` `hpc` | Run a pipeline from a settings dataclass; write suite2p-shaped outputs | Import `imgui_bundle`, `fastplotlib`, or `mbo_utilities.gui` |
| `gui/widgets/pipelines` | Draw a pipeline's config; spawn its worker task | Compute inline; hold pipeline math |
| `gui/tasks` + `gui/_worker` | Re-open the source in a subprocess and call the runner | Depend on GUI state; args are JSON |
| `cli` | Thin click wrappers over `imread`/`imwrite`/runners | Hold logic unreachable from Python |

If a change crosses a boundary, split it into two PRs or justify it in the description.

## 3. Coding standards

`STYLE.md` holds the rules ruff cannot check: dependencies, functions, comments,
logging, errors, scope of a change, indexing. `pyproject.toml` `[tool.ruff]` holds
the rules it can; `.github/workflows/format.yml` applies them on every push to
`main`. Gaps between the two and the code are listed under **Style** in §15.

## 4. Docstrings

Numpy style (ruff `pydocstyle` convention `numpy`). Content rules are in `STYLE.md`.

## 5. Lazy arrays: the 5D contract

`imread()` returns a `LazyArray`. Its rank and axis order are fixed.

### 5.1 Shape

- `.shape` is always `(T, C, Z, Y, X)`; `.ndim` is always 5; `.dims` is always
  `("T", "C", "Z", "Y", "X")`. Size-1 axes are kept, never dropped.
- Axis meaning: T = samples along time (§6.1), C = color channel, can be used as a bucket for arbitrary axis 
 (e.g. camera or view for isoview light sheet microscopy),
  Z = z-plane, Y = rows, X = columns. This is OME-NGFF 0.5
  order (time, channel, space).
- `nt`, `nc`, `nz`, `ny`, `nx` are the five sizes by position; `num_timepoints`,
  `num_zplanes` are the same sizes by name through `dimension_specs`.
- A subclass implements exactly `_shape5d()`, `__getitem__`, `dtype`, `can_open()`
  and sets `self._metadata`. Everything else (`shape`, `ndim`, `dims`, `metadata`,
  `dimension_specs`, `dx/dy/dz/fs/finterval`, `slider_dims`, `squeeze`,
  `source_path`) is inherited from `LazyArray`. Subclass `LazyArray` directly;
  `Shape5DMixin` is a compatibility alias scheduled for deletion.
- Spatial extent may depend on state (`roi`, axial shifts, phase correction); temporal
  and channel extent may depend on `frame_average` and `channel`. `_shape5d()` is
  therefore computed, not cached.

Pinned by `tests/test_lazyarray_contract.py`, `tests/test_shape5d.py`,
`tests/test_natural_rank.py`.

### 5.2 Rank inference on read

When a source carries no axis labels, rank alone decides the labels. This table is the
only guess the codebase makes (`lazy_array._DEFAULT_DIMS_BY_NDIM`,
`features/_dim_labels.DEFAULT_DIMS`, `arrays/h5._DEFAULT_RAW_DIMS`,
`arrays/numpy._apply_dim_order`, `arrays/zarr._shape_tzyx`):

| Source rank | Labels | Reported 5D shape |
|-------------|--------|-------------------|
| 2D | `YX` | `(1, 1, 1, Y, X)` |
| 3D | `TYX` | `(T, 1, 1, Y, X)` |
| 4D | `TZYX` | `(T, 1, Z, Y, X)` |
| 5D | `TCZYX` | `(T, C, Z, Y, X)` |
| >5D | rejected | `ValueError` |

- 3D is time, never a z-stack; 4D is time+z, never time+channel. A 3D z-stack or a
  4D two-channel movie must be declared: `imread(x, dims="ZYX")`, `dims="TCYX"`.
  unless automatically set via the metadata getter.
- Declared `dims` are characters from `TCZYX`, length equal to the source rank, no
  duplicates. An unusable declaration warns and falls back to the rank guess; it never
  raises. The declared order is kept on `input_dims`; `.dims` still reports the 5D
  canonical order.
- `dims` travels in `reader_kwargs` so a worker re-opening the path reads it the same
  way (`source_reader_kwargs(arr)`).

Format-specific labels win over the rank guess:

| Source | Axis evidence used |
|--------|--------------------|
| ImageJ / Fiji TIFF | `frames`, `slices`, `channels`. A plain stack (`slices>1`, no `hyperstack`, no `spacing`) is read as **T**; `dims="ZYX"` keeps it as Z. Page order XYCZT. |
| OME-TIFF | `SizeT`, `SizeZ`, `SizeC`; page order from `DimensionOrder`. |
| ScanImage TIFF | `stack_type` from `si`: LBM (>2 saved channels) puts beamlets on Z and color ports on C; piezo puts slices on Z and volumes on T; single_plane has Z=1. |
| Multi-file TIFF | `planeNN` in the filename groups files onto Z; otherwise files concatenate along T; `roiN` yields one array per ROI. |
| Suite2p dir | `ops.npy` per plane dir; plane dirs stack onto Z; T is derived from the binary's file size, not `ops["nframes"]`. |
| MESc | `MethodType`: 1 timeseries `(T, C, 1, Y, X)`; 2 z-stack `(1, C, Z, Y, X)`; 6/7 linescan `(T, C, R, n_lines, width)`; 8 chessboard and 9/10 ribbon `(T, C, R, Y, X)`; 11 multicube has real depth on Z. `metadata["mesc_z_axis_meaning"]` says whether Z is `roi_index`, `depth` or `none`. |
| Bruker HDF5 | the dataset's HDF5 dimension labels (`t z y x c`); `element_size_um` in the stored order of the spatial axes. |
| H5 | dataset rank per the table; `imaging/data` 5D is `TZYXC` (Mini2P); a 4D dataset with `scan_mode` + `n_channel == shape[-1]` is `TYXC`. |
| Zarr | array rank per the table; a directory of `.zarr` stores stacks them onto Z. |
| IsoView | tree layout: TM folders → T, cameras/views → C, volume → Z. |

Pinned by `tests/test_numpy_dims.py`, `tests/test_imagej_stack.py`.

### 5.3 Indexing

- `__getitem__` takes numpy 5D semantics: a key shorter than 5 is padded with
  `slice(None)`; integer axes squeeze out; `Ellipsis` expands. Wrappers over a
  natural-rank source map keys with `arrays._base._index_5d_into_raw`.
- `np.asarray(arr)` returns one representative `(Y, X)` frame, never the whole array.
  Use `arr[:]` or chunked reads for the data.
- Reductions (`mean`, `max`, `min`, `std`, `var`, `sum`) match numpy and stream in
  chunks above 100M elements (`ReductionMixin`).
- `arr.vmin` / `arr.vmax` are the display range of the representative frame.

### 5.4 Views and sanctioned exceptions

Read-time views wrap a 5D array and stay 5D: `FrameAveragedView` (temporal binning,
T // N), `PhaseCorrectedView` (bidirectional scan phase), `AxialShiftView`
(per-plane shifts; changes Y/X when enabled). `base_array(arr)` unwraps them for
`isinstance` checks.

Four objects deliberately report a different rank:

| Object | Reported shape | Why |
|--------|----------------|-----|
| `BinArray` | the shape you passed, usually `(T, Y, X)` | it is suite2p's input format; suite2p reads it 3D |
| `MP4Array` | `(T, Y, X)` | video, not a dispatch target |
| `_ChannelView` (`imread(path, channel=n)`) | `(T, Z, Y, X)` | feeds single-channel pipelines that expect TZYX |
| `SqueezedView` (`arr.squeeze()`, `imread(squeeze=True)`) | size-1 T/C/Z dropped | notebook ergonomics; `.base` is the 5D array; writers use the base |

All four still implement `_shape5d()` and `nt/nc/nz/ny/nx`, so writers and pipelines
never branch on rank.

### 5.5 `imread` dispatch

1. `np.ndarray` → `NumpyArray`. A `SqueezedView` → its base. An object with
   `_imwrite` and `shape` → returned unchanged.
2. A path inside a `.zarr` store is redirected to the store root.
3. Every class in the `mbo_utilities.lazy_arrays` entry-point group (plus
   `register_array_class` calls) is asked `can_open(path)` in descending
   `PRIORITY`; ties keep entry-point order. First `True` wins. Priorities today:
   `IsoviewArray` 90, `ResultsArray` 70, `MescArray` 60, `BrukerArray` 60, everything else 50.
4. Inputs no class claims (file lists, `.bin`, `.klb`, `.mp4`, `reg_tif/`, mixed
   directories) fall through to the legacy chain in `reader._imread_impl`.

`can_open` must be cheap (suffix, header, sidecar presence) and never raise. Add a
class by listing it in `pyproject.toml` under `[project.entry-points."mbo_utilities.lazy_arrays"]`.

Read-time kwargs `channel`, `frame_average`, `squeeze` are consumed by `imread`
itself; `unit` (MESc), `dataset` (H5), `dims` (TIFF/NumPy/Zarr) reach the class and
round-trip through `arr.reader_kwargs`.

### 5.6 Writers: what goes on disk

`imwrite(arr, outdir, ext=...)` always receives a 5D array. Per-format rank and
metadata placement are fixed:

| `ext` | Layout on disk | Rank rule | Axis labels on disk | Metadata placement |
|-------|----------------|-----------|---------------------|--------------------|
| `.tiff` / `.tif` | one ImageJ hyperstack BigTIFF | pages in **TZCYX** order (ImageJ XYCZT); `frames`, `slices`, `channels` always stamped | ImageJ tags + `Info` JSON (tag 50839) | ImageJ: `spacing`, `finterval`, `unit`, `XResolution`/`YResolution`; full dict after `strip_for_export` in `Info` |
| `.zarr` | one Zarr v3 group, array `0`, sharded per (c, z) | **4D TZYX** when C == 1, **5D TCZYX** when C > 1 | `dimension_names` (lowercase) and `attrs["dims"]` | `attrs["ome"]` NGFF 0.5 multiscales with axes + per-axis scale; every other key as a group attr; `0/.attrs["scale"]` for napari; `attrs["scanphase"]` when phase correction was baked in |
| `.h5` | one file, dataset `/mov` (override `dataset_name`) | 4D TZYX when C == 1, 5D TCZYX when C > 1 | `dset.attrs["dims"]` | root attrs = flat metadata (scalars as-is, others `str()`) |
| `.bin` | `zplaneNN_tpAAAAA-BBBBB/data_raw.bin` per plane (+`data_chan2.bin`) | 3D TYX int16 per (c, z) | none | `ops.npy` beside it via `write_ops`; `scanphase.npy` sidecar |
| `.npy` | `tpAAAAA-BBBBB_zplaneNN.npy` per plane | 3D TYX | none | packaged with the array (`npz` layout, `.npy` suffix) |
| `.mp4` | one file per (z, c) | video | none | none |

- Filenames come from `features/_dim_tags.OutputFilename`: tags in T, C, Z order,
  spatial dims omitted, `tp` zero-padded to 5, `zplane`/`ch` to 2, ranges as
  `start-stop[-step]`. The suite2p `.bin` layout is the one exception (`zplane` first,
  to match `lbm_suite2p_python`). The same vocabulary (`TAG_REGISTRY` + `DIM_ALIASES`,
  `session` is `S`) reads tags back out of a name with `filename_tags`
  (`plane_03.bin` → `zplane03`, `mouse_V1_session1.tif` → `session01`; one-letter
  labels and unknown words such as `stan112` are not tags); the results zarr is named
  from them (§7.5). A new tag label goes in `TAG_REGISTRY`, never in a regex elsewhere.
- Every writer computes its metadata through `OutputMetadata(source, source_shape,
  source_dims, selections)` so `dz`, `fs`, `num_zplanes`, `num_timepoints`, `Lx`,
  `Ly` follow the selection (§6.6). Never re-stamp those by hand.
- Reading any written file back yields the 5D shape it was written from.

Pinned by `tests/test_roundtrip.py`, `tests/test_zarr_chunking.py`,
`tests/local/test_metadata_carrythrough.py`.

### 5.7 Selections and ROIs

- Public: `planes`, `timepoints`, `channels` (1-based int, list, or `"start:stop:step"`),
  `num_timepoints`, `num_zplanes`. `frames` and `num_frames` are deprecated aliases
  that warn.
- Internal: `selection_to_indices(arr, {...})` → `{"T": [...], "C": [...], "Z": [...]}`
  0-based; `to_lsp_kwargs` re-emits 1-based `timepoints`/`planes`/`channels` for
  `imwrite` and `lbm_suite2p_python`; `to_isoview_kwargs` emits 0-based
  `timepoints`/`cameras`. Axis aliases (`view`, `cam`, `plane`, `tile`, ...) resolve
  through `_dim_labels._SLIDER_NAME_ALIASES`.
- `arr.roi`: `None` = stitched full FOV, `0` = split every ROI, `n` = ROI n, list =
  those ROIs. `imwrite(roi_mode=RoiMode.concat_y | separate)`; `separate` fans out
  one `roiNN/` directory per ROI.

Pinned by `tests/test_selection_indices.py`.

### 5.8 Motion correction

A recording that went through motion correction reports the shifts the stage
applied as `arr.motion_correction`, a `features.MotionCorrection` or `None`
(`LazyArray` answers `None`; a reader overrides it):

| Field | Meaning |
|-------|---------|
| `source` | the stage, as a plot labels it: `RTMC` today; `suite2p`, `masknmf` when their per-frame offsets land |
| `unit` | `um` for the AOD's real-time correction, `px` for a registration's offsets |
| `traces` | `{label: (t, shift)}`, `t` in seconds on the recording's T axis, one trace per axis; a label starts with its axis letter (`X`, `Z layer 3`) |

- `MescArray.motion_correction` is the `total` RTMC curves (`rtmc_motion`); the
  `intercycle` increments stay on `arr.rtmc`. A unit that armed RTMC without it
  ever moving reports `None`.
- One GUI consumer: `gui/imgui/motion.MotionPlot`, drawn under the trace in
  linked subplots by the Traces tab (`manual_roi.draw_traces`), the line-scan
  viewer's `LineTracesPanel` and the curation dashboard, behind one `MC`
  checkbox. The plot never clamps its x axis to the traces on disk: a pipeline
  run on a frame window leaves shorter traces than the recording.
- Adding a source means overriding `motion_correction` on the reader; nothing in
  `gui` names a source.

Pinned by `tests/test_motion_plot.py`, `tests/test_mesc.py`.

## 6. Metadata: the canonical vocabulary

`arr.metadata` is a plain dict. One **canonical key** per physical quantity is
authoritative; every other spelling is an **alias** resolved by the registry in
`metadata/base.py::METADATA_PARAMS`. The word *canonical* is the registry's own field
name (`MetadataParameter.canonical`) and stays.

### 6.1 Terms

- **Canonical key**: the registry key, e.g. `fs`, `dx`, `num_zplanes`. Consumers ask
  for it and only it.
- **Alias**: a key holding the same value verbatim (`PhysicalSizeX` ≡ `dx`). Listed in
  `aliases`.
- **Transform alias**: a key holding a converted form (`finterval` = 1/`fs`,
  `XResolution` = 1/`dx`). Listed in `transforms` as `(to_canonical, from_canonical)`.
- **Sampling rate** (`fs`, Hz): the rate of the array's T axis as reported. It is the
  frame rate only for a planar time series. For LBM and IsoView it is the volume
  rate, for a MESc AOD scan it is the per-light-path cycle rate, after temporal
  binning it is the binned rate. Invariant: `num_timepoints / fs` is the duration in
  seconds. "Frame rate" is a display alias, not the canonical name.
- **Sampling interval** (`finterval`, s): `1 / fs`.
- **Vendor namespace**: source-specific extras keep a prefix and are never aliased:
  `si` (nested ScanImage header), `mesc_*`, `isoview` / `views` / `tiles`,
  `roi_groups`, `scanphase`.

### 6.2 The registry

| Canonical | Meaning | Unit | dtype | Default |
|-----------|---------|------|-------|---------|
| `dx`, `dy` | pixel size along X, Y | µm | float | 1.0 |
| `dz` | z-step | µm | float | `None` (LBM: user-supplied, never inferred) |
| `fs` | sampling rate of T | Hz | float | `None` |
| `finterval` | sampling interval | s | float | `None` |
| `vps` | volume rate, IsoView only | Hz | float | `None` |
| `Lx`, `Ly` | width, height | px | int | from shape |
| `num_timepoints` | size of T | | int | `None` |
| `num_zplanes` | size of Z | | int | 1 |
| `nchannels` | interleaved pages per frame in a raw TIFF (LBM: beamlets × colors) | | int | 1 |
| `num_color_channels` | size of C | | int | 1 |
| `num_mrois` | ScanImage mROI count | | int | 1 |
| `roi`, `fov`, `fov_um` | per-strip size, tiled FOV (px, µm) | | tuple | `None` |
| `dtype`, `shape`, `size` | array facts | | | |
| `stack_type`, `lbm_stack`, `piezo_stack` | ScanImage acquisition kind | | | `single_plane` |
| `frames_per_file`, `file_paths` | source file boundaries along T | | list | `None` |

`IMAGING_METADATA_KEYS` is the subset the GUI metadata viewer always shows.
Precision: canonical values are stored at full float precision; rounding is for
display only.

### 6.3 Platform key map

What each platform stores, and which canonical key it resolves to. A reader deposits
the left-hand keys; the registry does the rest.

| Quantity | canonical | ScanImage (`si.*`) | Femtonics MESc | ImageJ / Fiji TIFF | OME (TIFF XML, NGFF) | Suite2p `ops` | IsoView XML | h5 converters (Mini2P, AOD mesc2h5) | PrairieView / Bruker |
|---|---|---|---|---|---|---|---|---|---|
| pixel size | `dx` `dy` | `objectiveResolution × scanfields.sizeXY / pixelResolutionXY` → `pixel_resolution` tuple | scan pattern `pixelSize` / `pixelSizeL` | `XResolution` `YResolution` (px per unit, needs `unit`) | `PhysicalSizeX/Y` + `*Unit`; NGFF scale[y], scale[x] | `dx` `dy` (`umPerPixX/Y` legacy) | `camera_pixel_pitch_um / magnification` → `pixel_resolution_um` | `pixel_size_um` | `micronsPerPixel` XAxis / YAxis |
| z-step | `dz` | `hStackManager.actualStackZStepSize` else `stackZStepSize`; never for LBM | multicube `voxelSizeZ`; else `None` (ROI slots are not depths) | `spacing` | `PhysicalSizeZ`; NGFF scale[z] | `dz` (`umPerPixZ`, `z_step`) | `z_step` / `axial_step` | | ZSeries `positionCurrent` ZAxis step |
| sampling rate | `fs` | `hRoiManager.scanFrameRate`, else `1/scanFramePeriod` | `1000 / TStepInMs ÷ light paths` (`mesc_raw_frame_rate` keeps the scanner rate) | `finterval` (s) | `TimeIncrement` + `TimeIncrementUnit`; NGFF time scale (s) | `fs` | `vps` = `fps / zplanes` (`fps` is the camera rate) | `frame_rate_hz`; `frame_period` (s) | `framePeriod` (s) |
| width, height | `Lx` `Ly` | page shape | `Channel_N` shape | page shape | `SizeX` `SizeY` | `Lx` `Ly` | `dimensions` | dataset shape | `pixelsPerLine` `linesPerFrame` |
| timepoints | `num_timepoints` | pages ÷ `nchannels`, summed over files | axis 0 of `Channel_N` per `MethodType` | `frames` (or `slices` for a plain stack) | `SizeT` | `nframes` (reader trusts file size) | TM folders | dataset shape | TSeries frame count |
| z-planes | `num_zplanes` | LBM: saved beamlets per color port; piezo: `hStackManager.numSlices` | ROI count / depth per `MethodType` | `slices` | `SizeZ` | `nplanes` / plane dirs | `zplanes` = `dimensions[-1]` | dataset shape | ZSeries frame count |
| colors | `num_color_channels` | unique `hScan2D.virtualChannelSettings__N.source` ports | count of `Channel_N` | `channels` | `SizeC` | `nchannels` | cameras / views (`num_views`) | `n_channel` | `<File channel=...>` count |
| dtype | `dtype` | page dtype | `Channel_N.dtype` | page dtype | `Type` | `int16` | volume dtype | dataset dtype | `bitDepth` |

PrairieView has no reader yet; the column is the deposit contract for when one lands
(register those keys as aliases at the same time). "h5 converters" are external
scripts whose attribute names are already registered aliases.

### 6.4 Inbound: readers

1. A reader deposits the source's keys into `self._metadata` and does not normalize.
   `MescArray._build_metadata` and `H5Array.metadata` are the models: canonical keys
   where the reader computed the value itself, source keys verbatim otherwise,
   vendor extras under a namespace.
2. Never stamp aliases from a reader (`nframes`, `num_frames`, `num_planes`,
   `frame_rate`, ...). Aliases are emitted only by `normalize_metadata` at write time.
3. Never round at ingest.
4. `num_timepoints`, `num_zplanes`, `num_color_channels`, `Lx`, `Ly` in the dict
   must agree with `_shape5d()`. Shape is the truth; if they disagree, fix the dict.
5. Keys the registry can resolve through a transform (`finterval`, `XResolution`,
   `TimeIncrement`) are deposited as-is, not pre-converted.
6. A reader's `metadata` getter returns the dict; it does not mutate it.
7. Unknown quantities stay absent. `arr.dx` returns 1.0 and `arr.fs`/`arr.dz` return
   `None` when nothing was stored; the layer does not distinguish "unknown" from a
   stored 1.0, so never store a placeholder.

### 6.5 Query: consumers

- On an array: `arr.dx`, `arr.dy`, `arr.dz`, `arr.fs`, `arr.finterval`,
  `arr.num_timepoints`, `arr.num_zplanes`, `arr.nt/nc/nz/ny/nx`. This is the
  sanctioned path; it stays correct when `dims` or `metadata` change.
- On a dict: `get_param(md, "fs")`, `get_voxel_size(md)`. `get_param` walks canonical
  → aliases → `pixel_resolution` tuple → transforms; for `fs`/`finterval` it goes
  through `resolve_effective_rate`, whose precedence is `fs` > `frame_rate` >
  `finterval` > OME time scale > remaining aliases (`fps` last), and which warns once
  per divergent alias set.
- Never `md.get("fs")`, `md["dx"]`, or a hand-written fallback chain. Every such
  site is a bug the layer exists to delete.
- `get_canonical_name(name)` maps any spelling to its canonical key or `None`.

### 6.6 Reactive core

- `LazyArray.dimension_specs` builds per-dimension `{role, size, scale, unit}` from
  `dims + shape + metadata` (`features/_dim_spec.DimensionSpecs`) and is
  invalidated when `dims` is reassigned. Roles: `Y`/`X` spatial, `T`/`Z`/`C`
  iteratable, camera/trial batch.
- `OutputMetadata(source, source_shape, source_dims, selections)` is the only
  output-side layer. It scales `dz` by the Z stride, divides `fs` by the T stride,
  nulls `fs` for non-contiguous T, recomputes every count and `Lx`/`Ly`, repairs
  every rate alias present in the source, drops a carried OME block, and stamps
  `_metadata_provenance = {field: {base, stride}}` so a second hop never
  double-scales. Do not replace it with `DimensionSpecs.with_selections`.
- `FrameAveragedView` retimes through `scale_frame_rate(md, factor)`, which touches
  every registered rate spelling.

### 6.7 Outbound: writers

A writer emits canonical values under the keys the target expects. Emitted keys come
from `OutputMetadata.to_dict()`, `to_imagej()`, `to_ome_ngff()`,
`VoxelSize.to_dict()`, `normalize_metadata()`, never from a hand-written map.

| Target | Keys written |
|--------|--------------|
| ImageJ TIFF | `frames` `slices` `channels` `hyperstack` `unit=um` `spacing` `finterval`; `XResolution`/`YResolution` = `1/dx`, `1/dy`; `Info` = filtered JSON |
| OME-NGFF zarr | `ome.version=0.5`, `multiscales[0].axes` from `dims_to_ome_axes`, `coordinateTransformations.scale` = `[1/fs, dz, dy, dx]` (with `1.0` for C); `dimension_names`; flat attrs |
| H5 | flat attrs; `dims` |
| Suite2p `ops.npy` | `fs` `dx` `dy` `dz` `umPerPixX/Y/Z` `pixel_resolution` `z_step` `Ly` `Lx` `nframes` + every timepoint alias, `raw_file`/`chan2_file`, `align_by_chan`; images and per-frame vectors restored to ndarrays (`normalize_ops_arrays`) |

`imwrite` records each write in `metadata["processing_history"]` via
`add_processing_step` (step, timestamp, version, inputs, outputs, duration, extra).

### 6.8 Export hygiene

- `strip_for_export(md)` runs before any TIFF/H5/Zarr stamp. It drops
  `EXPORT_DENYLIST` (suite2p registration internals, summary images, per-frame
  vectors, pipeline settings, mbo-internal keys) and any value over 8192 elements
  unless allow-listed. Suite2p-only fields live only in `ops.npy`.
- `rebase_provenance_paths(md, anchor)` repairs or drops recorded paths
  (`PROVENANCE_PATH_KEYS`) that do not exist on this machine. Never re-embed another
  machine's paths.
- `plane_shifts` / `plane_shifts_params` survive export on purpose; viewers align
  planes from them at read time.

### 6.9 Display

The GUI metadata viewer (`gui/_metadata.py`) reads canonical values through the
registry (`label`, `unit`, `description`) and groups the rest under vendor
namespaces. The metadata editor writes canonical keys only.

## 7. Pipelines

A pipeline is a processing workflow (suite2p, MaskNMF, voltage, isoview, ROI
workflow) that reads a `LazyArray` selection and writes an output directory. Every
pipeline touches five surfaces; the registration contract below is what makes them
one thing.

| Surface | Where | Purpose |
|---------|-------|---------|
| `PipelineInfo` | `pipeline_registry` | name, description, category, input/output patterns, marker files; drives file dialogs and `mbo info` |
| `PipelineWidget` | `gui/widgets/pipelines/<name>.py` | Run-tab config UI; declares availability, applicability, per-axis consumption |
| worker task | `gui/tasks.py::task_<name>(args, logger)` | re-opens the source in a subprocess and calls the runner |
| runner | `mbo_utilities/<name>/runner.py` (`run_plane`, `run_volume`) | the computation; GUI-free |
| CLI / HPC | `mbo <name>`, `mbo hpc run` | the same runner from a terminal or SLURM |

### 7.1 Package shape

```
mbo_utilities/<name>/
├── __init__.py   # lazy re-exports only
├── params.py     # settings dataclasses: GUI-free, JSON round-trip, defaults = the reference run
├── runner.py     # run_plane(arr_or_path, save_path, settings, ...) / run_volume(...)
├── outputs.py    # convert results to suite2p-shaped sidecars
└── qc.py         # figures (optional)
```

- Settings are nested dataclasses with `to_dict()` / `from_dict()`; stage tri-states
  use the suite2p convention `0` skip, `1` run, `2` force; each stage gates on its
  own native output file plus a settings hash, so a re-run with cached stages only
  redoes exports and figures.
- Heavy or optional imports (`masknmf`, `vnoiser`, `suite2p`, `torch`) are
  function-local inside the runner.
- Provenance: settings and source metadata (after `strip_for_export`) are written next
  to the outputs (`pipeline.json`, `mbo_provenance` attr) and `processing_history`
  is appended.
- Timing: the runner owns its step loop and closes every step (a plane, a scan's
  read, a domain's denoising, each write) with one INFO line carrying wall time, CPU
  time and process memory, one `processing_history` entry
  (`add_processing_step`, `duration_seconds`) and a `timing` summary (totals per
  step, per unit, peak memory) in the provenance; `timings.json` beside the outputs
  holds the same with one flat row per step. The voltage runner
  (`vnoiser/pipeline.py::_RunUsage`) is the model; suite2p's `plane_times` and
  `hpc.write_timing_report` are the same record. Progress goes through
  `progress_callback(fraction, message)`, never a heartbeat line.

### 7.2 Registration contract

One class registers the whole pipeline. Subclass `PipelineWidget` and declare:

```python
class VoltagePipelineWidget(PipelineWidget):
    name = "Voltage"                       # selector label; unique
    is_available = HAS_VNOISER             # bool or property; cheap
    install_command = "uv pip install vnoiser"
    info = PipelineInfo(name="voltage", category="processor", ...)  # patterns + marker files
    axes_consumed = {"T": "range", "Z": "all", "C": "select-one"}  # see AXIS_MODES
    task_type = "voltage"                  # key in gui.tasks.TASKS
    task_func = staticmethod(task_voltage) # task_<name>(args, logger)

    @classmethod
    def applies_to(cls, arr) -> bool: ...  # safe before instantiation; arr may be None
    def draw_config(self) -> None: ...
```

and list it in `pyproject.toml`:

```toml
[project.entry-points."mbo_utilities.pipelines"]
voltage = "mbo_utilities.gui.widgets.pipelines.voltage:VoltagePipelineWidget"
```

`load_entry_point_pipelines()` registers `info`; `load_plugin_tasks()` registers
`task_type` → `task_func`; the Run tab lists the class. Built-in and third-party
pipelines use the same path; nothing is hardcoded by name.

- `axes_consumed` values: `range` (user picks start:stop), `all` (whole axis, row
  disabled), `none` (axis hidden), `select-one` (single index). Default is `range` for
  T, Z, C.
- `PipelineInfo.category` is one of `reader`, `writer`, `processor`, `segmentation`.
  Readers register their own `PipelineInfo` at import (category `reader`); a
  pipeline's info lives on its widget.
- `extracts_traces = True` + `extract_traces(movie, labels)` opts the pipeline into
  the manual-ROI "Extract trace" action.

### 7.3 Input contract

A worker task receives a JSON dict and must be able to rebuild exactly what the user
was looking at:

| Key | Meaning |
|-----|---------|
| `input_path` | `arr.source_path` |
| `reader_kwargs` | `source_reader_kwargs(arr)`: `unit`, `dataset`, `dims`, `frame_average`, `channel` |
| `timepoints`, `planes`, `channels` | 1-based selection from `to_lsp_kwargs` |
| `fix_phase`, `use_fft`, `phasecorr_method`, `mean_subtraction` | read features, applied by `apply_read_features` |
| `output_path` | output directory |
| `settings` | `Settings.to_dict()` |
| `_uuid`, `_log_file` | injected by `ProcessManager.spawn`; read by `TaskMonitor` and `setup_logging` (§8.3) |

The selection keys keep the 5D names whatever the axis means for the source; the
runner translates through the array's metadata, never the widget. On a MESc AOD unit
Z is the ROI index (`mesc_z_axis_meaning == "roi_index"`), so `planes` are the lines or
patches to process: the voltage runner reads only those ROIs
(`linescan_roi_read(rois=...)`), cuts every domain down to them, drops a domain left
empty, and records `planes` in the provenance source block. The Voltage tab's slice
popup labels the row "ROIs" but still sends `planes`.

The worker does `arr = imread(input_path, **reader_kwargs)`, then
`apply_read_features(arr, args)`, then calls the runner. Runners take a `LazyArray`
or a path and the same 1-based selection kwargs; they never take GUI objects.

### 7.4 Output contract

Outputs are suite2p-shaped so `Suite2pArray`, the QC figures, the diagnostics and
summary widgets, and `mbo info` load them unchanged:

```
<save_path>/
  zplane01_tp00001-01574/
    ops.npy  stat.npy  iscell.npy  F.npy  Fneu.npy  spks.npy
    data.bin (registered)  data_raw.bin (optional)  scanphase.npy
  zplane02_tp00001-01574/
```

Filenames are matched; semantics may differ and the pipeline wins (MaskNMF writes
zeros for `Fneu`/`spks`). Anything pipeline-specific keeps its own name
(`demixing_results.hdf5`, `PF/`, `norm_traces.npy`).

`hpc/` runs the suite2p pipeline only (`lbm_suite2p_python.pipeline`), configured by
`hpc.toml` (`[io]`, `[slurm]`, `[pipeline]`, `[parameters]`); a second pipeline gets
HPC support by exposing `run_volume(arr, save_path, **selection)` and a `pipeline`
key in the config.

### 7.5 The results zarr

Native outputs differ per pipeline (§7.4's suite2p files, the voltage pipeline's `PF`
pickles). `mbo_utilities/results.py` fixes the one shape they all mold into: a zarr
v3 group, `<input stem>.<yyyy-mm-dd-HH-MM-SS>.<pipeline>.zarr`, that a reader, a
viewer or a notebook opens the same way whichever pipeline wrote it. It is the standard output format; a
pipeline's native files stay its cache and its compatibility layer.

```
<stem>.<stamp>.<pipeline>.zarr/      zarr v3 group; attrs: mbo_results (schema version),
                                     pipeline, created, tags, units, source, settings,
                                     metadata (after strip_for_export), provenance
  <unit>/                            one group per plane (zplane01) or scan (scan35)
    attrs: kind, index, fs, n_rois, n_timepoints, roi_names, member_kind, image_shape,
           + whatever the pipeline adds (scan_id, source_unit, plane_dir, ...)
    traces/<kind>                    (n_rois, n_timepoints) float32; kinds: raw, neuropil,
                                     dff, zscore, denoised, spikes (TRACE_KINDS)
    rois/offsets  rois/member  rois/weight
                                     ragged membership: ROI k is member[offsets[k]:offsets[k+1]];
                                     pixels as flat y * X + x (member_kind "pixel") or line
                                     indices ("line"); weight is suite2p's lam or 1.0
    rois/iscell                      (n_rois, 2) float32
    members/<kind>                   (n_members, n_timepoints) the members' own traces when
                                     they have them (a line scan's lines)
    events/frame  events/roi         detected events (peaks), sorted by ROI
    images/<kind>                    (Y, X) float32; kinds: mean, max, corr, ref (IMAGE_KINDS)
  <pipeline>/                        the run's own files, in a folder named after the
                                     pipeline that wrote it (`pipeline_files`):
                                     pipeline.json, timings.json, its native h5 and npy,
                                     traces/
```

- **Naming.** `results_name(source, when, extra_tags, pipeline)` is the source
  filename's stem, then `extra_tags`, then the timestamp, then the pipeline, dot
  separated: `session1.mesc` → `session1.2026-09-21-14-30-22.voltage.zarr`, written
  beside its input. The stamp is local time to the second (`RESULTS_STAMP`), so a
  rerun is a new file and a listing sorts chronologically. Nothing matches a
  results name by pattern: `results_stamp(path)` reads the stamp back with
  `datetime.strptime` and `newest_results(folder, pipeline)` picks the latest run.
  Unit groups keep the §5.6 vocabulary (`unit_name("plane", 1)` is `zplane01`;
  scans are `scan<id>`).
- **Molding.** A pipeline builds one `ResultUnit` per plane or scan and calls
  `write_results(path, units, pipeline=..., source=..., settings=..., metadata=...)`.
  `results_from_suite2p(dir)` molds suite2p and MaskNMF folders (`F` → `raw`, `Fneu` →
  `neuropil`, `spks` → `spikes`, `norm_traces` → `dff`, `stat` → pixel members,
  `meanImg`/`max_proj`/`Vcorr`/`refImg` → images); `results_from_pf(dir)` molds the
  voltage pipeline's `PF` folder (domains are the ROIs, their lines the members with
  `members/raw`, `test.h5` gives `dff` and `zscore`, peaks are the events). Copy one of
  them for a new pipeline; never invent a trace or image kind, add it to the registry.
- **Writing.** It is the voltage pipeline's default
  (`VoltageSettings.runtime.output_format == "zarr"`, the Run tab's Output format):
  the run works in a `<name>.work` scratch folder, writes the results file beside the
  input, moves `test.h5`, `traces/`, `pipeline.json` and `timings.json` into its
  `voltage/` (`pipeline_files`) and deletes the scratch folder, so one path is the
  whole output and no `PF` folder is left. `mbo voltage --pkl` (`output_format ==
  "pkl"`) writes the archive's PF folder of pickles instead. `mbo results <dir>`
  converts an existing suite2p, MaskNMF or PF folder.
- **Reading.** There is one reader and it is generic. `read_results(path)` returns
  `Results` (`.units[name]` → `ResultUnit`, every array in memory); `mold_results(dir)`
  turns a native output folder into the same object in memory without writing
  anything, and `open_results(path)` is the one door that takes either. `imread`
  returns a `ResultsArray` (`results.py`, `PRIORITY` 70) for a results file, a
  voltage `PF` folder, or a folder holding one; `results_dir_of(path)` is what it
  resolves with. No pipeline gets an array class of its own: a run's units, traces,
  ROIs, events and images are on `arr.results`, and the recording it processed is
  the image when `results.source` names a reachable file, else a trace raster.
  `results_pipeline(path)` and `results_summary(path)` read only `zarr.json` files
  and are what `can_open` and the run scanners use; `ZarrArray` declines a results
  file. `imread` never returns a results file as a movie of itself.
- **Viewing.** The ROI widget's Traces tab takes a results file through the same door
  as a run dir: `ManualRoiWidget.load_run(path)` (a file, or one unit as
  `<file>.zarr/zplane01`) calls `load_results`. A pixel unit becomes a `RunResult`
  (`roi_workflow.run_result_from_unit`) and loads as a derived set with its overlay,
  exactly like `stat.npy` + `F.npy`; a line unit becomes an `external` `TraceSet` with
  one row per ROI (its `denoised` trace, else `dff`, else `raw`) and one per member
  line, every row named by its ROI (`roi3`, `roi3 (raw)`; a line of a multi-line
  ROI adds itself, `roi3 line 12 (raw)`) and carrying its `fs`, the line it was read
  from on `z` (`extra["line"]`) and the pipeline's channel on `c`.
  `roi_runs.run_dir_complete` and
  `scan_run_dirs` treat results files as run dirs, `roi_runs.json` restores them, a
  finished `voltage` worker is adopted like a suite2p one, and the Voltage tab's
  "Load into Traces" button does it on demand. A new pipeline that writes the results
  zarr therefore reaches the Traces tab with no GUI code.

Pinned by `tests/test_results.py`, `tests/test_voltage_pipeline.py`.

### 7.6 Manual ROIs

Hand-drawn ROIs are a session model in `mbo_utilities/annotation/`, GUI-free and
observable (`events.Observable`, after fastplotlib's `GraphicFeature`: a view
subscribes with `add_event_handler(fn, "rois")` and redraws from the event). The
ROI widget (`gui/manual_roi.py`) and the Process tab's ROIs pipeline
(`gui/widgets/pipelines/rois.py`) are two views of one `RoiModel`; neither polls
the other.

| Object | Holds | Emits |
|--------|-------|-------|
| `RoiLabelStore` | the `(P, Y, X)` uint16 label volume, one `RoiRecord` per ROI (`plane`, `area`, `class_index`, `note`, `uid`, `source`, `color`), the label-name set, `plane_axes` | `rois` with `action` `add` `delete` `clear` `class` `note` `color` `labels` |
| `RoiTraceTable` | one `RoiTrace` per measurement | `traces` with `action` `add` `remove` `clear` |
| `RoiModel` | the two above plus the slider position (`view`, `plane`, `z`, `c`) | the two above forwarded, plus `view` when the plane changes |

- **Planes.** `RoiRecord.plane` is the flat index into the volume. `plane_axes`
  records which scrolling dims key planes (`(("c", 2), ("z", 3))`, z last, so a
  z-only store keeps `plane == z`); the store owns the arithmetic (`plane_of(pos)`,
  `plane_pos(plane)`, `plane_label(plane)`) and decodes an ROI's `roi_z(i)` /
  `roi_c(i)`. Axis roles resolve through `find_slider_name`, so IsoView's `Cam` /
  `Zplane` sliders key planes too. T never keys a plane.
- **Traces.** A drawn ROI's trace is keyed `("roi", uid, z, c, engine)`: the mask,
  where its pixels were read, how (`ENGINES` = `mean`, `suite2p`, `masknmf`). Running
  the same measurement again replaces the row; reading the same mask on another
  channel, z-plane or with another engine is another row. Rows that stand for no
  drawn ROI (an algorithm's component, a results file's line) are keyed
  `("member", source, k)` and are never pruned by ROI deletion. `frames` is the
  `(start, stop)` window read, `frame_average` the binning, `source` the run or origin.
  `RoiTrace.name` is a row's one display name (`label`, else source and member); the
  table, the legend and the sort key read it, nothing formats its own.
- **Trace display.** `annotation/display.py` says how a row is shown, and the
  pipeline that produced the row decides. One `TraceProfile` per pipeline
  (`TRACE_PROFILES`, keyed by the row's `engine`, which a results file sets to its
  `pipeline`; a plugin calls `register_trace_profile` beside its `PipelineInfo`;
  anything else gets `DEFAULT_TRACE_PROFILE`) declares the kinds its rows can show
  (`DISPLAY_KINDS`, the results zarr's `TRACE_KINDS`), the one shown first
  (`voltage` opens on `denoised`, the rest on `dff`), whether a neuropil correction
  is offered (only a pipeline that measured a real `Fneu`: `suite2p`, and `mean`'s
  ring; never `masknmf`, whose `Fneu` is zeros), the `DffSettings` for a dF/F
  computed from a raw row (`analysis/dff.py`: a rolling max-min baseline sized in
  seconds, the percentile when the row has no `fs`), whether a stored dF/F is
  percent or a fraction, and the raw trace's label. A row carries `F`, `Fneu`,
  `norm` and every other kind in `kinds`, all read by `RoiTrace.array(kind)`. The
  panel and the trace table read every row through
  `display_trace(trace, kind, settings, neuropil)`; `displayed_kind` says which kind
  that was (a kind the row lacks falls back to its profile's default) and `y_label`
  names the axis for it. The panel offers the kinds the plotted rows have, shows the
  neuropil checkbox only when a plotted row's profile offers it, a dF/F settings
  popup only when a shown dF/F is computed here, labels the y axis from the rows
  (joined when they differ) and opens in seconds whenever the data has a rate.
  The plot has no box of its own: `imgui/lines.plot_style` makes implot's frame,
  plot background and border transparent, its grid lines invisible (by colour,
  so one scope covers the subplots too) and its ticks and legend dim, so the
  traces sit on the panel. implot takes its frame colour from imgui's, which
  `style_imgui_opaque` makes a blue-grey, so a plot drawn without that scope
  sits in a blue box. One scope wraps the trace and motion plots together,
  a frame around either being a box around half the panel.
- **Run coordinates.** `RoiModel.targets(indices, z=, c=)` says where each ROI is
  read: the mask always from the plane it was drawn on, the pixels from `z` / `c`
  when given, else from where it was drawn. The widget's `run_where` is `drawn`,
  `screen` (the slice the sliders show when the run starts) or `fixed` (`run_z`,
  `run_c`); `run_frames` cuts T (`PlaneMovie.window`). `run_rois` groups targets by
  `(plane, z, c)` and writes one `rois_<tag>/` child per read (`zplane02`,
  `zplane02_ch01`) when there is more than one. Every run records `plane`
  (store plane), `z`, `c`, `frames` and `engine` in `ops["roi_workflow"]` and
  `plane` / `z` / `c` per row in `rois.json`; `RunResult.read_z` / `read_c` /
  `frames` / `engine` read them back and `roi_runs.result_traces` turns a result
  into table rows.
- **Draw -> run.** `ManualRoiWidget(auto_trace=True)` traces every ROI the moment it
  is drawn (its mean at the run coordinates); no tab is selected for the user. The row
  buttons on the ROIs tab, the `t` key and the Process tab all run one ROI the way
  the Process tab is set (`engine`, `run_where`, `run_frames`, `run_tag`).
- **The ROIs pipeline** (`RoiPipelineWidget`, name `ROIs`, `axes_consumed`
  `T: range, Z: select-one, C: select-one`) applies to any array with a time axis.
  It picks which ROIs (selected / group, listed, this slice, all), where they are
  read, the engine and tag, runs them, and shows the trace table cut down to those
  ROIs (`ManualRoiWidget.draw_trace_table(keys=...)`). Region and full-plane
  detection live there too. It turns Manual ROI Labeling on when it is off. The
  ROIs tab keeps only NAVIGATE, DRAW (with the region tool and the trace-on-draw
  switch), VIEW and LABELS, as sections over its table; the trace plot and its
  controls stay a panel on the top strip.
- **Any slice.** The viewer's sliders are the array's T, C, Z axes by position
  (`manual_roi.slider_roles`), whatever the array labels them (`Timepoint` /
  `Channel` / `ROI` for a MESc AOD unit, `Tile` / `View` for IsoView); the widget
  hands the store `axis_roles = {"z": <slider>, "c": <slider>}` so `roi_z` / `roi_c`
  and every run decode the right slice. Every `LazyArray` is indexable in y and x
  by the 5D contract (§5.3), and `PlaneMovie` reads only `arr[t, c, z, y, x]`, so
  no reader needs anything more for the ROI tool; whether a read touches only the
  bounding box is the reader's laziness, not the contract. Wherever the widget
  names an axis (trace legends, table headers, tooltips) it uses the slider's own
  word (`ManualRoiWidget.axis_label`): on an AOD unit the Z axis reads `ROI 3`,
  never `z3`, because R is not Z (`mesc_z_axis_meaning == "roi_index"`).
- **Line positions.** Where a MESc scan line or patch sits is
  `arrays/mesc_geometry.py`'s business and nothing else re-derives it: the segment
  the AOD scanned (`CoordinateMapJSON.maps[0].driftEndPoints`, or `contours` for
  patches; never the shorter hand-drawn `guideLine`, never `ROIJSON`, whose vertices
  are in the unrolled scan's pixels) in the file's absolute micron frame, where
  `ReferenceViewportJSON.geomTransTransl` is the `[0, 0]` corner of an image, rows
  grow with +y, and a stack's slice `k` sits at `geomTransTransl[2] + MinZ + k·step`
  (verified on two rigs: brightness ranking of the lines on their own snapshot, and
  cross-correlation of background frames against a stack). `line_positions(path,
  unit)` gives each ROI its `start_um` / `end_um`, `z_um`, `length_um`, `sample_um`
  and `dz_um` against the snapshot the lines were drawn on; `roi_placements` puts
  them on a stack's slices; `image_overlays` draws them. An AOD unit's rows of the
  trace table (`linescan_viewer.StandardTraces`, for every `ROI_LAYOUTS` layout:
  the mean of each line or patch, computed in the background unless `F.npy` or PF
  traces exist) are keyed `("member", "<MUnit> lines", k)` or `("member", "<MUnit>
  patches", k)`, carry `z = k` (the unit's ROI axis), `c` = the channel read, and
  that position in `extra` (`line`, `z_um`, `dz_um`, ...); the label is the ROI
  alone. Depth is never a column, a label or a caption: the GUI shows where a
  line was drawn (the reference image), not how deep it sits. Hovering a trace
  row lists the line's ends, length and sample spacing
  (`ManualRoiWidget._line_position`: the row's `extra["line"]`, else its `z` on
  an AOD unit, indexes the recording's `MescArray.line_positions`, the reader's
  cached facet over `mesc_geometry.line_positions`, under the row's own `extra`,
  `POSITION_KEYS`; a results file's line rows get theirs stamped at load when
  the shown recording is their scan).
- **Reference image.** `gui/mesc_reference.py` shows where an AOD unit's lines
  or patches were drawn: `reference_images(mesc, open_unit, c)` (GUI-free) is the
  picture the unit's ROIs were drawn on (`background_unit`), then every Z-stack
  holding them (`zstack_contents`), each a max projection in channel `c` (at most
  `MAX_ELEMENTS` samples read), carrying the unit's `image_overlays` records. A
  picture projects over its frames; a Z-stack projects **only over the slices its
  ROIs were scanned on**, never the whole stack, whose full max is a wall of
  tissue with the ROI's own plane lost in it. `ReferenceView` is that set in a
  `SummaryImageViewer` popup (masknmf's full-FOV viewer, `roi_provider(key)`
  returning `(points, rgba, thickness)` polylines: MESc's colours, every ROI
  dimmed to `ON_ALPHA`, the slider's ROI drawn last, opaque, at
  `SELECTED_THICKNESS` over a white `HALO_THICKNESS` halo); its caption says how many of the
  unit's ROIs a Z-stack holds when some were scanned outside it. Each
  `ReferenceImage` carries its `unit` and, for a stack, the `slice` its ROIs sit
  on, so the popup's one button (`on_show(unit, slice)`) displays that unit in
  the viewer at that slice. The MESc tab opens the popup from the picture cell's
  single button, redraws it from the top strip's hook, reopens it on a unit
  switch and hands the host `reference_view` (a callable), which the Traces panel
  offers as a button. A click on a line in it (`SummaryImageViewer.on_pick`,
  `ReferenceView.pick`) moves the ROI slider to it. Nothing draws ROIs on the
  viewer's own image, and no projection of the unit itself is offered.
- **A stack holds a scan only where it was scanned.** `image_overlays` on a
  Z-stack keeps an ROI only when its outline falls inside the stack's field
  **and** its depth range (`roi_placements`' `in_range`); one scanned above or
  below is left out rather than clamped onto an edge slice, which drew outlines
  on tissue the scan never touched (2026-09-14 rig: one chessboard box 720 um
  under the only stack, landing on slice 0 over unrelated cells). `on_plane` on a
  stack is therefore always True, `zstack_contents` lists only stacks that really
  hold a scan, and `line_positions` reports `stack`/`slice`/`in_stack` as None
  when no stack was scanned around the line. The micron-to-pixel mapping itself
  is verified: the 2026-09-14 pictures cross-correlate 0.86 against the stack
  slice at their own depth, peaking exactly at zero offset.
- **The MESc table.** One row per recording: a scan's picture
  (`background_unit`) and RTMC reference unit (`rtmc_unit`) fold into its row
  (`mesc_units.companions`). The `picture` cell is **one** button, naming the
  picture behind an `IMAGE_ICON`, and it opens the reference image; there is no
  second button and no popup, and the table has no Z-stack column, because
  everything about where a scan sits belongs next to the lines drawn on it.
  The `RTMC` cell is `yes` or `no` (`rtmc_on`: the scan carries RTMC curves,
  `list_mesc_units`' `rtmc` or `rtmc_armed`), the detail on hover. The popup's
  display button reaches the tab through `_show_reference_unit`, which parks the
  request on `_pending` for `_frame` to apply once the popup has drawn: a switch
  rebuilds the panel widgets, so it never happens inside another widget's draw.
  `_install(arr, z)` sets the viewer's `roi_slider` index after the swap, so a
  Z-stack opens on the tissue the ROIs were scanned in. Every header carries its
  meaning (`COLUMN_HELP`) and `?` opens `assets/docs/mesc.md`, the plain-words
  page on what a `.mesc` holds.
- **Full image.** The ROIs pipeline's `full image` target is the whole frame as
  one mask at the run coordinates: with `mean` a `FULL_IMAGE` row of the trace
  table (`ManualRoiWidget.trace_full`, keyed `("member", "full image", "z<z>c<c>")`
  so the same slice replaces itself); with `suite2p` / `masknmf` a full detection
  of that z-plane and channel (`run_full_plane`, whose worker args carry `channel`
  1-based and `tp_indices` for a frame window). No store mask is involved, so it
  never claims pixels.
- **Playhead.** `gui/playhead.Playhead` is the one time on screen, in seconds on the
  recording's clock (raw frames when `fs` is unknown); it emits `time` with its
  `source`. Every view keeps a `TimeAxis` (`per_second`, `offset`) and converts
  through it: the viewer's T slider (`viewer_axis`, `fs / frame_average`), each trace
  row (`trace_axis`: its own `fs` else the movie's, its `frame_average`, its frame
  window as the offset), the plot's unit (`plot_axis`: frames, seconds or ms).
  `TimeAxis.on(other)` gives the `(xscale, xstart)` a row is plotted with, so a
  windowed trace sits where it was recorded. The trace plot and the motion plot
  seek the playhead; the widget's handler moves the viewer's T, whose indices event
  snaps the playhead to the frame. The ROI widget shares its host's playhead
  (`PreviewDataWidget.playhead`); the line-scan viewer's `LineScanOverlay` owns one
  for its Timepoint slider, kymograph selector, trace and motion cursors. A new
  time-bound view subscribes to the playhead; it never reads another view's cursor.
- **Color by.** `RoiModel.column(name)` gives one number per uid (`plane`, `z`, `c`,
  `area`, `class`); `RoiModel.colorize(values, cmap, categorical)` maps them through
  a `cmap` colormap into `RoiLabelStore.tint`, a display-only color per uid that
  `roi_rgb` prefers while set (never saved; `set_tint(None)` clears). VIEW > color
  by drives it (`ManualRoiWidget.set_color_by`; `peak` uses the trace table) and
  reapplies it as ROIs, labels or traces change, so the overlay, the table and the
  trace legend agree without any view knowing why.
- **Shared vocabularies.** Nothing in the ROI work spells a name or a selection
  of its own: run children and full-image row keys are `OutputFilename` tags in
  T, C, Z order (`ch01_zplane02`; `build("")` names a folder), a single ROI's run
  is the R tag (`rois_roi02` for store index 1, 1-based like every tag), a full
  plane's dir is `results.unit_name("plane", n)`, and a run dir's per-slice child
  is recognised with `filename_tags`, never a regex. The frame selection is the
  string every Save As and pipeline row takes (`parse_timepoint_selection`,
  1-based `start:stop:step,exclude`), kept as the 0-based `run_tp` index list the
  worker tasks call `tp_indices`; `PlaneMovie.select(indices)` reads it (`window`
  is the contiguous case), `_slicing.index_window` says whether it is a
  `(start, stop, step)` window (stamped as `frames` on rows and runs) or a gapped
  list (stamped as `tp_indices`), and `_ops_for` scales `fs` for a stride through
  `OutputMetadata` (§6.6). 1-based plane / channel worker args come from
  `to_lsp_kwargs`. Slider roles come from `_dim_labels.slider_roles` (positional,
  the same table the viewer names its sliders from), which `resolve_dim_labels`
  and the ROI widget both use, so every slice popup and the ROI tool call an AOD
  unit's Z axis `ROI`.
- **Persistence.** The store autosaves to `manual_labels.zarr` (`plane` per ROI;
  older stores wrote `z` and load unchanged). Run rows reload from `rois_<tag>/`
  dirs through `roi_runs.json`; quick traces, full-image rows and tints live for
  the session. A file holding several recordings (a `.mesc`) keeps all of it per
  recording: the widget's `unit_key` (`manual_roi.unit_key`, the base array's)
  names the sidecars `manual_labels_<tag>.zarr`, `roi_runs_<tag>.json` and
  `rois_<tag>_<run>/` (`labels_path(fpath, tag)`, `registry_path`, `run_prefix`,
  tag `MSession_0_MUnit_3`), and a unit switch (`MescTabWidget._install`) parks
  the outgoing widget's store and runs under its unit and rebuilds the widget for
  the incoming one, so the ROIs, runs and traces on screen are the shown
  recording's.

Pinned by `tests/test_roi_model.py`, `tests/test_playhead.py`, `tests/test_trace_display.py`, `tests/test_manual_roi.py`
(`TestRunCoordinates`, `TestAutoTrace`, `TestRoiPipelineTab`, `TestPlayheadWiring`,
`TestColorBy`, `TestFullImage`, `TestSliderRoles`), `tests/test_roi_runs.py`.

## 8. Logging and the Process Console

One logger tree, one console sink per process, one log file per background task.
`logging.md` at the repo root is the audit this section is based on.

### 8.1 Loggers

- Every module gets its logger at import from `mbo_utilities.log`:
  `logger = log.get("arrays.zarr")` is `mbo.arrays.zarr`. Names follow the module path
  under `mbo`; `log.get()` with no name is the `mbo` root. Never
  `logging.getLogger(...)`, never `print()` in library code; `print` and `click.echo`
  are for CLI command output only.
- `log.py` owns the handlers. The root `mbo` logger carries the one console
  `StreamHandler`; `log.attach(handler)` adds a process-wide sink (the GUI panel, the
  worker file); `log.set_global_level(level)` sets every `mbo*` logger;
  `log.get_package_loggers()` lists them. Library code never adds a handler.
- Level: `MBO_DEBUG` selects DEBUG, else INFO, at import. `log.set_debug(enabled)`
  is the one way to change it afterwards: it writes `os.environ["MBO_DEBUG"]` (so
  spawned workers inherit it) and calls `set_global_level`, and `log.debug_enabled()`
  reads it back. `mbo --debug` / `mbo view --debug` set it for a run; the GUI
  "Debug logging" toggle (`_options_popup`, `file_dialog`) also persists the
  preference, which `run_gui` applies at launch unless `MBO_DEBUG` is already set.
- The Widgets menu (`gui/widgets/widget_toggles.py`) is one checkbox per
  `WidgetEntry`, no submenus: an entry shows or hides the whole widget, and a
  widget's `toggle_key` is an entry's key. Below the checkboxes it opens the
  floating tool windows, always available (no debug gate): Style Editor (§14),
  ImGui Debugger (`gui/widgets/imgui_debug.py`: `imgui_debugger`'s variable
  inspector over the preview window, and switches for Dear ImGui's
  metrics/debugger, debug log, ID stack tool, demo and about windows, drawn from
  `PreviewDataWidget.draw` every frame), BioHPC and Cloud.
- The GUI's Debug panel (`gui_logger.GuiLogger`) receives every `mbo.*` record through
  a `GuiLogHandler` attached in `preview_data._init_logging`; it filters by level and
  logger, and its master level dropdown calls `set_global_level`.

| Level | Use |
|-------|-----|
| DEBUG | per-chunk or per-frame detail, dispatch decisions, cache hits |
| INFO | one line at the start and end of a stage: what, shape, path, elapsed |
| WARNING | a recovered anomaly: fallback taken, stale alias, skipped file. Deduplicate repeats (`metadata/params._STALE_WARNED` pattern) |
| ERROR | `logger.exception(...)` once, immediately before re-raising or `monitor.fail` |

No logging above DEBUG inside per-frame loops. tqdm bars are for terminals; the
worker sets `TQDM_DISABLE=1` and collapses `\r` redraws to one line per bar.

Target wiring (the fix for `log.py:22`, tracked in §15): `log.get` leaves
`propagate=True`; the root `mbo` logger is the only logger with handlers;
`log.attach` adds to the root; the worker's `setup_logging` swaps the root console
handler for its rotating file handler, because its stdout and stderr already point at
the same file and a second stream sink duplicates every line. Then every `mbo.*`
record reaches the terminal, the Debug panel and the per-task log without bridges.
Until that lands, an `mbo.<sub>` message is visible only inside the GUI (which
attaches handlers to every child) or when written through the `logger` a task
receives; in scripts and workers, child INFO goes nowhere and WARNING+ leaks bare
through Python's `lastResort` handler.

### 8.2 Progress is not logging

- A long operation takes `progress_callback(fraction, message)` and calls it per
  chunk. It does not log per chunk.
- A worker task forwards progress to `TaskMonitor(output_dir, uuid)`, which writes
  `~/.mbo/logs/progress_{uuid}.json` atomically:
  `{pid, uuid, timestamp, status: running|completed|error, progress, message, details}`.
  A task ends with exactly one of `monitor.finish()` or `monitor.fail(error, details)`;
  `_worker.main` writes the same sidecar as a backstop (`completed` on return,
  `error` with the traceback on an exception, after logging memory at failure).
- A single blocking call that cannot report progress runs a heartbeat thread
  (`task_masknmf._heartbeat`, `_voltage_heartbeat`) so the watchdog sees activity.
  The watchdog terminates a worker after 120 minutes with no progress change and no
  log-file mtime change.
- Work on a GUI thread (ROI traces) has no pid or log file; it registers a
  `LocalJob` (`pm.start_job`, `set_progress`, `done` / `fail`) so it still shows in
  the console.

### 8.3 How a pipeline run reaches the Process Console

1. A widget calls `ProcessManager.spawn(task_type, args, description, output_path)`.
   `spawn` mints `task_uuid`, sets `args["_uuid"]` and
   `args["_log_file"] = ~/.mbo/logs/{YYYYmmdd_HHMMSS}_{task_type}_{uuid[:8]}.log`,
   writes `args_{uuid}.json` (the JSON can exceed the Windows command-line limit),
   and launches `python -m mbo_utilities.gui._worker <task_type> <args_file>`
   detached (`pythonw` + `CREATE_NO_WINDOW` on Windows, `start_new_session`
   elsewhere) with **stdout and stderr redirected to the log file**, `PYTHONUTF8=1`,
   and the GUI's `MBO_DEBUG`. It records
   `ProcessInfo(pid, description, task_type, output_path=<log file>, args)` in
   `~/.mbo/cache/running_processes.json`, so the process is still tracked after a
   GUI restart (entries older than 24 h are dropped).
2. `_worker.main` reconfigures stdio (utf-8, line-buffered, `_CRCollapseWriter`),
   loads and deletes the args file, and calls `setup_logging(log_file)`: the
   `mbo.worker` logger gets a `RotatingFileHandler` (5 MB, two backups) and every
   handler gets the unified format
   `%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s`. It then contains
   children in a Windows Job Object, starts the watchdog and the memory monitor
   (`mem_{uuid}.csv`), resolves `TASKS[task_type]` (loading entry-point tasks when
   missing), and runs `task_func(args, logger)`.
3. The log file therefore receives `mbo.worker` records, the root `mbo` console
   handler (stderr), everything the pipeline or its dependencies print (suite2p,
   tqdm's final line), and bridged third-party loggers. A third-party logger is
   bridged with a `_ForwardingHandler` that mirrors records onto the worker logger's
   handlers and sets `propagate=False` (`_bridge_isoview_logging`); never attach the
   root console handler to it, that writes every line twice.
4. The console (`gui/_popups.py`, opened from the menu-bar status button) calls
   `pm.cleanup_finished()` and `pm.get_running()` every frame. Each `ProcessInfo`
   re-reads its sidecar (`progress_{uuid}.json`, falling back to
   `progress_{pid}.json`, identity-checked) for status, progress and message. A dead
   process with no terminal status becomes `error: Process crashed unexpectedly`.
   Completed rows stay five minutes; errors stay until dismissed.
5. Each row shows `[...]` / `[OK]` / `[ERR]`, the description, `PID · elapsed`, Kill
   (alive; terminates the process tree) or Dismiss, Copy (the whole log file to the
   clipboard), the error message, and a collapsible Log Output that tails the last
   500 lines of the log file and follows it while pinned to the bottom. The same
   registry backs `mbo processes [--kill PID | --kill-all | --cleanup]`.
6. `prune_logs()` runs when the manager starts: sidecars, args files and `.tmp`
   older than 24 h are deleted; the newest 25 logs are kept
   (`MBO_LOG_KEEP_RECENT`) and the rest deleted after 14 days
   (`MBO_LOG_RETENTION_DAYS`).

### 8.4 HPC

SLURM jobs never see the GUI. `mbo hpc run` (submitit) streams job stdout and stderr
to `<output_dir>/logs/*.out` and `*.err` (never `$TMPDIR`, even with node-local
staging); `write_timing_report` writes `<output_dir>/timings.json`;
`write_failure_report` writes `FAILURE_{role}_{task}_{pid}.log` beside the run and a
rotated copy under `~/.mbo/logs` (last 20). `record_run` keeps the last 10 launches
in `~/.mbo/hpc/runs` so `mbo hpc watch` and `mbo hpc status` with no argument find
the latest run; `watch` follows `.err` by default (`-o` for `.out`; `o` `e` `n` `p`
`q` while following). Compute-node narration belongs in `log.get("hpc.pipeline")`
with a timestamp and the job and task id, not `print`; `print` and `click.echo` stay
for the `mbo hpc` command output itself.

### 8.5 Files

| File | Written by | Read by |
|------|------------|---------|
| `~/.mbo/logs/{ts}_{task}_{uuid8}.log` | worker stdio redirect + `mbo.worker` file handler | Process Console Log Output and Copy, `mbo processes` |
| `~/.mbo/logs/progress_{uuid}.json` | `TaskMonitor.update`, `_worker._update_status` | `ProcessInfo.update_from_sidecar`, watchdog |
| `~/.mbo/logs/args_{uuid}.json` | `ProcessManager.spawn` | worker, deleted after load |
| `~/.mbo/logs/mem_{uuid}.csv` | `_worker._start_mem_monitor` | user |
| `~/.mbo/logs/FAILURE_*.log` | `hpc.pipeline.write_failure_report` | user |
| `~/.mbo/cache/running_processes.json` | `ProcessManager._save` | GUI after restart, `mbo processes` |
| `~/.mbo/hpc/runs/*.json` | `hpc.history.record_run` | `mbo hpc watch` / `status` with no argument |
| `<output_dir>/logs/*.out`, `*.err`, `<output_dir>/timings.json` | SLURM / submitit, `write_timing_report` | `mbo hpc watch`, `mbo hpc status` |

Environment: `MBO_DEBUG`, `MBO_DIR` / `MBO_USER` (relocate `~/.mbo`),
`MBO_LOG_RETENTION_DAYS`, `MBO_LOG_KEEP_RECENT`, `MBO_MEM_LOG_INTERVAL`,
`MBO_MEM_LOG_EVERY`, `MBO_MEM_WARN_PCT`.

## 9. Adding a new array class

1. New module `mbo_utilities/arrays/<format>.py`. Subclass `LazyArray` (plus
   `ReductionMixin`; `RoiFeatureMixin` / `PhaseCorrectionMixin` if applicable).
2. Implement `can_open` (cheap, never raises), `__init__` (open lazily, keep
   `self.filenames`), `_shape5d`, `dtype`, `__getitem__` (5D keys), `close`.
3. Set `self._metadata` per §6.4. Put every source key you can into the registry's
   vocabulary; add missing aliases to `METADATA_PARAMS` (§10), never a private
   fallback.
4. Register a `PipelineInfo(category="reader", ...)` at import.
5. Add the class to `[project.entry-points."mbo_utilities.lazy_arrays"]` with a
   `PRIORITY` above 50 if its `can_open` is more specific than a suffix check.
6. Add it to `arrays/__init__._LAZY_IMPORTS`, the `imread` docstring, and
   `docs/file_formats.md` (Quick Reference row + detection tree).
7. Tests: shape/dims/indexing in the style of `tests/test_shape5d.py`, a round-trip
   in `tests/test_roundtrip.py`, synthetic data only.

## 10. Adding a metadata parameter, alias, or platform

1. Alias for an existing quantity: append to that entry's `aliases` (value verbatim)
   or `transforms` (converted value) in `metadata/base.py`. That is the only edit;
   `ALIAS_MAP`, `get_param`, `arr.<key>`, the viewer, and the writers pick it up.
2. New quantity: a new `MetadataParameter` keyed by its canonical name with `unit`,
   `dtype`, `default`, `label`, `description`. Add it to `IMAGING_METADATA_KEYS`
   only if every dataset should show it.
3. New platform: add its keys as aliases per §6.3, add a column to that table, and
   have the reader deposit the source keys (§6.4). Never add a per-platform branch to
   `get_param`.
4. Tests in `tests/test_metadata_module.py` (`get_param` from the alias,
   `get_canonical_name`) and `tests/test_effective_rate.py` if it is a rate.

## 11. Adding a new pipeline

1. Package per §7.1 with `params.py` first; make `Settings().to_dict()` round-trip
   and reproduce the reference run with defaults.
2. Runner per §7.3 (path or array in, suite2p-shaped directory out, §7.4).
3. `task_<name>(args, logger)` in `gui/tasks.py` that only unpacks args, re-opens the
   source, calls the runner, and reports through the task monitor.
4. Widget per §7.2, entry point in `pyproject.toml`.
5. `mbo <name>` in `cli.py` calling the runner.
6. Tests without the optional package installed: settings round-trip, stage gating,
   output conversion (`tests/test_masknmf_pipeline.py` is the model). Real runs are
   `@pytest.mark.slow` and skip without data (`tests/test_voltage_pipeline.py`).
7. Docs: a section in `docs/usage/gui_guide.md` and, when the pipeline has a CLI
   command, `docs/usage/cli.md`.

## 12. Tests

```bash
uv run pytest tests/ -v                  # everything that runs on synthetic data
KEEP_TEST_OUTPUT=1 uv run pytest tests/  # keep outputs
MBO_PIPELINE_TIFF=/path/to/raw uv run pytest tests/local/ -v   # needs real ScanImage data
```

- CI (`.github/workflows/test_python.yml`) runs only `tests/test_arrays.py
  tests/test_to_video.py -k synthetic`; everything else must still pass locally
  before a PR.
- Tests use synthetic fixtures from `tests/conftest.py`; real data lives under
  `~/.mbo/tests/lbm/mbo_utilities/` and tests skip when it is absent.
- `RENDERCANVAS_FORCE_OFFSCREEN=1` is set by `conftest.py`; no test opens a window.
- Contract tests to keep green when touching the three systems above:
  `test_lazyarray_contract.py`, `test_shape5d.py`, `test_natural_rank.py`,
  `test_numpy_dims.py`, `test_squeeze.py`, `test_imagej_stack.py`,
  `test_selection_indices.py`, `test_metadata_module.py`, `test_effective_rate.py`,
  `test_roundtrip.py`, `test_masknmf_pipeline.py`.
- No functions defined inside tests. Do not mock file formats; write a small real
  file to `tmp_path`.

## 13. Docs

```bash
cd docs && uv run make html
```

MyST + Sphinx book theme; published on merge to `master` by
`.github/workflows/deploy_docs.yml`. `docs/file_formats.md` (formats and shapes),
`docs/canonical_metadata.md` (metadata design), `docs/development.md` (internals),
`docs/usage/` (GUI, CLI, HPC). In-app help pages live in
`mbo_utilities/assets/docs/`. This file is the contract; the docs explain usage.
When they disagree, fix the docs.

## 14. Configuration

- User state under `~/.mbo/`: `settings/preferences.json`, `logs/` (§8.5),
  `cache/`, `imgui/`, `hpc/runs/`, `tests/` (test data), `templates/`. Resolve with
  `get_mbo_dirs()`, never hardcode.
- The imgui style is the user's, not the theme's. `gui/widgets/style_editor.py`
  holds the one `imgui_debugger.StyleEditor` for the process, opened from
  Widgets > Style Editor and backed by an `imgui_debugger.ConfigStore` at
  `get_mbo_dirs()["imgui"]`: `state.json` (the style as last left, plus the
  panel's own state), `styles/<name>.json` (named presets). It autosaves a
  second after the last slider moves; `apply_saved_style()` runs in
  `PreviewDataWidget.__init__` right after `style_imgui_opaque()`, so a saved
  style wins over the shipped theme. Window geometry stays imgui's, in
  `imgui/assets/app_settings/preview_settings.ini`. Nothing else writes the
  style; `imgui_debug.py` deliberately has no style entry. `imgui_debugger`
  is a base dependency, pinned to a commit of its GitHub repository in
  `pyproject.toml` (the PyPI release lags the API used here); bump the pin when
  it moves.
- The app host (`gui/app`) keeps its window geometry apart from the preview
  window's, in `get_mbo_dirs()["imgui"]/app.ini`.
- Environment: `MBO_GPU` (GPU toggle; also `mbo gpu`), `RENDERCANVAS_FORCE_OFFSCREEN`,
  `KEEP_TEST_OUTPUT`, `MBO_PIPELINE_TIFF`; logging and retention variables are
  listed in §8.5.
- HPC: `hpc.toml` written by `mbo hpc init`; shared cluster environment is described
  in `docs/usage/hpc.md`. Nothing in this repo contains cluster credentials.
- Nothing here is a secret; do not add any.

## 15. Conformance backlog

Known places where the code does not follow this file. Fix on touch; do not add new
ones. Remove an entry when its fix lands.

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
  built-in tasks in `gui/tasks.py:1608-1619`; of the five widgets only `ROIs`
  declares `info` and has a `pyproject.toml` `mbo_utilities.pipelines` entry (it is
  also in the hardcoded list so a checkout finds it); none declares `task_type` or
  `task_func`. Target: §7.2 for every built-in, entry points as the only
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

## 16. Imgui spacing

Names: Python (`imgui.*` unless prefixed). Defaults from `ImGuiStyle::ImGuiStyle()`. `avail` = `get_content_region_avail()`.

### A. Read position and sizes

| call | returns | coords | notes |
|---|---|---|---|
| `get_cursor_screen_pos()` | next emit position | absolute | preferred; matches draw-list coords |
| `get_cursor_pos()` / `_x()` / `_y()` | next emit position | window-local, scroll-affected | |
| `get_cursor_start_pos()` | position right after `begin()` | window-local | ~ `window_padding` |
| `get_content_region_avail()` | room from cursor to right/bottom edge | size | minus scrollbar; inside table/columns = cell |
| `get_window_pos()` / `get_window_size()` / `get_window_width()` / `_height()` | window rect | absolute | local -> absolute conversion only |
| `get_item_rect_min()` / `_max()` / `_size()` | last item bbox | absolute | after `end_group()` = whole group |
| `get_text_line_height()` | `font_size` | | |
| `get_text_line_height_with_spacing()` | `font_size + item_spacing.y` | | pitch of text rows |
| `get_frame_height()` | `font_size + 2*frame_padding.y` | | button height, checkbox square, `color_button` default |
| `get_frame_height_with_spacing()` | `frame_height + item_spacing.y` | | pitch of widget rows |
| `get_font_size()` | scaled font px | | the em unit |
| `calc_text_size(text, text_end=None, hide_text_after_double_hash=False, wrap_width=-1)` | text bbox | size | `True` strips `##id`; `wrap_width` for wrapped height |
| `calc_item_width()` | width next value widget gets | | negatives resolved |
| `get_tree_node_to_label_spacing()` | `font_size + 2*frame_padding.x` | | `bullet()` / `tree_node` label offset |
| `is_rect_visible(size)` / `(min, max)` | clip test | bool | skip offscreen work |
| `get_scroll_x()` / `_y()` / `get_scroll_max_x()` / `_y()` | scroll state | | `max = content - window - decorations` |

### B. Move the cursor

| call | effect | notes |
|---|---|---|
| `set_cursor_screen_pos(pos)` | jump, absolute | must be followed by an item or `dummy`, else content size not extended |
| `set_cursor_pos(pos)` / `_x(x)` / `_y(y)` | jump, window-local | same rule |
| `same_line(offset_from_start_x=0, spacing=-1)` | undo last carriage return | table C |
| `new_line()` | force carriage return | height `font_size`, or current line height if any |
| `spacing()` | blank of `item_spacing.y` | `item_size((0,0))` |
| `dummy(size)` | reserve `size`, no interaction, no nav | `invisible_button` = interactive twin |
| `indent(w=0)` / `unindent(w=0)` | `dc.indent += w` or `indent_spacing` | moves cursor x now |
| `tree_push(id)` / `tree_pop()` | `indent()` + `push_id()` | |
| `bullet()` | circle, cursor x += tree_node_to_label_spacing, stays on line | |
| `separator()` | 1px line to `work_rect.max.x`, no layout height | vertical inside horizontal layout / menu bar |
| `separator_text(label)` | line + label | `separator_text_padding` / `_align` / `_border_size` |
| `align_text_to_frame_padding()` | line height >= `frame_height`; baseline = `frame_padding.y` | before `text()` that precedes a framed widget |
| `begin_group()` / `end_group()` | lock line start x; group becomes one item | `same_line` / `is_item_hovered` / `get_item_rect_size` on it |
| `set_scroll_here_x(r=0.5)` / `_y(r=0.5)` | scroll so cursor visible | `0` = left/top, `0.5` = center, `1` = right/bottom |
| `set_scroll_from_pos_x(local, r)` / `_y` | same, explicit position | `get_cursor_start_pos() + offset` |
| `set_scroll_x(v)` / `_y(v)` | absolute scroll | |

### C. `same_line` cases

| call | `cursor.x` | gap |
|---|---|---|
| `same_line()` | `prev_line.x + item_spacing.x` | default |
| `same_line(0, 0)` | `prev_line.x` | none, touching |
| `same_line(0, w)` | `prev_line.x + w` | `w` |
| `same_line(x)` | `window.pos.x - scroll.x + x + group_offset + columns_offset` | none |
| `same_line(x, w)` | above `+ w` | `w` |

`cursor.y = prev_line.y`; line height and baseline restored from previous line.

### D. Text baseline

| widget | `item_size(size, baseline)` | line height |
|---|---|---|
| `text()` | baseline `0` | `font_size` |
| `button()`, `checkbox()`, `slider*`, `input*`, `combo` | baseline `frame_padding.y` | `frame_height` |
| `small_button()` | `frame_padding.y = 0` | `font_size` |
| `color_button(size < frame_height)` | baseline `0` | `size.y` |
| `align_text_to_frame_padding()` then `text()` | baseline `frame_padding.y` | `frame_height` |

Rule: shorter item shifted down by `curr_line_text_base_offset - baseline`; text after a framed item aligns for free, text before needs `align_text_to_frame_padding()`.

### E. Widths and size arguments

| call | scope |
|---|---|
| `push_item_width(w)` / `pop_item_width()` | value widgets: slider, drag, input, combo, color_edit, list_box |
| `set_next_item_width(w)` | next widget only |
| `imgui.internal.push_multi_items_widths(n, w)` | `drag_float3` style split: `(w - (n-1)*item_inner_spacing.x) / n` |

| value | meaning |
|---|---|
| `w > 0` | pixels |
| `w == 0` (push) | default: `0.65 * window width`, or `16 * font_size` in child/popup |
| `w < 0` | `avail.x + w` |
| `-imgui.FLT_MIN` | fill to right edge |

| size arg | `0` | `< 0` |
|---|---|---|
| `button(label, size)` | text + `2*frame_padding` | `avail + size` |
| `selectable(label, sel, flags, size)` | full width x text height | `avail + size` |
| `begin_child(id, size)` | fill remaining | `avail + size` |
| `input_text_multiline(label, s, size)` | `item_width` x 8 lines | `avail + size` |
| `progress_bar(frac, size=(-FLT_MIN, 0))` | full width | |
| `list_box(label, i, items, height_in_items=-1)` | ~7 items | |
| `plot_lines(..., graph_size)` | `item_width` x `frame_height` | |
| `implot.begin_plot(title, size)` | `plot_default_size` | `avail + size` |
| `implot.colormap_scale(label, lo, hi, size)` | 20px bar + labels, `plot_default_size.y` | `avail + size` |
| `image_button(id, tex, image_size)` | `image_size + 2*frame_padding` | |

### F. Text wrapping and in-rect alignment

| call | effect |
|---|---|
| `push_text_wrap_pos(x=0)` / `pop_text_wrap_pos()` | `< 0` none; `0` wrap at right edge; `> 0` wrap at window-local x |
| `text_wrapped(s)` | push(0) + text + pop; needs a sized window |
| `label_text(label, value)` | value in `item_width`, label after `item_inner_spacing.x` (same as sliders) |
| `bullet_text(s)` | `bullet()` + `text()` |
| `imgui.internal.render_text_clipped(pos_min, pos_max, text, text_end, size_known, align=(0,0), clip_rect=None)` | draw text aligned inside a rect; `(0.5,0.5)` = centered |
| `imgui.internal.render_text_clipped_ex(draw_list, ...)` | same, explicit draw list |
| `imgui.internal.text_aligned(align_x, size_x, fmt)` | text aligned within `size_x` |
| `imgui.internal.render_text_ellipsis(draw_list, pos_min, pos_max, ellipsis_max_x, text, ...)` | truncate with `...` |
| `imgui.internal.render_text_wrapped(pos, text, text_end, wrap_width)` | draw wrapped |
| `imgui.internal.calc_wrap_width_for_pos(pos, wrap_pos_x)` | wrap width at a position |

### G. Style: spacing and padding

| field | default | `StyleVar_` | consumed by |
|---|---|---|---|
| `window_padding` | (8,8) | yes | cursor start, content region |
| `frame_padding` | (4,3) | yes | framed widgets, `frame_height`, baseline |
| `item_spacing` | (8,4) | yes | `.x` after `same_line`, `.y` after each line |
| `item_inner_spacing` | (4,4) | yes | box->label, slider->label, multi-component gap |
| `cell_padding` | (4,2) | yes | tables; `.x` per table, `.y` per row |
| `indent_spacing` | 21 | yes | `indent`, tree nodes |
| `columns_min_spacing` | 6 | no | legacy columns |
| `separator_text_padding` | (20,3) | yes | `separator_text` |
| `separator_text_border_size` | 3 | yes | `separator_text` |
| `scrollbar_size` | 14 | yes | subtracts from `avail` |
| `scrollbar_padding` | 2 | yes | grab inside scrollbar |
| `grab_min_size` | 12 | yes | slider/scrollbar grab |
| `image_border_size` | 0 | yes | `image()` |
| `window_min_size` | (32,32) | yes | |
| `window_border_size` / `child_border_size` / `popup_border_size` / `frame_border_size` | 1 / 1 / 1 / 0 | yes | |
| `window_border_hover_padding` | 4 | no | resize hit zone |
| `touch_extra_padding` | (0,0) | no | hit boxes only |
| `display_window_padding` | (19,19) | no | keep windows on screen |
| `display_safe_area_padding` | (3,3) | no | popups/tooltips edge margin |
| `docking_separator_size` | 2 | yes | |
| `tab_bar_border_size` / `tab_bar_overline_size` | 1 / 1 | yes | |
| `tab_min_width_base` / `tab_min_width_shrink` | 1 / 80 | yes | |
| `tree_lines_size` / `tree_lines_rounding` | 1 / 0 | yes | |
| `drag_drop_target_padding` / `_border_size` | 3 / 2 | no | |
| `log_slider_deadzone` | 4 | no | |
| `layout_align` | 0.5 | yes | stack layout minor axis |

### H. Style: alignment fields (0 = left/top, 0.5 = center, 1 = right/bottom)

| field | default | `StyleVar_` | applies to |
|---|---|---|---|
| `button_text_align` | (0.5,0.5) | yes | `button` label when button larger than text |
| `selectable_text_align` | (0,0) | yes | `selectable` label |
| `separator_text_align` | (0,0.5) | yes | `separator_text` label |
| `window_title_align` | (0,0.5) | yes | title bar text |
| `table_angled_headers_text_align` | (0.5,0) | yes | angled headers |
| `table_angled_headers_angle` | 35 deg | yes | angled headers |
| `layout_align` | 0.5 | yes | `begin_horizontal` / `begin_vertical` cross-axis |
| `window_menu_button_position` | `Dir_.left` | no | collapse button side |
| `color_button_position` | `Dir_.right` | no | swatch side in `color_edit` |

### I. Style API

| call | notes |
|---|---|
| `get_style()` | poke fields between frames |
| `push_style_var(idx, float or ImVec2)` / `pop_style_var(n=1)` | inside a frame |
| `push_style_var_x(idx, x)` / `push_style_var_y(idx, y)` | one component of an ImVec2 var |
| `imgui_ctx.push_style_var(idx, val)` | context manager |
| `get_style().scale_all_sizes(f)` | truncs every size to int; once, on a fresh style |
| `font_scale_main` / `font_scale_dpi` / `font_size_base` | font scale only, sizes untouched |
| `imgui.internal.get_style_var_info(idx)` | offset / type of a `StyleVar_` |

### J. Stack layout (pthom fork, `IMGUI_HAS_STACK_LAYOUT`)

| call | notes |
|---|---|
| `begin_horizontal(id, size=(0,0), align=-1)` / `end_horizontal()` | `size` 0 on an axis = shrink to content |
| `begin_vertical(id, size=(0,0), align=-1)` / `end_vertical()` | |
| `spring(weight=1, spacing=-1)` | share of free space; `weight 0` = fixed `spacing` only; `spacing -1` = `item_spacing` |
| `suspend_layout()` / `resume_layout()` | opt out for self-laid-out widgets |
| `align -1` | use `style.layout_align` |
| nested layout with size 0 | inherits parent size on that axis |
| `imgui_ctx.begin_horizontal(...)` / `begin_vertical(...)` | context managers |
| `separator()` inside | auto vertical/horizontal |

### K. Tables

| item | notes |
|---|---|
| `begin_table(id, columns, flags=0, outer_size=(0,0), inner_width=0)` | `outer_size < 0` = `avail + size`; `inner_width` with `scroll_x` |
| `table_setup_column(label, flags=0, init_width_or_weight=0, user_id=0)` | `width_fixed` = px, `width_stretch` = weight |
| `TableFlags_.sizing_fixed_fit` | columns = content width (default with `scroll_x`) |
| `TableFlags_.sizing_fixed_same` | all columns = widest |
| `TableFlags_.sizing_stretch_prop` | stretch by content ratio |
| `TableFlags_.sizing_stretch_same` | equal stretch (default without `scroll_x`) |
| `TableFlags_.no_pad_outer_x` / `pad_outer_x` / `no_pad_inner_x` | outer/inner `cell_padding.x` |
| `TableFlags_.no_host_extend_x` / `_y` | do not grow to parent when `outer_size` is 0 |
| `TableFlags_.scroll_x` / `scroll_y` | wraps in a child |
| `TableColumnFlags_.indent_enable` / `indent_disable` | apply `dc.indent` in cell (default col 0 only) |
| `TableColumnFlags_.no_resize`, `angled_header` | |
| `table_next_row(flags=0, min_row_height=0)` | row height floor |
| `table_next_column()` / `table_set_column_index(i)` | |
| `cell_padding` | `.x` locked per table, `.y` may vary per row |
| alignment inside a cell | none built-in: `set_cursor_pos_x` + `calc_text_size`, or `internal.text_aligned` |
| `imgui.internal.table_get_column_width_auto(table, column)` / `table_get_header_row_height()` | |

### L. Legacy columns (prefer tables)

| call | notes |
|---|---|
| `columns(count=1, id=None, borders=True)` / `next_column()` | |
| `set_column_width(i, w)` / `get_column_width(i)` | `i = -1` current |
| `set_column_offset(i, x)` / `get_column_offset(i)` | from content region left |
| `columns_min_spacing` | style |
| `separator()` | spans all columns |

### M. Windows and children

| call | notes |
|---|---|
| `set_next_window_pos(pos, cond=0, pivot=(0,0))` | `pivot` = anchor: `(1,0)` top-right, `(0.5,0.5)` centered |
| `set_next_window_size(size, cond=0)` | axis `0` = auto-fit |
| `set_next_window_size_constraints(min, max, cb=None)` | `FLT_MAX` = unbounded |
| `set_next_window_content_size(size)` | scroll range without widgets |
| `set_next_window_scroll(pos)` | |
| `WindowFlags_.always_auto_resize` | window = content each frame |
| `begin_child(id, size=(0,0), child_flags=0, window_flags=0)` | `0` = fill, `< 0` = `avail + size` |
| `ChildFlags_.auto_resize_x` / `_y` | child = content on that axis |
| `ChildFlags_.always_auto_resize` | re-measure every frame |
| `ChildFlags_.borders`, `frame_style`, `always_use_window_padding` | border / widget-like padding / padding without border |
| `ChildFlags_.resize_x` / `_y` | user-resizable from border |
| `ChildFlags_.nav_flattened` | nav crosses child boundary |
| `hello_imgui.widget_with_resize_handle(id, fn, handle_size_em=1, on_item_resized=None, on_item_hovered=None)` | corner handle on any widget |
| `imgui.internal.calc_window_next_auto_fit_size(window)` | |

### N. Overlap and hit-testing

| call | notes |
|---|---|
| `set_next_item_allow_overlap()` | later items may sit on top of the next one |
| `invisible_button(id, size, flags=0)` | interactive `dummy` |
| `touch_extra_padding` | hit box only, not layout |
| `imgui.internal.item_hoverable(bb, id, flags)` | |

### O. Internal layout primitives (`imgui.internal`)

| call | notes |
|---|---|
| `item_size(size or bb, text_baseline_y=-1)` | advance cursor |
| `item_add(bb, id, nav_bb=None, extra_flags=0)` | register; returns visible |
| `calc_item_size(size, default_w, default_h)` | `0` -> default, `< 0` -> `avail + size` |
| `get_current_window().dc` | `cursor_pos`, `cursor_pos_prev_line`, `cursor_max_pos`, `curr_line_size`, `prev_line_size`, `curr_line_text_base_offset`, `indent`, `group_offset`, `columns_offset`, `is_same_line`, `item_width`, `text_wrap_pos`, `layout_type` |
| `render_frame(p_min, p_max, col, borders=True, rounding=0)` | framed background |
| `shrink_widths(items, count, width_excess, width_min)` | tab-bar style shrink |
| `set_window_pos(window, pos, cond=0)` | |
| `get_window_scrollbar_rect(window, axis)` | |

### P. imgui_bundle helpers

| call | notes |
|---|---|
| `hello_imgui.em_size()` | `get_font_size()` |
| `hello_imgui.em_size(n)` | `n` lines |
| `hello_imgui.em_to_vec2(x, y)` / `em_to_vec2(v)` | ImVec2 in em |
| `hello_imgui.pixels_to_em(v)` / `pixel_size_to_em(f)` | inverse |
| `immapp.em_size` / `immapp.em_to_vec2` | re-exports |
| `hello_imgui.begin_group_column()` / `end_group_column()` | `begin_group` / `end_group + same_line` |
| `hello_imgui.image_from_asset(path, size)` / `image_size_from_asset(path)` | |
| `hello_imgui.DpiAwareParams.dpi_window_size_factor` / `dpi_font_loading_factor` | |
| `imgui_ctx.begin_group()`, `begin_horizontal`, `begin_vertical`, `push_style_var`, `push_item_width`, `push_text_wrap_pos`, `push_id`, `begin_child`, `begin_table` | context managers |

### Q. ImPlot / ImPlot3D

| item | notes |
|---|---|
| `implot.StyleVar_.plot_padding` | plot edge to axes |
| `implot.StyleVar_.label_padding` | axis labels, tick labels, edge |
| `implot.StyleVar_.legend_padding` / `legend_inner_padding` / `legend_spacing` | (10,10) / (5,5) / (5,0) |
| `implot.StyleVar_.mouse_pos_padding` / `annotation_padding` / `fit_padding` | |
| `implot.StyleVar_.plot_default_size` / `plot_min_size` | `begin_plot` size 0 / shrink floor |
| `implot.StyleVar_.major_tick_len` / `minor_tick_len` / `plot_border_size` | |
| `implot.setup_legend(location, flags=0)` | `Location_` bitmask `north | south | west | east | center` |
| `implot.plot_text(text, x, y, pix_offset=(0,0), flags=0)` | centered at point; `TextFlags_.vertical` |
| `implot.annotation(x, y, col, pix_offset, clamp, round=False)` | offset label, `clamp` keeps inside plot |
| `implot.tag_x(x, col, round=False)` / `tag_y` | axis tags |
| `implot3d.StyleVar_.legend_padding` / `legend_inner_padding` / `legend_spacing`, `implot3d.setup_legend(location, flags)` | same model |

### R. Formulas

| need | expression |
|---|---|
| right-align item of width `w` | `set_cursor_pos_x(get_cursor_pos_x() + avail.x - w)` |
| right-align via `same_line` | `same_line(get_window_width() - w - style.window_padding.x)` |
| center item of width `w` | `set_cursor_pos_x(get_cursor_pos_x() + (avail.x - w) * 0.5)` |
| center text in rect (draw list) | `pos = rect.min + (rect.size - calc_text_size(s)) * 0.5` |
| centered label in oversized button | `style.button_text_align = (0.5, 0.5)` (default) |
| fill width | `-imgui.FLT_MIN` |
| `n` equal columns | `(avail.x - (n-1) * item_spacing.x) / n` |
| button width for label | `calc_text_size(label).x + 2 * frame_padding.x` |
| widget row pitch | `get_frame_height_with_spacing()` |
| text row pitch | `get_text_line_height_with_spacing()` |
| leave one bottom row in a child | `begin_child(id, (0, -get_frame_height_with_spacing()))` |
| checkbox total width | `frame_height + item_inner_spacing.x + label.x` |
| tree indent | `indent_spacing` (21 ~ `font_size + 2*frame_padding.x`) |
| em | `hello_imgui.em_size(n)` = `n * get_font_size()` |
| pixel snap | cursor truncated to int each `item_size`; pass integer sizes |

## 17. Proposal: arrays and pipelines as GUI contributors

**Status: a proposal, not a contract.** Nothing in this section is implemented, and
the stages are sketches, not complete designs. Until a stage lands and moves its
rules into §2 and §7, the code follows the sections above. Fill in the gaps when a
stage is taken up; do not build against this section as if it were settled.

The frame is model-view-controller with the immediate-mode twist: imgui keeps no
view objects, so the split that matters is a GUI-free observable model, draw
functions that only read it, and a command layer that mutates it or starts work.

### 17.1 Where it stands

Four places already have the shape and are the template:

- `motion_correction` (§5.8): a reader answers a typed, GUI-free dataclass or None,
  `MotionPlot` draws it, nothing in `gui` names a format.
- `annotation`: `RoiModel` is `Observable`; `manual_roi.py` and the ROIs pipeline
  are two views of it and neither polls the other.
- `Playhead` (§7.6): one piece of shared state every view subscribes to.
- `TraceProfile` (§7.6) and the results zarr (§7.5): the pipeline declares what its
  data means, generic views render any pipeline.
- `gui/app` (`mbo app`), the preview window's replacement in progress. `AppHost`
  is built on the viewer's figure (an `NDWidget` makes its own), holds the open
  `LazyArray`, the `Playhead` and the channel and z-plane on screen, and says
  `data_changed` to every app when other data opens. An `App` draws through
  `draw_options` / `draw_canvas` into a dock tab or a floating window the host
  picks. Ported: the viewer (`ViewerApp`), Open, Summary Images, Projections,
  Tile Grid, Metadata, Diagnostics, Log and the imgui tools. Not yet: window and
  spatial functions, frame averaging, scan phase, Signal Quality, keyboard
  shortcuts, Save As, Set Metadata, the Process tab, Manual ROI, MESc, IsoView
  tools, the process console, Help / Keybinds / Options, BioHPC / Cloud.
  `HostAsParent` is the shim a ported `Widget` reads; it only shrinks.

Everything else is the opposite shape (counts from 2026-09-19):

- The host widget is a grab bag: 215 distinct `parent._x` attributes across `gui`,
  and `_dialogs._reset_per_data_state` is a hand-kept list of which to clear when
  the array changes. Every unit swap risks a leak.
- Widgets sniff formats: the `mesc_units` tab, `tile_grid` and
  `isoview_align_views` decide `is_supported(parent)` by unwrapping the array and
  checking its class. A new format has to write imgui code in `gui/widgets/` to
  appear anywhere.
- Discovery is a package scan (`gui/widgets/__init__._discover_widgets`), so no
  reader and no plugin can contribute a panel, tab or table.
- Pipelines are hardcoded widget classes (§15) drawing their settings by hand
  (`pipelines/voltage.py` 875 lines, `pipelines/isoview.py` 3303).
- Tables are written four times (MESc units, ROIs, Traces, runs), each with its own
  sort, hide and action code.
- Two side apps rebuild the viewer: `linescan_viewer.py` (1729 lines) and the
  curation dashboard.

### 17.2 Target shape

Three GUI-free things an array or pipeline provides, and generic views that draw them.

- **Facets.** Typed answers an array gives about itself beyond pixels, as properties
  returning a dataclass or None: `units` (sibling units and their links),
  `overlays(unit)` (ROI outlines in an image's pixels), `line_positions`, `views`
  and `tiles` for IsoView, `scan_phase`, `motion_correction` (exists). `LazyArray`
  answers None; a reader overrides. §5.8 generalized; `arrays` still never imports
  `gui` (§2).
- **Session.** One observable object per viewer (`mbo_utilities/session.py`,
  GUI-free): the array, the indices, the playhead, the selection, the open-unit
  cache, loaded runs and results, display options. It replaces the host widget's
  attributes; `swap_viewer_array` becomes `session.open(arr)`, one event, and views
  rebuild from it instead of a reset list.
- **Contributions.** What a facet or a pipeline wants on screen, as data:
  `Panel(section, controls)`, `Tab(table)`, `Table(columns, rows, actions, links)`,
  `Overlay(records)`, `Form(dataclass)`. Controls are a small vocabulary: choice,
  toggle, number, button, info, table, overlay, plot. A reader ships a
  `contributions(session)` function beside its class, found through the entry point
  that registers the reader. A pipeline's `PipelineInfo` gains `settings`,
  `trace_profile`, `axes_consumed` and `applies_to`, so the Run tab draws it as a form.

| Role | What | Rule |
|------|------|------|
| Model | arrays + facets, session, annotation, results, registries | GUI-free, observable, tested on synthetic files |
| View | generic draw functions: `TableView`, `FormView`, `OverlayView`, the trace and motion plots | reads the session, emits intents, holds no state |
| Controller | intent handlers and commands: `session.open_unit`, `ProcessManager.spawn`, worker tasks | the only code that mutates the model or starts work |

For MESc this would read (illustrative; names not settled):

```python
class MescArray(LazyArray):
    units: UnitSet | None                 # list_mesc_units + links as a dataclass
    def overlays(self, unit) -> list[OverlayRecord]   # image_overlays

def contributions(session):
    a = session.array
    return [
        Tab("MESc", Table(UNIT_COLUMNS, unit_rows(a.units), links=unit_links, on_row=session.open_unit)),
        Overlay("ROI Overlay", a.overlays(session.unit), plane=session.indices),
    ]
```

The voltage pipeline contributes the same way: a form from `VoltageSettings`, its
domain table, its trace profile; its results tab already comes from the results zarr.

### 17.3 Stages

Ordered by what unblocks what. Each stage is shippable alone and lands with a pinned
test; each one's rules move into §2 and §7 when it lands.

1. **Session.** Move per-dataset state off the host widget; widgets take `session`,
   not `parent`; `swap_viewer_array` and `_reset_per_data_state` collapse into
   `session.open`. Pin: open two arrays in sequence, assert nothing leaks.
2. **Facets.** `units`, `overlays`, `line_positions` on `MescArray`; `views`, `tiles`
   on `IsoviewArray`. The four format widgets become views gated on
   `session.array.units is not None`; `mesc_array_of` goes away. Pin: one test per
   facet on the synthetic files `tests/test_mesc_geometry.py` builds.
3. **Table and Panel vocabulary.** `Table` model and one `TableView`; port the MESc
   table first, then runs, ROIs, Traces. Discovery moves from the package scan to
   registries: built-ins, readers' `contributions`, pipelines' infos. Pin: a Table
   model test plus one bare-context draw test (the `tests/test_mesc_tab.py` pattern).
4. **Declarative pipelines.** Entry points as the only registration (§15).
   `PipelineInfo` carries `settings`, `trace_profile`, `axes_consumed`, `applies_to`;
   a `FormView` draws any settings dataclass; a widget class keeps only sections a
   form cannot express (voltage's domains and curation). Pin: a pipeline registered
   only through an entry point shows up with a runnable form.
5. **Fold the side apps.** The line-scan viewer's snapshot, stack and reference
   panels and the curation dashboard become contributions on the standard viewer,
   bound to the session's playhead; `viewers.get_viewer_class` (pollen) becomes one too.

Rules to add to §2 when stage 1 lands: a `gui` module never names a format, it asks
the session for a facet; state lives on the session, never on the host widget; a
table is a `Table` model drawn by `TableView`; a pipeline appears through its
`PipelineInfo`, never by editing the Run tab.

Open questions, deliberately unanswered here: where a format's `contributions`
module lives and how the reader entry point points at it; whether `Session` is one
class or a small tree (viewer, annotation, runs); how a contribution declares its
imgui-only escape hatch; what a `Form` does with nested dataclasses and tri-state
stage toggles; how the side apps' own sliders map onto the session's indices.

Two cautions. Do not start in `manual_roi.py` (4390 lines) or `pipelines/isoview.py`;
stages 1 to 3 shrink them by subtraction. Keep the control vocabulary small: choice
and table cover every format need seen so far, forms cover most pipeline settings,
and anything else stays hand-written imgui behind a contribution rather than a new
abstraction.
