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
│   │   ├── tiff.py zarr.py h5.py numpy.py bin.py mesc.py pf.py suite2p.py mp4.py
│   │   └── isoview/          # IsoView light-sheet trees (four layouts, one class)
│   ├── metadata/             # canonical vocabulary, alias resolution, OutputMetadata
│   ├── pipeline_registry.py  # PipelineInfo + entry-point loading
│   ├── results.py            # the results zarr every pipeline molds into (§7.5)
│   ├── masknmf/  vnoiser/    # pipeline packages: params / runner / outputs / qc
│   ├── roi_workflow.py       # register -> ROI subset -> extract | demix | discover
│   ├── hpc/                  # submitit/SLURM runner for the suite2p pipeline (`mbo hpc`)
│   ├── gui/                  # Miller Brain Studio (imgui + fastplotlib)
│   │   ├── widgets/pipelines # Run tab: one PipelineWidget per pipeline
│   │   ├── tasks.py          # worker task table: task_<name>(args, logger)
│   │   └── _worker.py        # python -m mbo_utilities.gui._worker <task_type> <args_json>
│   ├── analysis/             # scan-phase, linescan, phasecorr math
│   ├── annotation/           # GUI-free manual-ROI label store + NGFF labels zarr
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
   `IsoviewArray` 90, `PfArray` 70, `MescArray` 60, everything else 50.
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
- Internal: `selection_to_canonical(arr, {...})` → `{"T": [...], "C": [...], "Z": [...]}`
  0-based; `to_lsp_kwargs` re-emits 1-based `timepoints`/`planes`/`channels` for
  `imwrite` and `lbm_suite2p_python`; `to_isoview_kwargs` emits 0-based
  `timepoints`/`cameras`. Axis aliases (`view`, `cam`, `plane`, `tile`, ...) resolve
  through `_dim_labels._SLIDER_NAME_ALIASES`.
- `arr.roi`: `None` = stitched full FOV, `0` = split every ROI, `n` = ROI n, list =
  those ROIs. `imwrite(roi_mode=RoiMode.concat_y | separate)`; `separate` fans out
  one `roiNN/` directory per ROI.

Pinned by `tests/test_selection_canonical.py`.

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
v3 group, `<yyyy-mm-dd>_<tags>.zarr`, that a reader, a viewer or a notebook opens
the same way whichever pipeline wrote it. It is the standard output format; a
pipeline's native files stay its cache and its compatibility layer.

```
<yyyy-mm-dd>_<tags>.zarr/            zarr v3 group; attrs: mbo_results (schema version),
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
```

- **Naming.** `results_name(source)` is today's date, then the tags the source
  filename carries in the §5.6 vocabulary (`filename_tags`: `session01`, `zplane03`,
  `tp00001-01574`; `session` is the `S` tag), then any `extra_tags`. A source with no
  tags contributes its stem (`stan112_expt12.mesc` → `2026-09-16_stan112_expt12.zarr`)
  so the file still says what it is. Unit groups are named by the same vocabulary
  (`unit_name("plane", 1)` is `zplane01`; scans are `scan<id>`).
- **Molding.** A pipeline builds one `ResultUnit` per plane or scan and calls
  `write_results(path, units, pipeline=..., source=..., settings=..., metadata=...)`.
  `results_from_suite2p(dir)` molds suite2p and MaskNMF folders (`F` → `raw`, `Fneu` →
  `neuropil`, `spks` → `spikes`, `norm_traces` → `dff`, `stat` → pixel members,
  `meanImg`/`max_proj`/`Vcorr`/`refImg` → images); `results_from_pf(dir)` molds the
  voltage pipeline's `PF` folder (domains are the ROIs, their lines the members with
  `members/raw`, `test.h5` gives `dff` and `zscore`, peaks are the events). Copy one of
  them for a new pipeline; never invent a trace or image kind, add it to the registry.
- **Writing.** The voltage pipeline writes it when `VoltageSettings.runtime.output_format`
  is `"zarr"` (the Run tab's Output format, `mbo voltage --zarr`): the pickles are
  deleted, `test.h5`, `traces/` and `pipeline.json` stay, and `PfArray` opens the folder
  from the zarr (`pf_results_in`). `mbo results <dir>` converts an existing suite2p,
  MaskNMF or PF folder. The curation window opens the folder through `PfArray`, so a
  folder written as zarr curates like one written as pickles.
- **Reading.** `read_results(path)` returns `Results` (`.units[name]` → `ResultUnit`,
  every array in memory). `results_pipeline(path)` and `results_summary(path)` read
  only `zarr.json` files and are what `can_open` and the run scanners use: `ZarrArray`
  declines a results file, `PfArray` claims a voltage one. `imread` never returns a
  results file as an image.
- **Viewing.** The ROI widget's Traces tab takes a results file through the same door
  as a run dir: `ManualRoiWidget.load_run(path)` (a file, or one unit as
  `<file>.zarr/zplane01`) calls `load_results`. A pixel unit becomes a `RunResult`
  (`roi_workflow.run_result_from_unit`) and loads as a derived set with its overlay,
  exactly like `stat.npy` + `F.npy`; a line unit becomes an `external` `TraceSet` with
  one row per ROI (its `denoised` trace, else `dff`, else `raw`) and one per member
  line, each entry carrying its `label` and `fs`. `roi_runs.run_dir_complete` and
  `scan_run_dirs` treat results files as run dirs, `roi_runs.json` restores them, a
  finished `voltage` worker is adopted like a suite2p one, and the Voltage tab's
  "Load into Traces" button does it on demand. A new pipeline that writes the results
  zarr therefore reaches the Traces tab with no GUI code.

Pinned by `tests/test_results.py`, `tests/test_voltage_pipeline.py`.

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
- Level: `MBO_DEBUG=1` selects DEBUG, else INFO, decided once at import. The GUI
  "Debug logging" toggle (`_options_popup`, `file_dialog`) persists the preference,
  sets `os.environ["MBO_DEBUG"]` so spawned workers inherit it, and calls
  `set_global_level`; `run_gui` applies the persisted value at launch.
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
  `test_selection_canonical.py`, `test_metadata_module.py`, `test_effective_rate.py`,
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
- No `--log-level` or `-v` on `mbo` or `mbo hpc`; `MBO_DEBUG` is read once at import
  (`log.py:29-30`).
- `docs/development.md:98-99,117-118` document `log.enable` and `log.disable`, which
  do not exist.

**Pipelines**

- Built-in widgets are hardcoded in `gui/widgets/pipelines/__init__.py:51-73` and
  built-in tasks in `gui/tasks.py:1608-1619`; none of the four widgets declares
  `info`, `task_type`, or `task_func`; `pyproject.toml` has no
  `mbo_utilities.pipelines` table. Target: §7.2 for every built-in, entry points as
  the only registration path.
- Suite2p's `PipelineInfo` is registered from the reader module with category
  `segmentation` (`arrays/suite2p.py:31-54`); MaskNMF, ROI workflow, and the IsoView
  processing modes have no `PipelineInfo` at all (the four `isoview-*` infos are
  readers).
- `hpc/` is suite2p-only and routes `[parameters]` keys by a hardcoded suite2p list
  (`hpc/config.py:66-73`, `hpc/pipeline.py:65-100`).
- Legacy `hpc/*.sh` and `hpc/run_pipeline.py` duplicate `mbo hpc`; remove once the
  submitit path is validated on the cluster.

**Style**

Counts from `ruff check` on the `voltage-pipeline` branch (2026-09-15), before the
first `format.yml` run on `main`. The workflow autofixes what it can; the rest is
fix-on-touch. When a family reaches zero, move its rule into `select`.

- Selected, not autofixable: `PTH` 39, `ERA001` 44, `F841` 11, `F403`/`F405` star
  imports 4, `D301` 14, `D200` 5, `D404` 2, `UP` 13, `E402`/`E702`/`E721`/`E741` 17.
- Ignored until swept: `E501` 1827 (recount after the first format run; the rest are
  long strings and comments), `D205` 505, `D400` 37.
- Not yet selected: `T20` 111 `print` calls in library code, `BLE001` 414 blind
  excepts, `S110` 117 `try`/`except`/`pass`, `B` 76, `SIM` 125, `N` 281, `G004` 424
  f-strings in log calls, `PLC0415` 1475 function-local imports (most are the
  sanctioned heavy packages; needs per-import `noqa` before enabling).
- Not ruff-checkable: 195 banner comments and 251 section-header comments in 22
  files, 19 `logging.getLogger` calls, 73 nested `def`s in the library and 60 in
  tests.
