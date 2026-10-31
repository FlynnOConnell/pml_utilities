# AGENTS.md

Single source of truth for any agent (human or AI) contributing to pml_utilities.
The importable package is `mbo_utilities`; `pml_utilities` is only the distribution
name. `CLAUDE.md` imports this file and `STYLE.md`. Where code and this file
disagree, this file wins: fix the code when you touch it, and record anything you
cannot fix in `CONFORMANCE.md`.

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
│   ├── roi_workflow.py       # drawn ROIs -> extract (mean | suite2p) | demix (masknmf)
│   ├── hpc/                  # submitit/SLURM runner for the suite2p pipeline (`mbo hpc`)
│   ├── gui/                  # The Studio (imgui + fastplotlib)
│   │   ├── gateway.py        # `mbo serve`: launcher + one studio_server process per browser session
│   │   ├── studio_server.py  # one Studio on a BrowserCanvas (canvas_server.py), served over http
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

`pip install -e .`, Python 3.12.7 to 3.13, `uv` for environments. Console scripts
`mbo`, `pollen`. Extras: `hpc` (submitit), `server` (uvicorn, websockets), `napari`,
`notebooks`, `docs`, `all`. Every base dependency is imported somewhere under
`mbo_utilities/`; a package one optional command needs is an extra, with the install
hint in its `ImportError`. Entry-point groups: `mbo_utilities.lazy_arrays` (readers),
`mbo_utilities.pipelines` (pipelines).

- The suite2p pipeline is a uv dependency group only: `lbm-suite2p-python` depends on
  the PyPI `mbo_utilities`, which `[tool.uv] override-dependencies` removes for a
  checkout and nothing can remove for a consumer.
- masknmf is a base dependency pinned to a commit of its `ui/launcher-pages` branch:
  the ROI tool draws with its imgui widgets (§7.6); the MaskNMF pipeline and
  `mbo reg-denoise` run it. Bump the pin when the branch moves; `uv pip install -e
  ~/repos/masknmf-toolbox` tests a local checkout. `import mbo_utilities` never
  imports it; only GUI modules and runners do.

## 2. Layer responsibilities

| Layer | Responsibility | Must not |
|-------|----------------|----------|
| `lazy_array` | `LazyArray` base, 5D accessors, `register_array_class`, `_dispatch` | Import numpy, tifffile, zarr, or any `arrays` module |
| `arrays/*` | Open one format lazily; deposit the source's metadata; implement the 5D contract (§5) | Normalize metadata; write files; import `gui` |
| `arrays/features` | Dims, tags, selection, ROI, phase, frame average, stats; format-agnostic | Know about any one file format |
| `metadata` | Canonical vocabulary, alias resolution, `OutputMetadata`, ScanImage parsing | Read pixels |
| `writer` + `_writers` | `imwrite`; emit canonical values under each format's keys; `ops.npy`; provenance | Hand-roll alias fan-out; every emitted key comes from the registry or `OutputMetadata` |
| `masknmf` `vnoiser` `roi_workflow` `hpc` | Run a pipeline from a settings dataclass; write its outputs (§7.4) | Import `imgui_bundle`, `fastplotlib`, or `mbo_utilities.gui` |
| `gui` views | Draw what the array answers (`results`, `motion_correction`, `behavior`) at the position the host's `Slice` and `Playhead` say is on screen (§7.6) | Read `iw.indices` itself; parse a pipeline's files; name a format |
| `gui/widgets/pipelines` | Draw a pipeline's config; spawn its worker task | Compute inline; hold pipeline math |
| `gui/tasks` + `gui/_worker` | Re-open the source in a subprocess and call the runner | Depend on GUI state; args are JSON |
| `gui/gateway` | Start, proxy and stop one `studio_server` process per browser session; browse folders | Import fastplotlib, `rendercanvas.auto` or any viewer; hold more than session bookkeeping |
| `cli` | Thin click wrappers over `imread`/`imwrite`/runners | Hold logic unreachable from Python |

If a change crosses a boundary, split it into two PRs or justify it in the description.

## 3. Coding standards

`STYLE.md` holds the rules ruff cannot check: dependencies, functions, comments,
logging, errors, scope of a change, indexing. `pyproject.toml` `[tool.ruff]` holds
the rules it can; `.github/workflows/format.yml` applies them on every push to
`main`. Gaps between the two and the code are listed under **Style** in `CONFORMANCE.md`.

## 4. Docstrings

Numpy style (ruff `pydocstyle` convention `numpy`). Content rules are in `STYLE.md`.

## 5. Lazy arrays: the 5D contract

`imread()` returns a `LazyArray`. Its rank and axis order are fixed.

### 5.1 Shape

- `.shape` is always `(T, C, Z, Y, X)`, `.ndim` 5, `.dims` `("T", "C", "Z", "Y", "X")`.
  Size-1 axes are kept. T = samples along time (§6.1), C = color channel (also a
  bucket for an arbitrary axis, e.g. camera or view for IsoView), Z = z-plane, Y =
  rows, X = columns: OME-NGFF 0.5 order.
- `nt`, `nc`, `nz`, `ny`, `nx` are the sizes by position; `num_timepoints`,
  `num_zplanes` the same by name through `dimension_specs`.
- A subclass implements exactly `_shape5d()`, `__getitem__`, `dtype`, `can_open()`
  and sets `self._metadata`. Everything else (`shape`, `ndim`, `dims`, `metadata`,
  `dimension_specs`, `dx/dy/dz/fs/finterval`, `slider_dims`, `squeeze`,
  `source_path`) is inherited from `LazyArray`. Subclass `LazyArray` directly;
  `Shape5DMixin` is a compatibility alias scheduled for deletion.
- Spatial extent may depend on state (`roi`, axial shifts, phase correction),
  temporal and channel extent on `frame_average` and `channel`, so `_shape5d()` is
  computed, not cached.

Pinned by `tests/test_lazyarray_contract.py`, `tests/test_shape5d.py`,
`tests/test_natural_rank.py`.

### 5.2 Rank inference on read

When a source carries no axis labels, rank alone decides the labels. This is the only
guess the codebase makes (`lazy_array._DEFAULT_DIMS_BY_NDIM`,
`features/_dim_labels.DEFAULT_DIMS`, `arrays/h5._DEFAULT_RAW_DIMS`,
`arrays/numpy._apply_dim_order`, `arrays/zarr._shape_tzyx`):

| Source rank | Labels | Reported 5D shape |
|-------------|--------|-------------------|
| 2D | `YX` | `(1, 1, 1, Y, X)` |
| 3D | `TYX` | `(T, 1, 1, Y, X)` |
| 4D | `TZYX` | `(T, 1, Z, Y, X)` |
| 5D | `TCZYX` | `(T, C, Z, Y, X)` |
| >5D | rejected | `ValueError` |

- 3D is time, never a z-stack; 4D is time+z, never time+channel. Anything else is
  declared, `imread(x, dims="ZYX")`, `dims="TCYX"`, unless the metadata getter sets it.
- Declared `dims` are characters from `TCZYX`, length equal to the source rank, no
  duplicates. An unusable declaration warns and falls back to the rank guess, never
  raises. The declared order is kept on `input_dims`; `.dims` still reports 5D order.
- `dims` travels in `reader_kwargs` so a worker re-opening the path reads it the same
  way (`source_reader_kwargs(arr)`).

Format-specific labels win over the rank guess:

| Source | Axis evidence used |
|--------|--------------------|
| ImageJ / Fiji TIFF | `frames`, `slices`, `channels`. A plain stack (`slices>1`, no `hyperstack`, no `spacing`) is read as **T**; `dims="ZYX"` keeps it as Z. tifffile's `imagej=True` default writes `channels=N`, read as C until `dims` says otherwise. Page order XYCZT. |
| Plain TIFF (no axis tags) | pages go on **T**; a three-letter `dims` (`"TYX"`, `"ZYX"`, `"CYX"`) puts the one populated axis on the axis named (`tiff._apply_dims_override`, `_declared_layout`). The Studio's Page Axis panel and `mbo view --dims` are the same `dims`. |
| OME-TIFF | `SizeT`, `SizeZ`, `SizeC`; page order from `DimensionOrder`. |
| ScanImage TIFF | `stack_type` from `si`: LBM (>2 saved channels) puts beamlets on Z and color ports on C; piezo puts slices on Z and volumes on T; single_plane has Z=1. |
| Multi-file TIFF | `planeNN` in the filename groups files onto Z; otherwise files concatenate along T; `roiN` yields one array per ROI. |
| Suite2p dir | `ops.npy` per plane dir, or suite2p 1.0's `settings.npy` + `reg_outputs.npy` + `detect_outputs.npy` read as one ops dict (`file_io.load_plane_ops`; `ops.npy` wins when both exist; raw binary `data_raw.bin` or `raw_data.bin`); plane dirs stack onto Z; T comes from the binary's file size, not `ops["nframes"]`. |
| MESc | `MethodType`: 1 timeseries `(T, C, 1, Y, X)`; 2 z-stack `(1, C, Z, Y, X)`; 6/7 linescan `(T, C, R, n_lines, width)`; 8 chessboard and 9/10 ribbon `(T, C, R, Y, X)`; 11 multicube has real depth on Z. `metadata["mesc_z_axis_meaning"]` says whether Z is `roi_index`, `depth` or `none`. |
| Bruker HDF5 | the dataset's HDF5 dimension labels (`t z y x c`); `element_size_um` in the stored order of the spatial axes. |
| lab4 H5 | `/imaging/data` is `TZYXC` (`TRYXC` for an AOD scan, ROIs on Z) unless HDF5 dimension labels say otherwise; claimed when `/imaging` carries `frame_rate_hz` (`Lab4H5Array`). |
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
- `arrays._base.temporal_mean(arr)` is the per-pixel mean over T as `(C, Z, Y, X)`
  float32; a reader that can do better overrides it (`MescArray` reads each raw
  channel once). The viewer's Mean Subtraction and Invert Deflection use it, never the
  Signal Quality samples, which are strided and binned.
- `arr.vmin` / `arr.vmax` are the display range of the representative frame.

### 5.4 Views and sanctioned exceptions

Read-time views wrap a 5D array and stay 5D: `FrameAveragedView` (temporal binning,
T // N), `PhaseCorrectedView` (bidirectional scan phase), `AxialShiftView` (per-plane
shifts; changes Y/X when enabled), `InvertedDeflectionView` (float32, `2 * mean -
frame` about the whole source's temporal mean; the `invert_deflection` read feature,
the viewer's Invert Deflection as data). `base_array(arr)` unwraps them for
`isinstance` checks.

Four objects deliberately report a different rank:

| Object | Reported shape | Why |
|--------|----------------|-----|
| `BinArray` | the shape you passed, usually `(T, Y, X)` | suite2p's input format; suite2p reads it 3D |
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
   `register_array_class` calls) is asked `can_open(path)` in descending `PRIORITY`;
   ties keep entry-point order; first `True` wins. Today: `IsoviewArray` 90,
   `MasknmfRunArray` 75, `ResultsArray` 70, `Lab4H5Array` 65, `MescArray` 60,
   `BrukerArray` 60, everything else 50.
4. Inputs no class claims (file lists, `.bin`, `.klb`, `.mp4`, `reg_tif/`, mixed
   directories) fall through to the legacy chain in `reader._imread_impl`.
5. A directory or list of files a class above 50 claims (Bruker h5, MESc: whole
   recordings each) opens one through that class, the first by name for a directory
   and the first listed for a list, and logs the rest (`reader._open_first_recording`);
   it never reaches a suffix reader that would guess the axes and never concatenates.
   A directory or list of suffix-level files (TIFF, plain h5) still concatenates
   through the legacy chain.

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
  `start-stop[-step]`; the suite2p `.bin` layout is the one exception (`zplane` first,
  to match `lbm_suite2p_python`). The same vocabulary (`TAG_REGISTRY` + `DIM_ALIASES`,
  `session` is `S`) reads tags back with `filename_tags` (`plane_03.bin` → `zplane03`,
  `mouse_V1_session1.tif` → `session01`; one-letter labels and unknown words such as
  `stan112` are not tags); the results zarr is named from them (§7.5). A new tag label
  goes in `TAG_REGISTRY`, never in a regex elsewhere.
- Every writer computes its metadata through `OutputMetadata(source, source_shape,
  source_dims, selections)` so `dz`, `fs`, `num_zplanes`, `num_timepoints`, `Lx`, `Ly`
  follow the selection (§6.6). Never re-stamp those by hand.
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
  those ROIs. `imwrite(roi_mode=RoiMode.concat_y | separate)`; `separate` fans out one
  `roiNN/` directory per ROI.

Pinned by `tests/test_selection_indices.py`.

### 5.8 Motion correction

`arr.motion_correction` is the shifts a stage applied, a `features.MotionCorrection`
or `None` (`LazyArray` answers `None`; a reader overrides it; nothing in `gui` names a
source):

| Field | Meaning |
|-------|---------|
| `source` | the stage, as a plot labels it: `RTMC` for an AOD scan, `suite2p` for a suite2p run, `masknmf` when its per-frame offsets land |
| `unit` | `um` for the AOD's real-time correction, `px` for a registration's offsets |
| `traces` | `{label: (t, shift)}`, `t` in seconds on the recording's T axis, one trace per axis; a label starts with its axis letter (`X`, `Z layer 3`) |
| `planes` | `{label: z}` for a stage that ran per z-plane; a label left out applies to every plane |

- `MescArray.motion_correction` is the `total` RTMC curves (`rtmc_motion`); the
  `intercycle` increments stay on `arr.rtmc`. A unit that armed RTMC without it ever
  moving reports `None`.
- `Suite2pArray.motion_correction` is each plane's rigid `xoff` / `yoff` from its
  `ops.npy`, in px, at `frame / fs` (frames without an `fs`); a plane with neither, or
  both all zero, gives no trace; no trace at all is `None`. One plane's traces are `X`
  and `Y`; a volume's are named after their plane dir (`X zplane03_tp00001-01574`) and
  listed in `planes`, and the Traces tab draws only the plane on screen
  (`MotionPlot.draw(z=slice.z)`, y refit on a plane change). Nonrigid block offsets
  (`xoff1` / `yoff1`) are not shown. `suite2p.registration_shifts` builds it for
  `Suite2pArray` and `Lab4H5Array`, whose `motion_correction` is the newest suite2p
  run beside its H5 (`<params uuid>/suite2p/plane<N>/`, `lab4.suite2p_run_planes`).
- GUI consumers: the Traces tab draws the shifts as a panel of masknmf's `TracePlot`
  over the traces (§7.6, `manual_roi._motion_lines`, interpolated onto the viewer's
  frames); `gui/imgui/motion.MotionPlot` draws them for the curation dashboard. Both sit
  behind one `MC` checkbox and
  neither clamps its x axis to the traces on disk: a run on a frame window leaves
  shorter traces than the recording.

Pinned by `tests/test_motion_plot.py`, `tests/test_mesc.py`,
`tests/test_suite2p_results.py`.

### 5.9 Behavior

`arr.behavior` is what the animal did, a `behavior.Behavior` or `None`, on the
recording's clock like §5.8, in the three parts NWB and Neo use:

| Field | Meaning | NWB / Neo |
|-------|---------|-----------|
| `signals` | `{name: BehaviorSignal(t, values, unit)}`: continuous measurements (`position` in mm, `speed` in mm/s) | `TimeSeries` / `AnalogSignal` |
| `events` | `{name: t}`: instants (`lick`, `reward`, `lap`) | events / `Event` |
| `epochs` | `{name: (n, 2)}`: start and stop of intervals (a reward zone, a trial) | `TimeIntervals` / `Epoch` |
| `sync`, `offset_s` | the imaging sync pulses on the logger's clock, and the logger time the recording started at (`sync[0]`, else 0); every other time is already shifted by it | |
| `source`, `subject`, `start`, `path`, `info` | logger and version, animal, wall-clock start, file, logger settings verbatim | |

- A behavior log is a file of its own, so no reader deposits it. `LazyArray.behavior`
  is a settable facet; `behavior.behavior_for(arr)` fills it on first use with
  `find_behavior(arr.source_path)`: a file a reader knows (`READERS`, by suffix) beside
  the recording, one folder up or in a sibling `behavior*` folder, named after the same
  subject and day (the first two `_` words, `u005a04_20260915`). The answer, found or
  not, is kept on the array.
- One reader: `behavior/tdml.py` for BehaviorMate's newline-delimited JSON. Treadmill
  position wraps at `track_length` (the speed unwraps it) and is zeroed after each
  lap's inter-trial interval (a jump above `MAX_SPEED_MM_S` is NaN in the speed). The
  reward valves are every valve but `sync_pin`; each context id is one epoch kind. A
  new logger is a function in `READERS`, nothing else.
- One GUI consumer: `gui/imgui/behavior.BehaviorPlot`, drawn over the trace panels by
  the Traces tab, x range held on theirs (`x_limits`), behind a `Behavior` checkbox.
  Three layers: epochs as translucent bands over the full height, the first signal on
  the left axis and the second on the right over the upper part, events as a raster
  strip along the bottom (`LANE_SHARE`: one lane per kind, a tick per event, the name
  at the left edge). Bands and lanes sit on a hidden third axis locked to lane units so
  zooming the signals never moves them; every layer is a legend entry. Never draw
  events as full-height lines: a few thousand licks bury everything. `shade_into` puts
  the same bands behind another implot plot; the Traces tab gives masknmf's
  `TracePlot` the epochs as its `span`s instead. It seeks the playhead like the motion
  plot.

Pinned by `tests/test_behavior.py`, `tests/test_manual_roi.py`
(`TestTracePlotView::test_a_recordings_behavior_rides_over_the_traces`).

## 6. Metadata: the canonical vocabulary

`arr.metadata` is a plain dict. One **canonical key** per physical quantity is
authoritative; every other spelling is an **alias** resolved by the registry in
`metadata/base.py::METADATA_PARAMS` (`MetadataParameter.canonical` is the registry's
own field name and stays).

### 6.1 Terms

- **Canonical key**: the registry key (`fs`, `dx`, `num_zplanes`). Consumers ask for
  it and only it.
- **Alias**: a key holding the same value verbatim (`PhysicalSizeX` ≡ `dx`), listed in
  `aliases`. **Transform alias**: a converted form (`finterval` = 1/`fs`,
  `XResolution` = 1/`dx`), listed in `transforms` as `(to_canonical, from_canonical)`.
- **Sampling rate** (`fs`, Hz): the rate of the array's T axis as reported: frame rate
  for a planar time series, volume rate for LBM and IsoView, per-light-path cycle rate
  for a MESc AOD scan, the binned rate after temporal binning. Invariant:
  `num_timepoints / fs` is the duration in seconds. "Frame rate" is a display alias.
  **Sampling interval** (`finterval`, s): `1 / fs`.
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

`IMAGING_METADATA_KEYS` is the subset the GUI metadata viewer always shows. Canonical
values are stored at full float precision; rounding is for display only.

### 6.3 Platform key map

What each platform stores and which canonical key it resolves to. A reader deposits
the left-hand keys; the registry does the rest.

| Quantity | canonical | ScanImage (`si.*`) | Femtonics MESc | ImageJ / Fiji TIFF | OME (TIFF XML, NGFF) | Suite2p `ops` | IsoView XML | lab4 H5 (`/imaging` attrs) | PrairieView / Bruker |
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
(register those keys as aliases at the same time). lab4 H5 is the file every
`lab4.convert` device writes; its attribute names are registered aliases.

### 6.4 Inbound: readers

1. A reader deposits the source's keys into `self._metadata` and does not normalize.
   `MescArray._build_metadata` and `H5Array.metadata` are the models: canonical keys
   where the reader computed the value itself, source keys verbatim otherwise, vendor
   extras under a namespace.
2. Never stamp aliases from a reader (`nframes`, `num_frames`, `num_planes`,
   `frame_rate`, ...); `normalize_metadata` emits them at write time. Never round at
   ingest. Keys the registry resolves through a transform (`finterval`,
   `XResolution`, `TimeIncrement`) are deposited as-is.
3. `num_timepoints`, `num_zplanes`, `num_color_channels`, `Lx`, `Ly` must agree with
   `_shape5d()`. Shape is the truth; if they disagree, fix the dict.
4. A reader's `metadata` getter returns the dict and does not mutate it.
5. Unknown quantities stay absent. `arr.dx` returns 1.0 and `arr.fs`/`arr.dz` return
   `None` when nothing was stored; the layer cannot tell "unknown" from a stored 1.0,
   so never store a placeholder.

### 6.5 Query: consumers

- On an array: `arr.dx`, `arr.dy`, `arr.dz`, `arr.fs`, `arr.finterval`,
  `arr.num_timepoints`, `arr.num_zplanes`, `arr.nt/nc/nz/ny/nx`. This is the
  sanctioned path; it stays correct when `dims` or `metadata` change.
- On a dict: `get_param(md, "fs")`, `get_voxel_size(md)`. `get_param` walks canonical
  → aliases → `pixel_resolution` tuple → transforms; for `fs`/`finterval` it goes
  through `resolve_effective_rate`, precedence `fs` > `frame_rate` > `finterval` >
  OME time scale > remaining aliases (`fps` last), warning once per divergent alias
  set. `get_canonical_name(name)` maps any spelling to its canonical key or `None`.
- Never `md.get("fs")`, `md["dx"]`, or a hand-written fallback chain. Every such site
  is a bug the layer exists to delete.

### 6.6 Reactive core

- `LazyArray.dimension_specs` builds per-dimension `{role, size, scale, unit}` from
  `dims + shape + metadata` (`features/_dim_spec.DimensionSpecs`), invalidated when
  `dims` is reassigned. Roles: `Y`/`X` spatial, `T`/`Z`/`C` iteratable, camera/trial
  batch.
- `OutputMetadata(source, source_shape, source_dims, selections)` is the only
  output-side layer: it scales `dz` by the Z stride, divides `fs` by the T stride,
  nulls `fs` for non-contiguous T, recomputes every count and `Lx`/`Ly`, repairs every
  rate alias present in the source, drops a carried OME block, and stamps
  `_metadata_provenance = {field: {base, stride}}` so a second hop never
  double-scales. Do not replace it with `DimensionSpecs.with_selections`.
- `FrameAveragedView` retimes through `scale_frame_rate(md, factor)`, which touches
  every registered rate spelling.

### 6.7 Outbound: writers

Emitted keys come from `OutputMetadata.to_dict()`, `to_imagej()`, `to_ome_ngff()`,
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

- `strip_for_export(md)` runs before any TIFF/H5/Zarr stamp: drops `EXPORT_DENYLIST`
  (suite2p registration internals, summary images, per-frame vectors, pipeline
  settings, mbo-internal keys) and any value over 8192 elements unless allow-listed.
  Suite2p-only fields live only in `ops.npy`.
- `rebase_provenance_paths(md, anchor)` repairs or drops recorded paths
  (`PROVENANCE_PATH_KEYS`) that do not exist on this machine. Never re-embed another
  machine's paths.
- `plane_shifts` / `plane_shifts_params` survive export on purpose; viewers align
  planes from them at read time.

### 6.9 Display

The GUI metadata viewer (`gui/_metadata.py`) reads canonical values through the
registry (`label`, `unit`, `description`) and groups the rest under vendor namespaces.
The metadata editor writes canonical keys only.

## 7. Pipelines

A pipeline (suite2p, MaskNMF, vnoiser, isoview, ROI workflow) reads a `LazyArray`
selection and writes an output directory. Every pipeline touches five surfaces; the
registration contract (§7.2) makes them one thing.

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
  use the suite2p convention `0` skip, `1` run, `2` force; each stage gates on its own
  native output file plus a settings hash, so a re-run with cached stages only redoes
  exports and figures.
- Heavy or optional imports (`masknmf`, `vnoiser`, `suite2p`, `torch`) are
  function-local inside the runner.
- Provenance: settings and source metadata (after `strip_for_export`) are written next
  to the outputs (`pipeline.json`, `mbo_provenance` attr) and `processing_history` is
  appended.
- Timing: the runner owns its step loop and closes every step (a plane, a scan's read,
  a domain's denoising, each write) with one INFO line carrying wall time, CPU time and
  process memory, one `processing_history` entry (`add_processing_step`,
  `duration_seconds`) and a `timing` summary (totals per step, per unit, peak memory)
  in the provenance; `timings.json` beside the outputs holds one flat row per step.
  `vnoiser/pipeline.py::_RunUsage` is the model; suite2p's `plane_times` and
  `hpc.write_timing_report` are the same record. Progress goes through
  `progress_callback(fraction, message)`, never a heartbeat line.

### 7.2 Registration contract

One class registers the whole pipeline. Subclass `PipelineWidget` and declare:

```python
class VnoiserPipelineWidget(PipelineWidget):
    name = "vnoiser"                       # selector label; unique
    is_available = HAS_VNOISER             # bool or property; cheap
    install_command = "uv pip install vnoiser"
    info = PipelineInfo(name="vnoiser", category="processor", ...)  # patterns + marker files
    axes_consumed = {"T": "range", "Z": "all", "C": "select-one"}  # see AXIS_MODES
    task_type = "vnoiser"                  # key in gui.tasks.TASKS
    task_func = staticmethod(task_vnoiser) # task_<name>(args, logger)

    @classmethod
    def applies_to(cls, arr) -> bool: ...  # safe before instantiation; arr may be None
    def draw_config(self) -> None: ...
```

and list it in `pyproject.toml`:

```toml
[project.entry-points."mbo_utilities.pipelines"]
vnoiser = "mbo_utilities.gui.widgets.pipelines.vnoiser:VnoiserPipelineWidget"
```

`load_entry_point_pipelines()` registers `info`; `load_plugin_tasks()` registers
`task_type` → `task_func`; the Run tab lists the class. Built-in and third-party
pipelines use the same path; nothing is hardcoded by name.

- `axes_consumed` values: `range` (user picks start:stop), `all` (whole axis, row
  disabled), `none` (axis hidden), `select-one` (single index). Default `range`.
- `PipelineInfo.category` is `reader`, `writer`, `processor` or `segmentation`.
  Readers register their own `PipelineInfo` at import (category `reader`); a
  pipeline's info lives on its widget.
- `extracts_traces = True` + `extract_traces(movie, labels)` opts the pipeline into
  the manual-ROI "Extract trace" action.
- A pipeline's widget is one object per host (`pipelines.pipeline_instance`), drawn
  by the Process tab; `open_pipeline(host, name)` selects it there and brings the tab
  forward. A widget never opens a window of its own or assumes what it is drawn in, so
  the Process tab can move into a window and back. The host calls every built widget's
  `poll()` once a frame whatever tab is showing; a widget that watches a worker it
  started does it there, never in `draw_config`.
- `seeds_from_view = True` + `seed_from_view()` sets the widget's selection to what
  the viewer shows (the recording on screen, the slice its sliders are on);
  `open_pipeline(..., seed=True)` calls it first, and `quick_pipelines(host)` lists
  the pipelines that apply and set it, which is how the MESc tab header and `Shift+P`
  offer "vnoiser on MUnit_3" without naming a pipeline.

### 7.3 Input contract

A worker task receives a JSON dict and must rebuild exactly what the user was looking
at:

| Key | Meaning |
|-----|---------|
| `input_path` | `arr.source_path` |
| `reader_kwargs` | `source_reader_kwargs(arr)`: `unit`, `dataset`, `dims`, `frame_average`, `channel` |
| `timepoints`, `planes`, `channels` | 1-based selection from `to_lsp_kwargs` |
| `fix_phase`, `use_fft`, `phasecorr_method`, `mean_subtraction`, `invert_deflection` | read features, applied by `apply_read_features` |
| `output_path` | output directory |
| `settings` | `Settings.to_dict()` |
| `_uuid`, `_log_file` | injected by `ProcessManager.spawn`; read by `TaskMonitor` and `setup_logging` (§8.3) |

The worker does `arr = imread(input_path, **reader_kwargs)`, then
`apply_read_features(arr, args)`, then calls the runner. Runners take a `LazyArray`
or a path and the same 1-based selection kwargs; they never take GUI objects. The
selection keys keep the 5D names whatever the axis means for the source; the runner
translates through the array's metadata, never the widget. On a MESc AOD unit Z is
the ROI index (`mesc_z_axis_meaning == "roi_index"`), so `planes` are the lines or
patches to process: the vnoiser runner reads only those, cuts every domain to them,
drops an emptied domain, and records `planes` and the given `domains` in the
provenance source block. `mbo vnoiser -p` sends `planes`; the vnoiser tab sends
domains instead (`axes_consumed` Z and C are `none`; the channel is the Trace
block's).

**vnoiser domains.** A domain is the pixels of one or more ROIs averaged into one
trace (`analysis/roi_traces.py`, GUI-free, no vnoiser import). An ROI is one Z
index: on an AOD unit a line, patch or box cropped to `mesc_roi_extents` (a ragged
box of a Y-flipped layout sits at `y_start`); elsewhere the plane's whole frame, so a
single-patch chessboard or a raster is a scan too (`arrays.mesc.is_trace_unit`: at
least `MIN_TRACE_FRAMES`). `domains.json` and the task's `domains` take an ROI list
(every pixel; the archive's `domain_ROInumber`) or `{"rois", "pixels", "fraction" |
"mask"}`: `bright` / `active` / `correlated` keep the top fraction of each ROI's
pixels by mean, high-passed SD or correlation with the ROI's mean (scored on
`score_seconds` from the middle of the frames read); `mask` lists flat ROI-local
pixels. `extract_traces` reads each ROI once and applies `VnoiserSettings.traces` in
order: conversion offset, background (`dim`, `outside`, a named `domain`), reference
channel (`ratio`, `regress`), inversion. dF/F and z-score are
`VnoiserSettings.dfof` (`analysis/dff.dfof_zscore`; `gaussian` is
`vnoiser.preprocess.domain_zscore` bit for bit). Polarity (`POLARITIES`): `flip_z`
(vnoiser negates z; the archive's), `invert_f` (`traces.invert`: `2 * mean - F`
before dF/F) or `none`; `VnoiserSettings.polarity` raises when both are set and the
runner reads it first. The results' `raw` kind is each domain's trace before
inversion, so the Traces tab's own Invert Deflection is the only flip applied on
screen.

**The vnoiser widget** holds no trace math (the preview is
`vnoiser.pipeline.preview_domain` on a `TracePreview` thread). Its domain table
(`DomainRow`: name, ROIs, pixel rule, mask, `run`) fits the scans ticked from the
screen (`_fit_domains`): one domain per ROI unless a loaded or edited table names
only ROIs those scans have; a previous run's uncut `domains` seed a scan only when it
has as many ROIs. `seed_from_view` ticks `run` on the domains holding the ROI on
screen, adding one when the table leaves it out; the task gets the ticked domains
and the background domain. `drawn_domains` turns the ROI tool's ROIs into mask
domains (`Domain.from_frame_mask`). The viewer's Invert Deflection seeds polarity
`invert_f` and stays display only.

**Runs and the viewer.** `MescArray.results` is the newest vnoiser run of the file
that processed the unit on screen (`vnoiser_run_for_mesc`, `unit_for_source`), never
the container's. Saved per-ROI traces (`linescan_viewer.saved_roi_traces`) are each
ROI's raw mean from that run (`members/raw`, photon counts, same channel), never a
domain's denoised trace; the curation window shows raw recordings inverted
(`EventCurationWidget.raw_inverted`, default on, `/inverted` in the recording id)
because vnoiser's dashboard looks for peaks. `ManualRoiWidget._load_array_results`
loads a run as one row per domain with every kind written (`_line_unit_rows`; member
lines are the recording's own rows; `extra["source_unit"]`; a frame-window run
carries `RoiTrace.frames`); `load_results` (Load into Traces,
`_adopt_finished_runs`, a `--pkl` PF folder) loads only the scan that processed the
unit on screen. Domains never stand in for the recording: the ROI slider plots the
line's own mean (`StandardTraces`) and the domain holding it is the Traces panel's
**vnoiser** panel (`ManualRoiWidget._vnoiser_rows`); picking a domain in the trace
table moves the slider to its first line; with no run loaded, **Run vnoiser** opens
`open_pipeline(host, "vnoiser", seed=True)`. A run the vnoiser tab started loads
into the Traces tab when its worker finishes (`VnoiserPipelineWidget.poll` →
`_load_traces`, which turns Manual ROI Labeling on).

**Help page.** `gui/imgui/vnoiser_help.draw_vnoiser_help(is_open, keys_open)`,
opened by the vnoiser window's `vnoiser guide` button, the curation window's `guide`
(`h`), or `python -m mbo_utilities.gui.imgui.vnoiser_help` (`--frames N
--screenshot out.png`). It holds the curation window's `KEYBINDS`, imports neither
vnoiser nor `gui._setup`, and draws an opaque window whatever the host's style.
Every number it quotes is a `VnoiserSettings` default and every colour the curation
window's; `tests/test_vnoiser_help.py` says which guide line to change with them.

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

Filenames are matched; semantics may differ and the pipeline wins. Pipeline-specific
files keep their own names (`demixing_results.hdf5`, `PF/`, `norm_traces.npy`).

MaskNMF is the exception: it writes masknmf's own run folder per plane and no movie,
so `masknmf view`, `masknmf.io.OpenedResults` and the masknmf launcher open its runs
as masknmf's:

```
<save_path>/
  20261007T183740_masknmf_zplane01_tp00001-01574/
    results.hdf5        RigidRegistrationArray + RigidMotionCorrector, CompressionArray,
                        DemixingResults; each stage group's mbo_provenance attr
    config.json         masknmf.io.write_run_config: pipeline mbo_utilities.masknmf,
                        configs (MasknmfSettings), inputs.movie (path, reader_kwargs,
                        read_features, plane, z, c, frames | tp_indices, fs), run, timings
    <folder name>.log   masknmf's log
    alignment.hdf5      the denoised copy registration ran on (Denoise before registration)
    <stem>.<stamp>.masknmf.zarr   the results file (§7.5) when demixing ran
    *.png               QC figures
```

- `MasknmfRunArray` (`imread(<run folder>)`) re-opens the recording from
  `inputs.movie` and replays the stored shifts, so the registered movie is never on
  disk; its `results` is the zarr, its `motion_correction` the shifts. With Invert
  Deflection the runner registers the movie as recorded and inverts the registered
  movie for compression and demixing (`2 * mean - x` holds a still mean against a
  moving frame, so registering it leaves motion behind);
  `inputs.movie.registered_before_inversion` makes the reader replay the shifts the
  same way.
- **Denoise before registration** (`registration.denoised_reference`) estimates the
  shifts on a quick PMD copy of the raw movie and applies them to the raw frames. A
  stage set to Run copies its group from the newest earlier run folder of the same
  plane when the group's provenance (settings hash and input) matches.
- **View movies** opens `gui/reg_denoise_viewer` on the newest run folder in its own
  process; `RegDenoiseRun.open` takes a run folder or its `results.hdf5`
  (`from_run_folder`: raw and registered from `MasknmfRunArray`, `pmd(raw)` the
  alignment copy, `pmd(registered)` the compression stage, the recording's RTMC) or a
  reg-denoise file. A viewer that dies leaves its traceback in
  `~/.mbo/logs/<stamp>_movies_<run>.log` and its last line in the widget's status.
  `mbo <run folder> --qc` and the launcher's **Registration QC (MaskNMF run)** open
  the same viewer; without `--qc` a run folder opens in the Studio
  (`MasknmfRunArray`), or in masknmf's demixing viewer when its `results.hdf5` holds
  a `DemixingResults` (`arrays.masknmf_run.run_demixing`).
- `mbo reg-denoise` (`masknmf/reg_denoise.py`) is pre-registration denoising for one
  channel: the raw movie's PMD (`raw/CompressionArray`), the rigid registration
  estimated on it, and the registered movie's PMD (`CompressionArray`) in one
  `results.hdf5` whose root attrs `mbo_pipeline = "reg_denoise"` and
  `mbo_provenance` (source, unit, channel, first frame, fs, settings) let `mbo <file>`
  open it in `RegDenoiseViewer`: the four movies on one time axis under masknmf's
  `TracePlot` (RTMC, shifts, frame means, a rectangle ROI).
- **Spike-triggered average** (`analysis/spike_average.SpikeSource`): one ROI of any
  run's `results` (§7.5) over the movie it was measured on (the run array's own movie
  at the unit's z, or a line unit's first line, cut to the frames its source block
  names; a MaskNMF run adds its compressed movie). Spikes are the unit's `events`, the
  events a `.curation` file beside the run accepts for `results.recording_id(unit,
  roi)`, or a threshold on one of its traces; motion is every `motion_correction` the
  movie went through, each on its own clock (`MotionCorrection.at`).
  `gui/spike_average_viewer` draws it like Registration QC (an `NDWidget` over the
  lag, `TracePlot` linked to it) and only runs in its own process
  (`launch_spike_average`, watched by `gui.launch.LaunchedWindow`): from the Traces
  tab's **Spike average**, the MaskNMF tab (newest run that demixed), the vnoiser tab
  (scan and ROI on screen), the launcher's **Spike-triggered average** entry and
  `mbo spike-average <run>`.

`hpc/` runs the suite2p pipeline only (`lbm_suite2p_python.pipeline`), configured by
`hpc.toml` (`[io]`, `[slurm]`, `[pipeline]`, `[parameters]`); a second pipeline gets
HPC support by exposing `run_volume(arr, save_path, **selection)` and a `pipeline`
key in the config.

### 7.5 The results zarr

Native outputs differ per pipeline (§7.4's suite2p files, the vnoiser pipeline's `PF`
pickles). `mbo_utilities/results.py` fixes the one shape they all mold into, the
standard output format; a pipeline's native files stay its cache and compatibility
layer.

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

- **Naming.** `results_name(source, when, extra_tags, pipeline)`: the source
  filename's stem, `extra_tags`, the timestamp, the pipeline, dot separated
  (`session1.mesc` → `session1.2026-09-21-14-30-22.vnoiser.zarr`), written beside its
  input. The stamp is local time to the second (`RESULTS_STAMP`), so a rerun is a new
  file and a listing sorts chronologically. Nothing matches a results name by
  pattern: `results_stamp(path)` reads the stamp back with `datetime.strptime`,
  `newest_results(folder, pipeline)` picks the latest run, and a run belongs to the
  recording its source block names (`results_source(path)`), so `newest_results(...,
  source=file)` and `vnoiser_run_for_mesc` never hand one `.mesc` another's run; a
  run naming no recording (the archive's PF folders) counts for any. Unit groups keep
  the §5.6 vocabulary (`unit_name("plane", 1)` is `zplane01`; scans are `scan<id>`).
- **Building.** A pipeline builds one `ResultUnit` per plane or scan, constructs
  `Results(pipeline=..., units={name: unit}, source=..., settings=..., metadata=...)`
  and calls `results.write(path)`. `Results.from_suite2p(dir, plane_dirs=None)` reads
  suite2p and MaskNMF folders into the same shape without writing: `F` → `raw`,
  `Fneu` → `neuropil`, `spks` → `spikes`, `norm_traces` → `dff` (or `zscore` per
  `norm_method`, `norm_traces_kind`), `stat` → pixel members,
  `meanImg`/`max_proj`/`Vcorr`/`refImg` → images (cropped ones put back into the full
  frame at `yrange`/`xrange`); ops through `load_plane_ops`; ROIs with no frame size
  raise. Plane dirs go in plane-number order (`zplaneNN` 1-based, vanilla `planeN`
  0-based) or the order given; a dir without `stat.npy` + `F.npy` gives no unit;
  every unit records `attrs["plane_dir"]` and `attrs["z"]`. `Results.from_pf(dir)`
  reads the vnoiser pipeline's `PF` folder (domains are the ROIs, their lines the
  members with `members/raw`, `test.h5` gives `dff` and `zscore`,
  `traces/scan<id>_f.npy` each domain's `raw`, peaks are the events; with
  `runtime.denoise` off the tables come from `pipeline.json`). Copy one of them for a
  new pipeline; never invent a trace or image kind, add it to the registry.
- **Writing.** It is the vnoiser pipeline's default
  (`VnoiserSettings.runtime.output_format == "zarr"`): the run works in a
  `<name>.work` scratch folder, writes the results file beside the input, moves
  `test.h5`, `traces/`, `pipeline.json` and `timings.json` into its `vnoiser/`
  (`pipeline_files`) and deletes the scratch folder. `mbo vnoiser --pkl`
  (`output_format == "pkl"`) writes the archive's PF folder instead. `mbo results
  <dir>` converts an existing suite2p, MaskNMF or PF folder.
- **Reading.** One generic reader. `Results.read(path)` reads a results file back
  (`.units[name]` → `ResultUnit`, every array in memory); `Results.open(path)` is the
  one door that takes a file, a folder holding one, or a pipeline's native folder.
  `imread` returns a `ResultsArray` (`results.py`, `PRIORITY` 70) for a results file,
  a vnoiser `PF` folder, or a folder holding one (`results_dir_of(path)`); its image
  is the recording when `results.source` names a reachable file, else a trace raster.
  No pipeline gets an array class of its own: a run's units, traces, ROIs, events and
  images are on `arr.results` (`LazyArray.results`, settable, None by default).
  `ResultsArray` sets it to what it opened; `Suite2pArray` answers
  `Results.from_suite2p` of its own plane dirs, read once, in the array's z order with
  `attrs["z"]` the 5D z (None for a registration-only run); `Lab4H5Array` the newest
  suite2p run beside its H5; `MescArray` the newest vnoiser run that processed its
  unit (§7.3). `results_pipeline(path)` and `results_summary(path)` read only
  `zarr.json` files and are what `can_open` and the run scanners use; `ZarrArray`
  declines a results file; `imread` never returns a results file as a movie of
  itself. No widget parses a plane dir: it asks the array for `results`.
- **Viewing.** The ROI widget takes every run through one door:
  `ManualRoiWidget.load_run(path)` opens a results file, one unit of one
  (`<file>.zarr/zplane01`), a suite2p plane dir, a masknmf run folder or a vnoiser
  `PF` folder with `Results.open`. A pixel unit becomes a `roi_runs.DerivedSet` (the
  unit, its footprints as `(ypix, xpix, lam)`, its `accepted` flags from `iscell`),
  keyed by `attrs["plane_dir"]` or `<file>/<unit>`; the run's files are never written
  to: accept / reject, marks, labels and group colors are curation and stay with the
  ROIs (the rois zarr's `runs` list, §7.6 Persistence). The array's own `results`
  load the same way when the widget opens (`_load_array_results`), one derived set
  per plane on its own z, `spks` as `spikes` and `iscell[:, 1]` as the ROI table's
  `prob` column. A manual run (pipeline `manual`, §7.6 Run coordinates) whose unit's
  `uids` name drawn ROIs still in the store merges into the trace table instead
  (`_merge_run_traces`, one `roi_runs.unit_trace` row per uid). A line unit becomes
  an `external` `TraceSet`: one row per ROI (`denoised`, else `dff`, else `raw`) and
  one per member line, named by ROI (`roi3`, `roi3 (raw)`, `roi3 line 12 (raw)`),
  carrying `fs`, the line on `z` (`extra["line"]`) and the pipeline's channel on
  `c`. `roi_runs.run_dir_complete` and `scan_run_dirs` treat results files as run
  dirs, the rois zarr's `runs` list restores them, and a finished `vnoiser` worker is
  adopted like a suite2p one, so a new pipeline that writes the results zarr reaches
  the Traces tab with no GUI code.

Pinned by `tests/test_results.py`, `tests/test_suite2p_results.py`,
`tests/test_vnoiser_pipeline.py`.

### 7.6 Manual ROIs

Hand-drawn ROIs are a GUI-free, observable session model in
`mbo_utilities/annotation/` (`events.Observable`, after fastplotlib's
`GraphicFeature`: a view subscribes with `add_event_handler(fn, "rois")` and redraws
from the event). `gui/manual_roi.ManualRoiWidget` is one view of a `RoiModel`: the
glue (`__init__`, slider and playhead wiring, `add_roi`, loading runs as derived sets,
promotion, rois zarr persistence) over one mixin per concern in `gui/roi/`:

| Module | Mixin | Holds |
|--------|-------|-------|
| `_shared` | | the constants, `ROI_KEYS` / `KEYBINDS`, the help text, `_PlaneOrder`, `_RunSpec`, `unit_key`, `help_markdown` |
| `keys` | `KeysMixin` | `handle_keys`, `_keys_live`, the per-frame `_frame` hook |
| `overlay` | `OverlayMixin` | the mask / contour overlay, region drawing, pointer picking, `set_color_by` |
| `selection` | `SelectionMixin` | selection, the group buffer, labeling, marks, delete, undo |
| `jobs` | `JobsMixin` | quick and pixel traces (`measure_mean`), `run_rois` / `_run_job`, job polling, run adoption |
| `traces_panel` | `TracesPanelMixin` | `draw_traces`, `_sync_plot`, the motion and behavior rows, kind and dF/F settings |
| `trace_table` | `TraceTableMixin` | `draw_trace_table` and its rows, cells, sorting |
| `curation` | `CurationMixin` | `draw_rois`, `draw_tab`, the labels, on-disk and exchange rows, the filter, `exchange` |

A mixin has no state of its own and no `__init__`; every attribute is the widget's. A
test that patches a module global (`measure_mean`, `demix_rois`) patches the mixin's
module (`gui.roi.jobs`), where the call lives. Names other modules import
(`PANEL_HEIGHT`, `KEYBINDS`, `attach_roi_widget`, `labels_path`, ...) are
re-exported from `gui/manual_roi` and listed in its `__all__`.

| Object | Holds | Emits |
|--------|-------|-------|
| `RoiLabelStore` | one `RoiRecord` per ROI (flat `pixels` and `weights` on one `plane`, `area`, `class_index`, `note`, `uid`, `source`, `color`, the `polygon` as drawn), so ROIs may overlap; `label_image(plane)` derived on demand; the label-name set, `plane_axes` | `rois` with `action` `add` `delete` `clear` `restore` `class` `note` `color` `labels` `tint` |
| `RoiTraceTable` | one `RoiTrace` per measurement | `traces` with `action` `add` `remove` `clear` |
| `RoiModel` | the two above plus the slider position (`view`, `plane`, `z`, `c`) | the two above forwarded, plus `view` when the plane changes |

- **Planes.** `RoiRecord.plane` is the flat index into the volume; `plane_axes`
  records which scrolling dims key planes (`(("c", 2), ("z", 3))`, z last, so a
  z-only store keeps `plane == z`). The store owns the arithmetic (`plane_of(pos)`,
  `plane_pos(plane)`, `plane_label(plane)`, `roi_z(i)`, `roi_c(i)`). Axis roles
  resolve through `find_slider_name`, so IsoView's `Cam` / `Zplane` sliders key planes
  too. T never keys a plane.
- **Traces.** A drawn ROI's trace is keyed `("roi", uid, z, c, engine)` (`ENGINES`:
  `mean`, `suite2p`, `masknmf`): the same measurement again replaces the row; another
  channel, z-plane or engine is another row. Rows that stand for no drawn ROI (an
  algorithm's component, a results file's line) are keyed `("member", source, k)` and
  are never pruned by ROI deletion. `frames` is the `(start, stop)` window read,
  `frame_average` the binning, `source` the run or origin. `RoiTrace.name` is a row's
  one display name; the table, the legend and the sort key read it, nothing formats
  its own.
- **Trace display** (`annotation/display.py`): the pipeline that produced a row
  decides how it is shown, through one `TraceProfile` per pipeline (`TRACE_PROFILES`,
  keyed by the row's `engine`, which a results file sets to its `pipeline`; a plugin
  calls `register_trace_profile` beside its `PipelineInfo`; else
  `DEFAULT_TRACE_PROFILE`): the kinds its rows can show (`DISPLAY_KINDS` =
  `TRACE_KINDS` plus `SUBTRACTED`, `raw - neuropil`, computed here and stored
  nowhere), the kind shown first (`vnoiser` opens on `denoised`, the rest on `dff`),
  the `DffSettings` for a dF/F and z-score computed from a raw row (`DFF_METHODS` in
  `analysis/dff.py`, all sized in seconds; a row with no `fs` gets a static
  percentile), whether a stored dF/F is percent (`masknmf`) or a fraction (`vnoiser`,
  `suite2p`; the plot is always percent), the raw `unit`, and, only when the pipeline
  measured a real `Fneu` (`suite2p`, `mean`; never `masknmf`, whose `Fneu` is zeros),
  `neuropil` and `SUBTRACTED` with its `neuropil_coeff`. A row carries every kind in
  `kinds` (`RoiTrace.array(kind)`), and the panel and the trace table read every row
  through `display_trace(trace, kind, settings, subtract, invert, flip)`; nothing
  else decides what is subtracted or inverted (no neuropil checkbox, no second line
  under a trace). `raw` is `F`, `neuropil` is `Fneu`, `SUBTRACTED` is `F -
  neuropil_coeff * Fneu`, `dff` and `zscore` the row's own else `computed_dff` (from
  `SUBTRACTED` when the row has a neuropil, else `F`) and `computed_zscore` of it;
  `displayed_kind` and `y_label` (the row's `extra["unit"]`, else the profile's)
  report what was drawn. Deflection (`deflect`, about the row's own mean over T) is
  the viewer's Mean Subtraction and Invert Deflection unless the panel's **Trace**
  popup overrides them (`trace_subtract`, `trace_invert`: `viewer`, `off`, `on`,
  resolved by `deflection()`); it applies to measured kinds and to a dF/F or z-score
  computed here, never to a pipeline's own kinds. `flip` negates whatever is shown
  and prefixes the label `-`. The plot is masknmf's `TracePlot` (**The stack**
  below): one line per row on the viewer's frames (`_on_frames`, interpolated from
  the row's `trace_axis`, NaN outside it, decimated to a min/max band above the
  panel's pixel count), its background set by `_TRACE_MODES`.
- **Run coordinates.** `RoiModel.targets(indices, z=, c=)` says where each ROI is
  read: the mask always from the plane it was drawn on, the pixels from `z` / `c`
  when given, else from where it was drawn. The widget's `run_where` is `drawn`,
  `screen` (the slice on screen when the run starts) or `fixed` (`run_z`, `run_c`);
  `run_tp` cuts T (`PlaneMovie.select`). A run is one results zarr (§7.5), pipeline
  `manual`, `<stem>.<tag>.<stamp>.manual.zarr` in the ROI folder
  (`roi_workflow.run_path`; the stamp advances a second while the name is taken):
  `run_rois` groups targets by `(plane, z, c)`, `extract_rois` or `demix_rois` makes
  one `ResultUnit` per group named by its slice (`slice_name`: `zplane02`,
  `ch01_zplane02` when more than one channel was read), and `write_run` writes them
  with the recording's path, reader kwargs and unit in the source block. Unit `attrs`
  record `engine`, `plane` (store plane), `z`, `c`, `frames` / `tp_indices`, the
  `uids` and `store_indices` of its ROIs (`roi<uid>` names); `members` are the
  store's pixels and weights (masknmf's own footprints after demixing, with
  `n_components`). `DerivedSet.read_z` / `read_c` / `frames` / `engine` read them
  back; `roi_runs.unit_trace` turns a row into a trace. masknmf's native files land
  in the zarr's `manual/` folder (`results.pipeline_files`); a plane dir's PMD cache
  stays in the plane dir.
- **masknmf's curation viewer.** The ROI tool is laid out, keyed and styled as
  masknmf's `SingleSessionDemixingVis`, drawn with `masknmf.visualization.imgui`
  widgets (`TracePlot`, `draw_roi_table`, `RoiOrder`, `grid`, `section`,
  `draw_switch`, `help_mark`) and `masknmf.visualization.rois.FootprintSet`. The keys
  are one table, `ROI_KEYS`: masknmf's `DEMIXING` binds and labeling keys plus this
  tool's own; while the tool is on it claims `m` `c` `p` `h` `k` `o` and the arrows
  from the viewer's global shortcuts (`_keyboard.claim_keys`). Draw (`a`) arms a
  fastplotlib `PolygonSelector`; Add ROI (`r`) fills it into the store (`add_roi`);
  pixel traces (`p`, `add_pixel`) are `"pixel"` rows of the movie's 5x5 mean; delete
  (`d`) removes a drawn ROI and marks an algo row (`DerivedSet.discarded`, kept
  listed and pinned first until unmarked); every add, delete, mark and deselect is
  one Ctrl+z step (`_undo`). A click picks a drawn ROI before an algo one; ctrl
  toggles a row in the group (`buffer`), shift adds; a group plots one line per
  member in its `GROUP_COLORS` colour, shared by mask and table row. The ROIs tab is
  masknmf's Tools panel (Full FOV, guide `h`, keybinds `k`; a Curation tab with
  OVERLAY, SELECTION, LABELS, RUN; an ROIs tab with masknmf's filter and table); the
  Traces tab is the trace table in `draw_roi_table`.
- **Draw -> run.** `ManualRoiWidget(auto_trace=True)` traces every ROI the moment it
  is added (its mean at the run coordinates); no tab is selected for the user. The
  RUN section, `shift+t` and the Process tab run ROIs the way the widget is set
  (`engine`, `run_where`, `run_frames`, `run_tag`).
- **Fiji and CSV exchange.** `annotation/imagej.py` (`roifile`, a base dependency) is
  the one place ROIs and traces leave or enter in another tool's format:
  `export_roiset` writes every drawn ROI as its polygon (`mask_outline` for a
  promoted mask) into a `RoiSet.zip` Fiji's ROI Manager opens (`c`, `z` 1-based,
  `t_position` 0 so it shows on every frame); `import_roiset` fills each polygon on
  the plane its positions name, ImageJ name as the note; `export_label_tiff` is the
  label image as an ImageJ hyperstack (`slices`, `channels` from `plane_axes`);
  `export_traces` is one CSV, `time_s` or `frame` then one column per row named by
  `RoiTrace.name`; `export_footprints_npz` is masknmf's `spatial_footprints` seed
  file. The ROIs tab's exchange row runs them through
  `ManualRoiWidget.exchange(kind, path)`; an import is one undo step. Nothing else in
  `gui` imports `roifile` or writes a TIFF of ROIs.
- **Any slice.** The viewer's sliders are the array's T, C, Z axes by position
  (`_dim_labels.slider_roles`, the table the viewer names its sliders from and
  `resolve_dim_labels` uses), whatever the array labels them (`Timepoint` /
  `Channel` / `ROI` for a MESc AOD unit, `Tile` / `View` for IsoView); the widget
  hands the store `axis_roles = {"z": <slider>, "c": <slider>}` so `roi_z` / `roi_c`
  and every run decode the right slice. Every `LazyArray` is indexable in y and x
  (§5.3) and `PlaneMovie` reads only `arr[t, c, z, y, x]`, so no reader needs
  anything more for the ROI tool. Wherever the widget names an axis it uses the
  slider's own word (`ManualRoiWidget.axis_label`): on an AOD unit the Z axis reads
  `ROI 3`, never `z3` (`mesc_z_axis_meaning == "roi_index"`).
- **Line positions.** Where a MESc scan line or patch sits is
  `arrays/mesc_geometry.py`'s business and nothing else re-derives it: the segment the
  AOD scanned (`CoordinateMapJSON.maps[0].driftEndPoints`, or `contours` for patches;
  never the shorter hand-drawn `guideLine`, never `ROIJSON`, whose vertices are in the
  unrolled scan's pixels) in the file's absolute micron frame, where
  `ReferenceViewportJSON.geomTransTransl` is the `[0, 0]` corner of an image, rows
  grow with +y, and a stack's slice `k` sits at `geomTransTransl[2] + MinZ + k·step`
  (verified on two rigs: brightness ranking of lines on their own snapshot, and 0.86
  cross-correlation at zero offset against the stack on the 2026-09-14 pictures).
  `line_positions(path, unit)` gives each ROI `start_um` / `end_um`, `z_um`,
  `length_um`, `sample_um`, `dz_um` against the snapshot the lines were drawn on;
  `roi_placements` puts them on a stack's slices; `image_overlays` draws them. An AOD
  unit's trace-table rows (`linescan_viewer.StandardTraces`, for every `ROI_LAYOUTS`
  layout and the `PATCH_MODALITIES`: the mean of each line or patch in photon counts,
  computed in the background unless `F.npy` or the vnoiser run's line means exist)
  are keyed `("member", "<MUnit> lines", k)` or `("member", "<MUnit> patches", k)`,
  carry `z = k`, `c` = the channel read, and the position in `extra` (`line`, `z_um`,
  `dz_um`, ...; `POSITION_KEYS`); the label is the ROI alone. Depth is never a
  column, label or caption: the GUI shows where a line was drawn, not how deep it
  sits. Hovering a row lists the line's ends, length and sample spacing
  (`ManualRoiWidget._line_position` over `MescArray.line_positions`, the reader's
  cached facet; a results file's line rows get theirs stamped at load).
- **Reference image.** `gui/mesc_reference.py` shows where an AOD unit's lines or
  patches were drawn: `reference_images(mesc, open_unit, c)` (GUI-free) is the
  picture the ROIs were drawn on (`background_unit`), then every Z-stack holding them
  (`zstack_contents`), each a max projection in channel `c` (at most `MAX_ELEMENTS`
  samples) carrying the unit's `image_overlays`. A Z-stack projects **only over the
  slices its ROIs were scanned on**, and `image_overlays` keeps an ROI only when its
  outline falls inside the stack's field **and** depth range (`roi_placements`'
  `in_range`); one scanned above or below is left out, never clamped onto an edge
  slice (2026-09-14 rig: a chessboard box 720 um under the only stack landed on slice
  0 over unrelated cells). So `on_plane` on a stack is always True, `zstack_contents`
  lists only stacks that really hold a scan, and `line_positions` reports
  `stack`/`slice`/`in_stack` as None when no stack was scanned around the line.
  `ReferenceView` is that set in a `SummaryImageViewer` popup (masknmf's full-FOV
  viewer; `roi_provider(key)` returns `(points, rgba, thickness)` polylines, MESc's
  colours, every ROI dimmed to `ON_ALPHA`, the slider's ROI drawn last at
  `SELECTED_THICKNESS` over a `HALO_THICKNESS` halo). Each `ReferenceImage` carries
  its `unit` and, for a stack, the `slice` its ROIs sit on, so the popup's one button
  (`on_show(unit, slice)`) displays that unit at that slice; a click on a line
  (`SummaryImageViewer.on_pick`, `ReferenceView.pick`) moves the ROI slider to it.
  The MESc tab opens the popup from the picture cell, reopens it on a unit switch and
  hands the host `reference_view` (a callable) the Traces panel offers as a button.
  Nothing draws ROIs on the viewer's own image; no projection of the unit itself is
  offered.
- **The MESc table.** One row per recording: a scan's picture (`background_unit`)
  and RTMC reference unit (`rtmc_unit`) fold into its row (`mesc_units.companions`).
  The `picture` cell is **one** button (`IMAGE_ICON`) that opens the reference image;
  no second button, no popup, no Z-stack column. The `RTMC` cell is `yes` or `no`
  (`rtmc_on`: `list_mesc_units`' `rtmc` or `rtmc_armed`), detail on hover. The
  popup's display button reaches the tab through `_show_reference_unit`, which parks
  the request on `_pending` for `_frame` to apply after the popup has drawn: a unit
  switch rebuilds the panel widgets, never inside another widget's draw.
  `_install(arr, z)` sets the viewer's `roi_slider` after the swap, so a Z-stack
  opens on the tissue the ROIs were scanned in. Every header carries its meaning
  (`COLUMN_HELP`); `?` opens `assets/docs/mesc.md`. The header line carries one
  button per `quick_pipelines` entry (`vnoiser on MUnit_3`) that opens the pipeline
  seeded from the unit and sliders on screen (§7.2).
- **The stack.** The Traces panel is masknmf's `TracePlot` over the viewer's frames
  (seconds when the data has a rate; right-click picks frames or time, autofit, fit
  now), with no autofit so a zoom holds while selecting others; `t` keeps the
  playhead centered. Panels top to bottom: the recording's motion correction (`MC`,
  §5.8, named after its `y_label`), then the traces (named after their y label). The
  behavior raster (`Behavior`, §5.9) is a plot of its own over them, x range held on
  the trace panels'; its epochs shade the trace panels (`TracePlot.span`).
  `_sync_plot` resets the plot when frames or panels change; the panel grows by each
  part's height (`MOTION_PANEL_HEIGHT - PANEL_HEIGHT`, `BEHAVIOR_PLOT_HEIGHT`). A new
  facet sampled on the recording's clock is another `TracePlot` panel; one that is
  not (events, epochs) rides over it like the behavior.
- **Playhead.** `gui/playhead.Playhead` is the one time on screen, in seconds on the
  recording's clock (raw frames when `fs` is unknown); it emits `time` with its
  `source`. Every view keeps a `TimeAxis` (`per_second`, `offset`) and converts
  through it: the viewer's T slider (`viewer_axis`, `fs / frame_average`), each trace
  row (`trace_axis`: its own `fs` else the movie's, its `frame_average`, its frame
  window as the offset), the plot's unit (`plot_axis`: frames, seconds or ms).
  `TimeAxis.on(other)` gives the `(xscale, xstart)` a row is plotted with, so a
  windowed trace sits where it was recorded. The trace plot and the motion plot seek
  the playhead; the widget's handler moves the viewer's T, whose indices event snaps
  the playhead to the frame. The ROI widget shares its host's playhead
  (`PreviewDataWidget.playhead`). A new time-bound view subscribes to the playhead;
  it never reads another view's cursor.
- **Slice.** `gui/slice.Slice` is the slice on screen the same way: every slider's
  0-based index by the viewer's dim name (`positions`), which array axis each slider
  scrolls (`roles`, from `viewer_roles`: the squeeze wrapper's kept axes when the
  viewer has one, else T, C, Z by position) and `t` / `c` / `z` by role; it emits
  `slice` with its `source` when anything but T moves. One per host
  (`PreviewDataWidget.slice`), fed by the host's one handler on `ndwidget.indices`,
  which also seeks the playhead; a widget on a bare viewer makes its own. Every view
  that depends on the channel or z-plane shown subscribes to it and never reads
  `iw.indices` itself: the ROI widget's store plane (`_on_slice` →
  `RoiModel.set_view`), its overlays and both tables (the ROI table's "this plane"
  filter turns on when per-plane results load; the trace table lists the slice on
  screen unless "all slices" is ticked), the host's mean image and auto-contrast,
  Signal Quality's active plane (`_stats._active_stat`), the Summary Images and
  Diagnostics. Pinned by `tests/test_slice.py`.
- **Color by.** `RoiModel.column(name)` gives one number per uid (`plane`, `z`, `c`,
  `area`, `class`); `RoiModel.colorize(values, cmap, categorical)` maps them through
  a `cmap` colormap into `RoiLabelStore.tint`, a display-only colour per uid that
  `roi_rgb` prefers while set (never saved; `set_tint(None)` clears). The Curation
  tab's color by drives it (`ManualRoiWidget.set_color_by`; `peak` uses the trace
  table) and reapplies it as ROIs, labels or traces change, so overlay, table and
  legend agree without any view knowing why.
- **Shared vocabularies.** Nothing in the ROI work spells a name or a selection of
  its own: a run's units are `OutputFilename` tags in T, C, Z order (`ch01_zplane02`),
  a single ROI's run tag is the R tag (`roi02` for store index 1, 1-based like every
  tag), a full plane's dir is `results.unit_name("plane", n)`, and a run dir's
  per-slice child is recognised with `filename_tags`, never a regex. The frame
  selection is the string every Save As and pipeline row takes
  (`parse_timepoint_selection`, 1-based `start:stop:step,exclude`), kept as the
  0-based `run_tp` index list the worker tasks call `tp_indices`;
  `PlaneMovie.select(indices)` reads it (`window` is the contiguous case),
  `_slicing.index_window` says whether it is a `(start, stop, step)` window (stamped
  as `frames` on rows and runs) or a gapped list (stamped as `tp_indices`), and
  `_ops_for` scales `fs` for a stride through `OutputMetadata` (§6.6). 1-based plane /
  channel worker args come from `to_lsp_kwargs`.
- **Persistence.** Nothing is written beside the data unasked: data is often opened
  where the user cannot or should not write. Everything the widget keeps is one file,
  the rois zarr `<stem>.rois.zarr` (`annotation/ngff.py::RoiFile`,
  `roi_workflow.labels_path`), written whole on Save, or as anything changes when
  `roi_autosave` is on (off by default), into a `.tmp` sibling swapped into place. It
  is a zarr v3 group: `rois/` is the truth, ragged membership like the results
  zarr's (`offsets`, `member` as flat `y * X + x`, `weight`) plus one column per ROI
  (`uid`, `plane`, `class`, `color`, the drawn `polygon`) and the notes and sources
  in attrs, so ROIs may overlap and keep every pixel; `traces/` holds the rows of
  `SAVED_SOURCES` (quick traces and pixel averages, `raw` and `neuropil` ragged, row
  records in attrs); `labels/drawn` is an OME-NGFF 0.5 label image derived from
  `rois/` (uid per pixel, smallest ROI on top; `multiscales` axes from `plane_axes`
  plus y, x, scaled by `dz`, `dy`, `dx`; `image-label` colors and properties per
  uid; `source.image` the recording when it is a zarr) for napari, and nothing here
  reads it back; the root attrs hold `mbo_rois`, `shape`, `plane_axes`,
  `label_names`, `next_uid`, the `source` block and `runs` (the loaded runs with
  their curation: `discarded`, `classes`, `colors`, `iscell` overrides).
  `annotation.write_label_image` writes any label image that way
  (`arrays/suite2p._add_suite2p_labels` uses it for a converted zarr's
  `labels/suite2p`). A restore reads the ROI folder, then beside the data; a file
  saved for data of another shape is left alone and the data saves as
  `<stem>_<nz>x<ny>x<nx>.rois.zarr` (`_save_suffix`). The ROI folder
  (`roi_workflow.roi_folder(fpath, root)`) is beside the data, or
  `<root>/<name>_<hash of the path>/` when the user picks a folder (`roi_folder`);
  both options live in `preferences`, set from the ROIs tab's OVERLAY > on disk row,
  and every manual run goes there too. Tints live for the session. Ctrl+z is a stack
  of store snapshots (`_push_undo` before every add, delete, mark and deselect;
  `undo` restores the store in place, `RoiLabelStore.restore`, one `restore` event).
  A file holding several recordings (a `.mesc`) keeps all of it per recording:
  `unit_key` (`manual_roi.unit_key`, the base array's) names the files
  `<stem>_<tag>.rois.zarr` and `<stem>.<tag>_<run>.<stamp>.manual.zarr`
  (`labels_path(fpath, tag)`, `_run_out_path`, tag `MSession_0_MUnit_3`), and a unit
  switch (`MescTabWidget._install`) parks the outgoing widget's store and runs under
  its unit and rebuilds the widget for the incoming one.

Pinned by `tests/test_roi_model.py`, `tests/test_playhead.py`,
`tests/test_trace_display.py`, `tests/test_imagej.py`, `tests/test_manual_roi.py`
(`TestRunCoordinates`, `TestAutoTrace`, `TestRoiPipelineTab`, `TestPlayheadWiring`,
`TestColorBy`, `TestFullImage`, `TestSliderRoles`), `tests/test_roi_runs.py`.

## 8. Logging and the Process Console

One logger tree, one console sink per process, one log file per background task.
`logging.md` at the repo root is the audit this section is based on.

### 8.1 Loggers

- Every module gets its logger at import from `mbo_utilities.log`: `logger =
  log.get("arrays.zarr")` is `mbo.arrays.zarr`; names follow the module path under
  `mbo`; `log.get()` with no name is the `mbo` root. Never `logging.getLogger(...)`,
  never `print()` in library code; `print` and `click.echo` are for CLI command
  output only.
- `log.py` owns the handlers. The root `mbo` logger carries the one console
  `StreamHandler`; `log.attach(handler)` adds a process-wide sink (the GUI panel, the
  worker file); `log.set_global_level(level)` sets every `mbo*` logger;
  `log.get_package_loggers()` lists them. Library code never adds a handler.
- Level: `MBO_DEBUG` selects DEBUG, else INFO, at import. `log.set_debug(enabled)` is
  the one way to change it afterwards: it writes `os.environ["MBO_DEBUG"]` (so
  spawned workers inherit it) and calls `set_global_level`; `log.debug_enabled()`
  reads it back. `mbo --debug` / `mbo view --debug` set it for a run; the GUI "Debug
  logging" toggle (`_options_popup`, `file_dialog`) also persists the preference,
  which `run_gui` applies at launch unless `MBO_DEBUG` is already set.
- The GUI's Debug panel (`gui_logger.GuiLogger`) receives every `mbo.*` record
  through a `GuiLogHandler` attached in `preview_data._init_logging`; it filters by
  level and logger, and its master level dropdown calls `set_global_level`. The
  Widgets menu (`gui/widgets/widget_toggles.py`, one checkbox per `WidgetEntry`, no
  submenus; a widget's `toggle_key` is an entry's key) also opens the floating tool
  windows with no debug gate: Style Editor (§14) and ImGui Debugger
  (`gui/widgets/imgui_debug.py`: `imgui_debugger`'s variable inspector plus Dear
  ImGui's metrics, debug log, ID stack, demo and about windows, drawn from
  `PreviewDataWidget.draw` every frame).

| Level | Use |
|-------|-----|
| DEBUG | per-chunk or per-frame detail, dispatch decisions, cache hits |
| INFO | one line at the start and end of a stage: what, shape, path, elapsed |
| WARNING | a recovered anomaly: fallback taken, stale alias, skipped file. Deduplicate repeats (`metadata/params._STALE_WARNED` pattern) |
| ERROR | `logger.exception(...)` once, immediately before re-raising or `monitor.fail` |

No logging above DEBUG inside per-frame loops. tqdm bars are for terminals; the
worker sets `TQDM_DISABLE=1` and collapses `\r` redraws to one line per bar.

Target wiring (the fix for `log.py:22`, tracked in `CONFORMANCE.md`): `log.get`
leaves `propagate=True`; the root `mbo` logger is the only logger with handlers;
`log.attach` adds to the root; the worker's `setup_logging` swaps the root console
handler for its rotating file handler, because its stdout and stderr already point
at the same file and a second stream sink duplicates every line. Until that lands,
an `mbo.<sub>` message is visible only inside the GUI (which attaches handlers to
every child) or through the `logger` a task receives; in scripts and workers, child
INFO goes nowhere and WARNING+ leaks bare through Python's `lastResort` handler.

### 8.2 Progress is not logging

- A long operation takes `progress_callback(fraction, message)` and calls it per
  chunk. It does not log per chunk.
- A worker task forwards progress to `TaskMonitor(output_dir, uuid)`, which writes
  `~/.mbo/logs/progress_{uuid}.json` atomically: `{pid, uuid, timestamp, status:
  running|completed|error, progress, message, details}`. A task ends with exactly one
  of `monitor.finish()` or `monitor.fail(error, details)`; `_worker.main` writes the
  same sidecar as a backstop (`completed` on return, `error` with the traceback on an
  exception, after logging memory at failure).
- A single blocking call that cannot report progress runs a heartbeat thread
  (`task_masknmf._heartbeat`, `_vnoiser_heartbeat`) so the watchdog sees activity.
  The watchdog terminates a worker after 120 minutes with no progress change and no
  log-file mtime change.
- Work on a GUI thread (ROI traces) has no pid or log file; it registers a
  `LocalJob` (`pm.start_job`, `set_progress`, `done` / `fail`) so it still shows in
  the console.

### 8.3 How a pipeline run reaches the Process Console

1. A widget calls `ProcessManager.spawn(task_type, args, description, output_path)`.
   `spawn` mints `task_uuid`, sets `args["_uuid"]` and `args["_log_file"]`
   (`~/.mbo/logs/{YYYYmmdd_HHMMSS}_{task_type}_{uuid[:8]}.log`), writes
   `args_{uuid}.json` (the JSON can exceed the Windows command-line limit), and
   launches `python -m mbo_utilities.gui._worker <task_type> <args_file>` detached
   (`pythonw` + `CREATE_NO_WINDOW` on Windows, `start_new_session` elsewhere) with
   **stdout and stderr redirected to the log file**, `PYTHONUTF8=1` and the GUI's
   `MBO_DEBUG`. It records a `ProcessInfo` (pid, description, task_type, log file,
   args) in `~/.mbo/cache/running_processes.json`, so the process is still tracked
   after a GUI restart (entries older than 24 h are dropped).
2. `_worker.main` reconfigures stdio (utf-8, line-buffered, `_CRCollapseWriter`),
   loads and deletes the args file, and calls `setup_logging(log_file)`: the
   `mbo.worker` logger gets a `RotatingFileHandler` (5 MB, two backups) and every
   handler the format `%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s`.
   It contains children in a Windows Job Object, starts the watchdog and the memory
   monitor (`mem_{uuid}.csv`), resolves `TASKS[task_type]` (loading entry-point
   tasks when missing), and runs `task_func(args, logger)`.
3. The log file therefore receives `mbo.worker` records, the root `mbo` console
   handler (stderr), everything the pipeline or its dependencies print, and bridged
   third-party loggers: a `_ForwardingHandler` mirrors records onto the worker
   logger's handlers and sets `propagate=False` (`_bridge_isoview_logging`); never
   attach the root console handler to a third-party logger, that writes every line
   twice.
4. The console (`gui/_popups.py`, from the menu-bar status button) calls
   `pm.cleanup_finished()` and `pm.get_running()` every frame; each `ProcessInfo`
   re-reads its sidecar (`progress_{uuid}.json`, falling back to
   `progress_{pid}.json`, identity-checked). A dead process with no terminal status
   becomes `error: Process crashed unexpectedly`. Completed rows stay five minutes;
   errors stay until dismissed. Each row shows status, description, `PID · elapsed`,
   Kill (terminates the process tree) or Dismiss, Copy (the whole log file), the
   error message, and a Log Output tailing the last 500 lines. The same registry
   backs `mbo processes [--kill PID | --kill-all | --cleanup]`.
5. `prune_logs()` runs when the manager starts: sidecars, args files and `.tmp` older
   than 24 h are deleted; the newest 25 logs are kept (`MBO_LOG_KEEP_RECENT`) and the
   rest deleted after 14 days (`MBO_LOG_RETENTION_DAYS`).

### 8.4 HPC

SLURM jobs never see the GUI. `mbo hpc run` (submitit) streams job stdout and stderr
to `<output_dir>/logs/*.out` and `*.err` (never `$TMPDIR`, even with node-local
staging); `write_timing_report` writes `<output_dir>/timings.json`;
`write_failure_report` writes `FAILURE_{role}_{task}_{pid}.log` beside the run and a
rotated copy under `~/.mbo/logs` (last 20). `record_run` keeps the last 10 launches
in `~/.mbo/hpc/runs` so `mbo hpc watch` and `mbo hpc status` with no argument find
the latest run; `watch` follows `.err` by default (`-o` for `.out`; `o` `e` `n` `p`
`q` while following). Compute-node narration belongs in `log.get("hpc.pipeline")`
with a timestamp and the job and task id, not `print`.

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
   `@pytest.mark.slow` and skip without data (`tests/test_vnoiser_pipeline.py`).
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
  `RENDERCANVAS_FORCE_OFFSCREEN=1` is set by `conftest.py`; no test opens a window.
- Contract tests to keep green when touching §5, §6 or §7:
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

MyST + Sphinx book theme; published on merge to `main` by
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
  holds the one `imgui_debugger.StyleEditor` for the process (Widgets > Style
  Editor), backed by an `imgui_debugger.ConfigStore` at `get_mbo_dirs()["imgui"]`:
  `state.json` (the style as last left) and `styles/<name>.json` (presets),
  autosaved a second after the last slider moves. `apply_saved_style()` runs in
  `PreviewDataWidget.__init__` right after `style_imgui_opaque()`, so a saved style
  wins over the shipped theme. Window geometry stays imgui's, in
  `imgui/assets/app_settings/preview_settings.ini`. Nothing else writes the style;
  `imgui_debug.py` deliberately has no style entry. `imgui_debugger` and
  `imgui_notebooks` are base dependencies pinned to GitHub commits in
  `pyproject.toml` (PyPI lags the API used here); bump the pins when they move.
- File > Open Notebook (`gui/widgets/notebooks.py`) is `imgui_notebooks`. Its sources
  are the `scripts` folder beside the open recording and the shipped templates that
  fit it (`notebook_templates.py`, the table `mbo init` writes from; `demos/` is the
  source of truth, copied into the package at build time by `setup.py`). A template
  is written beside the data with the path and `.mesc` unit filled in when first
  picked. Servers run in the Studio's own interpreter; added folders and the recent
  list are in `get_mbo_dirs()["imgui"]/notebooks.json`.
- Environment: `MBO_GPU` (GPU toggle; also `mbo gpu`), `RENDERCANVAS_FORCE_OFFSCREEN`,
  `KEEP_TEST_OUTPUT`, `MBO_PIPELINE_TIFF`; logging and retention variables in §8.5.
- HPC: `hpc.toml` written by `mbo hpc init`; the shared cluster environment is
  described in `docs/usage/hpc.md`. Nothing in this repo contains cluster
  credentials or any other secret; do not add any.

## 15. Imgui layout

The spacing, sizing and alignment reference for the Studio's imgui code is the
`imgui-layout` skill (`~/.claude/skills/imgui-layout/SKILL.md`). Read it before
placing or aligning widgets.
