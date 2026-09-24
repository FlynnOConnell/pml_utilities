(gui_guide)=

# Miller Brain Studio

Interactive data preview and processing tools for calcium imaging data.

```{figure} /_images/gui/readme/02_step_data_view.png
:width: 100%
:alt: Miller Brain Studio

The time-series viewer: image canvas with axes and histogram, plus the side panel (Preview / Signal Quality / Run tabs) for window functions, spatial filtering, scan-phase correction, z-stats, and pipelines.
```

## Quick Start

```bash
uv pip install mbo_utilities
mbo                    # opens file dialog
mbo /path/to/data      # opens specific file
mbo /path --metadata   # metadata only
```

From Python, the one-call form builds and shows the viewer for wherever it is running:

```python
import mbo_utilities as mbo

mbo.run_gui("/path/to/data")

# or from a numpy array
import numpy as np
data = np.random.rand(100, 512, 512)
mbo.run_gui(data)
```

In a terminal or script this opens a window and blocks until it closes. With no input it opens the launcher, where you pick a file or folder, or type a path.

### Notebooks

The same viewer is a class, built the way masknmf's viewers are: construct it with the data, `show()` it, `close()` it. Construct and show in the same cell; the canvas is the cell's output and the kernel drives it, so do not call `fpl.loop.run()`.

```python
from mbo_utilities import DataVis

vis = DataVis("/path/to/data", roi=1)   # widget="manualroi" adds the ROI tools
vis.show()
```

```python
vis.iw.cmap = "viridis"   # the NDViewer underneath: sliders, cmap, window functions
vis.widget                # the side panel
vis.close()
```

`run_gui` does the same and returns the `DataVis`, already displayed. It needs a path in a notebook: the launcher is a desktop window. A `.mesc` needs `unit=` when it holds more than one measurement unit, since the picker is a desktop dialog too. Pass `size=(w, h)` to change the canvas size; the notebook default of 1400x900 leaves room for the side panel and top strip.

The notebook canvas streams frames over the kernel, so this is also how the viewer runs on a remote server or JupyterHub. Native file dialogs open on the machine the kernel runs on, so every place the GUI asks for a path (File > Open, `o`, `Shift+O`, the launcher, Save As) takes a typed path first and offers the native dialog as a browse shortcut, disabled when the machine cannot draw one.

(gui-open-data)=
## Opening Data

```{image} /_images/gui/readme/01_step_file_dialog.png
:width: 80%
:alt: File Selection Dialog
```

### Open File vs Select Folder

- **Open File(s)** (`o`): select one or more tiff files. loads exactly the file(s) you pick — nothing else in the directory is touched.
- **Select Folder** (`Shift+O`): scans the directory and loads all compatible files. the reader auto-detects the format from the first file and filters out incompatible files (e.g. previously saved outputs or unrelated TIFFs are excluded).

### Supported Formats

| Format | Description |
|--------|-------------|
| `.tiff` | raw ScanImage, BigTIFF, OME-TIFF, ImageJ hyperstacks |
| `.zarr` | zarr v3 arrays |
| `.bin` | suite2p binary format |
| `.h5` | HDF5 files |
| `.npy` | numpy arrays (memory-mapped) |

### Load Options

| Option | Description |
|--------|-------------|
| Separate ScanImage mROIs | split multi-ROI acquisitions into separate panels |
| Enable Threading | parallel z-stats computation on load |
| Enable Data Preview Widget | full preview with window functions and controls |
| Metadata Preview Only | show only metadata, skip image rendering |

(gui-viewers)=
## Viewers

The GUI selects a viewer automatically based on the data type.

### Time-Series Viewer

The default viewer for calcium imaging data (TZYX). Used for ScanImage TIFFs, standard TIFFs, ImageJ hyperstacks, and other volumetric data.

Features:
- temporal projections (mean, max, std) over a sliding window
- spatial filtering (gaussian blur, mean subtraction)
- scan-phase correction for bidirectional raster scanning
- frame averaging for piezo z-stacks
- z-stats signal quality analysis
- suite2p pipeline integration

### masknmf viewers

A masknmf demixing result (`mbo run/demixing_results.hdf5`, or any
`*_demixing.hdf5` from a glutamate/calcium spine run) does not open in the
Studio viewer. It opens in one of masknmf's own viewers, chosen before launch
(`--vis`, or a prompt in the terminal when `--vis` is omitted); the viewer
window is masknmf's, with nothing added:

- **Demixing**: `SingleSessionDemixingVis`, the PMD movie, signals, background,
  residual and colourful signals, with masknmf's ROI and signal curation tools.
- **Classification**: `ClassificationVis`, accept / reject and class labels
  one ROI at a time; labels save to `<file>.labels.hdf5` beside the result.

`mbo view run/demixing_results.hdf5 --vis classification` skips the prompt. Torch and masknmf must be installed; the device follows the
compute-GPU policy (`MBO_GPU`, `CUDA_VISIBLE_DEVICES`, File > Options).

### Pollen Calibration Viewer

Specialized viewer for LBM beamlet calibration data (`stack_type == "pollen"`). Automatically selected when pollen calibration data is loaded.

Features:
- automatic bead detection via cross-correlation
- manual interactive calibration (click-to-mark beads)
- cavity A/B discrimination for dual-cavity LBM
- result visualization with XY position and offset plots
- previous calibration result loading from H5 files

(gui-navigation)=
## Navigation

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Navigation** | |
| `←` / `→` | previous / next frame (T) |
| `↑` / `↓` | previous / next z-plane (Z) |
| `Shift + ←/→` | jump 10 frames |
| `Shift + ↑/↓` | jump 10 z-planes |
| **File** | |
| `o` | open file |
| `Shift+O` | open folder |
| `s` | save as |
| **View** | |
| `m` | toggle metadata viewer |
| `p` | toggle side panel |
| `v` | reset contrast (vmin/vmax) |
| `c` | toggle auto-contrast on z-change |
| **Help** | |
| `h` / `F1` | open help |
| `k` | keybinds cheatsheet |

Press `k` in the GUI to see this list at any time.

```{figure} /_images/gui/readme/10_keybinds.png
:width: 70%
:alt: Keybinds cheatsheet

The in-app shortcut cheatsheet, opened with `k`.
```

### Menu Bar

- **File**: Open File, Open Folder, Save As
- **Docs**: Help viewer, Keybinds, Online Docs link
- **Settings**: Scope Inspector toggle, Status Indicator toggle

### Options

**File > Options** sets the render GPU adapter, debug logging, and memory-usage logging.

```{image} /_images/gui/readme/12_options.png
:width: 60%
:alt: Options
```

(gui-preview)=
## Preview Controls

### Window Functions

Apply temporal projections over a sliding window of frames.

| Function | Description |
|----------|-------------|
| mean | average intensity over window |
| max | maximum intensity projection |
| std | standard deviation over window |

**Parameters:**

- **Window Size**: number of frames to include (3-20 recommended)
- **Apply to dataset**: turn the window size into temporal binning of the data itself. Every `N` consecutive frames become one averaged frame, the frame rate is divided by `N`, and the window size resets to 1 so it applies on top of the binned frames.
- **Gaussian Sigma**: spatial gaussian filter (0 = disabled)
- **Mean Subtraction**: subtract per-z-plane mean image to highlight activity. requires z-stats to finish computing first.

Like scan-phase correction, binning applied here is a pipeline step, not a display effect. The ROI traces, Save As, Suite2p and MaskNMF all pick up the factor as their default: it shows up as **Frame Average** in each Options menu, where it can be changed per run. Programmatically the same thing is `imread(path, frame_average=N)` or `imwrite(arr, out, frame_average=N)`.

### Scan-Phase Correction

Preview bidirectional raster-scan phase correction before saving. Only available for ScanImage data.

| Parameter | Description |
|-----------|-------------|
| Fix Phase | enable/disable correction |
| Sub-Pixel | FFT-based sub-pixel correction |
| Upsample | sub-pixel precision factor (1/N pixel) |
| Exclude border-px | exclude edge pixels from correlation |
| max-offset | limit allowed pixel offset |

**Workflow:**

1. view mean or mean-subtracted projection (window 3-15)
2. toggle Fix Phase on/off to compare
3. adjust border-px and max-offset if needed
4. toggle Sub-Pixel for further improvement
5. adjust Upsample factor (2-3 typical)

### Frame Averaging

Readout for the temporal binning set with **Apply to dataset** above: frames per averaged frame, the frame count and frame rate before and after.

Also hosts piezo z-stack averaging. When `frames_per_slice > 1`, toggle averaging based on ScanImage's `logAverageFactor`. This changes the effective shape of the data.

### MESc

Femtonics MESc files only. A `.mesc` holds one measurement unit per scan the operator ran, and `mbo scan.mesc` opens the first AOD scan (line scan, chessboard or ribbon), or the first unit when there is none. The **MESc** tab, the first tab for a `.mesc`, is how the rest are opened: it lists every unit as a table, with session and unit (MESc numbers each session's units from scratch, so `MSession_0/MUnit_3` and `MSession_1/MUnit_3` are different recordings), modality, shape, rate, duration, start time and comment. **ROIs** is the lines or patches a scan recorded (`15 lines`, `7 patches`). One row per recording: the picture MEScan took just before a scan (its `BackgroundImagePath`, role `background`) and the small reference region RTMC re-scanned every cycle (`MotionCorrectionImagePath`, role `motionCorrection`) are folded into the scan's row as its **picture** and **RTMC** cells (**companion units as rows** unfolds them). **picture** is one button naming the picture MEScan took just before the scan; click it to see the scan's lines or patches drawn on it. **RTMC** is `yes` or `no`: whether real-time motion correction was on for the recording; hover it for whether the scan was moved (the Traces tab's MC plot then draws the shifts). There is no Z-stack column: which stack was taken around a scan, and which slice its lines sit on, is in the reference image, next to the lines themselves. Hover a column header for what it means, a unit name for what that recording is, and click **?** for the MESc help page. Click a row, or an entry in a cell's popup, to display it. The Manual ROI widget follows the unit: the ROIs, runs and traces on screen are always the shown unit's, parked when you switch away and back when you return, and autosaved per unit (`manual_labels_MSession_0_MUnit_3.zarr`, `roi_runs_MSession_0_MUnit_3.json`, `rois_MSession_0_MUnit_3_<tag>/`).

**Reference image** (the **picture** button on the MESc tab, and a button on the Traces panel) opens a popup showing where the shown scan's lines or patches were drawn: the picture MEScan took just before the scan, and every Z-stack holding them, projected over the slices they were scanned on rather than over the whole stack. Each carries the lines or patches drawn in MESc's colours, the one the ROI slider is on thicker. The picture carries every line whatever its depth, since that is where the operator drew them; a Z-stack carries only the lines actually scanned inside it, and the caption says how many of the scan's lines those are. Click a line to move the ROI slider to it. The popup's **display** button shows the image's own unit in the viewer, a Z-stack at the slice its lines sit on, so the Z-plane slider lands on the right tissue instead of slice 1. Pan, zoom, colormap, contrast and pixel values work as in **Open full FOV**; the popup follows a unit switch while it is open. The pairing is by the file's own metadata (`CoordinateMapJSON` outlines, `ReferenceViewportJSON` placement, `BackgroundImagePath`), so a unit MESc recorded no ROIs for has no reference image. A scan whose ROI list was edited after acquisition falls back to the viewer's own colours.

(gui-metadata)=
## Metadata Viewer

Toggle with `m` or the **Metadata** button in the status bar.

```{image} /_images/gui/readme/03_metadata_viewer.png
:width: 80%
:alt: Metadata Viewer
```

Displays all metadata attached to the current array, including ScanImage headers, dimension tags, and user-supplied fields.

(gui-zstats)=
## Z-Stats

Per-z-plane signal quality statistics, computed in the background on load.

### Metrics

| Metric | Description |
|--------|-------------|
| Mean | average fluorescence intensity |
| Std | standard deviation |
| SNR | signal-to-noise ratio (mean / std) |

### Visualization

The z-stats panel adapts to the data:

- **single z-plane**: stats table with bar chart
- **2 z-planes**: grouped bar charts (Z1 vs Z2)
- **many z-planes**: line plots with error bars and z-plane signal profiles
- **multiple ROIs**: combined per-ROI profiles with mean +/- std shading

(gui-save)=
## Saving Data

Open via **File > Save As** or press `s`.

```{figure} /_images/gui/readme/04_save_as_dialog.png
:width: 80%
:alt: Save As Dialog

The filename, estimated size, and output shape update live as you change the format and the timepoint/plane selection.
```

### Important: Save As Does Not Change the Active Dataset

Save As exports a copy of the data to a new file. The viewer continues to display the **original** dataset. Any subsequent operations (Suite2p, further saves) still use the original data.

To work with the saved file, open it explicitly via **File > Open File**.

### Output Formats

| Format | Description |
|--------|-------------|
| `.tiff` | BigTIFF with ImageJ/OME metadata |
| `.zarr` | zarr v3 (recommended for large data) |
| `.bin` | suite2p binary format |
| `.h5` | HDF5 |

### Selection

The save dialog provides dimension-specific subsetting:

- **Timepoints**: `start:stop:step` syntax with optional exclusion ranges
- **Z-planes**: range and step selection
- **Channels**: multi-channel selection (when applicable)
- **Output suffix**: custom suffix appended to filename

An output preview shows the filename, estimated size, and output shape before saving.

### Options

```{figure} /_images/gui/readme/05_save_options.png
:width: 80%
:alt: Save Options

General write options (`.tiff` shown). Format-specific sections appear below as you change the output extension.
```

| Option | Description |
|--------|-------------|
| Run in Background | save without blocking the GUI |
| Overwrite | replace existing output files |
| Fix Scan Phase | apply phase correction on write |
| Subpixel Correction | FFT-based phase correction on write |
| Frame Average | average every N frames into one on write; defaults to the viewer's "Apply to dataset" factor |
| Register Z-Planes | axial (plane-to-plane) phase-correlation registration |
| Chunk Size (MB) | memory chunk size for writing |

### Zarr-Specific Options

```{image} /_images/gui/readme/08_save_options_zarr.png
:width: 80%
:alt: Zarr Save Options
```

| Option | Description |
|--------|-------------|
| Sharding | enable zarr sharding for faster access |
| OME-Zarr | write OME-Zarr compliant metadata |
| Compression Level | zstd compression level (0 = none) |
| Pyramid | generate multi-resolution pyramid |
| Pyramid Layers | max number of downsampled levels |

### Video (.mp4) Options

```{image} /_images/gui/readme/09_save_options_mp4.png
:width: 80%
:alt: Video Export Options
```

Exporting to `.mp4` exposes playback and rendering controls: frame rate, speed factor, contrast percentiles, temporal/spatial smoothing, gamma, colormap, quality, and codec.

### Metadata

```{image} /_images/gui/readme/04_configurable_metadata.png
:width: 80%
:alt: Configurable Metadata
```

The save dialog includes a metadata editor:

- suggested fields are auto-populated from the array
- fields can be auto-detected from the filename
- custom key/value pairs can be added
- missing recommended fields are highlighted

(gui-process-manager)=
## Manual ROIs

**Widgets > Manual ROI Labeling** (or `mbo <path> --widget manualroi`) adds the
**ROIs** and **Traces** tabs to the right bar and the **Traces** panel to the strip
over the image: the ROIs tab holds the NAVIGATE, DRAW, VIEW and LABELS sections
(captioned settings rows like the Process tab's, each switched off from the
Widgets menu) over the ROI table, the Traces tab lists every trace,
and the panel plots the selected ones under its controls. Nothing switches tabs or
panels for you. Arm **Add ROI** (`a`), drag a closed stroke around a cell, release:
the enclosed pixels become a mask on the exact slice on screen (z-plane, channel,
any extra slider) and, with **trace on draw** ticked, its mean trace appears on the
Traces panel at once. Masks autosave beside the data as `manual_labels.zarr`; a
`.mesc` gets one per unit (`manual_labels_MSession_0_MUnit_3.zarr`), and the ROIs and
traces on screen are always the shown unit's.

Running ROIs is the **Process tab > ROIs** pipeline:

- **Which ROIs**: the selected one (or the ctrl / shift click group), the rows the
  ROIs tab lists (its filters apply), every ROI drawn on the slice on screen, all of
  them, or the **full image**: the whole frame as one mask. With `mean` that is the
  frame's mean trace; with `suite2p` or `masknmf` it is a full detection of that
  z-plane and channel.
- **Read from**: **as drawn** reads each mask on the z-plane and channel it was drawn
  on; **slice on screen** reads it wherever the sliders are when you press Run, so
  a cell drawn on the structural channel is traced on the functional one by
  scrolling there and running again; **fixed** picks a z-plane and channel. A
  **frames** window (`start:stop`, 1-based) cuts T.
- **Engine**: `mean` (raw mask mean plus a neuropil ring, no pipeline), `suite2p`
  (suite2p's extractor), `masknmf` (seeded demixing). `suite2p` and `masknmf` use
  the settings of their own tab, including the skip / run / force toggles; **Open**
  jumps there. The **tag** names the output folder `rois_<tag>/` beside the data.
- **Run** writes suite2p-shaped outputs (`F.npy`, `Fneu.npy`, `stat.npy`,
  `rois.json`, `ops.npy` with `roi_workflow` recording `z`, `c`, `frames` and
  `engine`); **Trace** computes an in-memory mean without writing.
- **Find cells in a region**: draw a region (`r`) and let suite2p or masknmf look
  for cells inside it. Results arrive as algo overlays: promote (`y`) or discard
  (`n`) each component.

The sliders are the array's axes whatever they are called: a MESc AOD unit's
**ROI** slider or an IsoView **View** slider keys masks and runs like any z-plane
or channel, and the trace table and legends name the axis the same way (`ROI 3`,
not `z3`). An AOD unit's ROIs, a line scan's lines or a chessboard or ribbon scan's
patches, land on the Traces tab as their mean traces (computed in the background
when no `F.npy` or PF traces exist; **Options** can turn that off) with their index
and channel. Every row read on an AOD unit knows its line, whether a quick trace of
a drawn ROI, a line-scan trace or a pipeline's denoised row: hovering the row lists
the line's ends, length and sample spacing, from the scan's own geometry.
**Reference image** on the Traces panel opens the picture the lines or patches were
drawn on, with them drawn and the slider's one thick; click a line there to select
its ROI.

**VIEW > color by** tints every ROI by a value through a colormap: its class,
z-plane or channel (one color per level), its area, or the peak of its traces (a
gradient). The overlay, the ROI table and the trace legend all follow; **none**
restores the class / group colors. The time cursor on the trace and motion plots is
one playhead: drag either, scrub the T slider, and every plot and the image land on
the same instant, each trace drawn where it was recorded (its own frame window and
binning).

Every measurement is one row of the **Traces** tab: which ROI, on which z-plane and
channel, with which engine (the **pipeline** column: `mean`, `suite2p`, `masknmf`, or
the pipeline that wrote a results file), from which run. A results file's rows are
named by their ROI (`roi3` for its trace, `roi3 (raw)` for its line; a line of a
multi-line ROI adds itself, `roi3 line 12 (raw)`) and put the line on the ROI axis
column. The plot shows the checked rows the way their pipeline says: the **kind**
combo picks `raw`, the pipeline's own `dF/F` (or one computed here over a rolling
max-min baseline sized in seconds; the **dF/F** button sets its window and smoothing
or switches to a percentile baseline), `denoised` or `z-score`, and a row without
that kind shows its pipeline's default (the voltage pipeline's curated trace, dF/F
elsewhere); the y axis is labelled from what is on it. **neuropil corrected** appears
only when a plotted row's pipeline measured a neuropil (suite2p, or the mean engine's
ring). The x axis opens in seconds whenever the data has a sampling rate. Running
the same ROI the same way again
replaces its row; reading it on another channel, z-plane or with another engine adds
one. Click a row to plot it, ctrl+click to plot several; the plot follows the ROI
the image shows. The ROIs tab's row buttons and the `t` key run one ROI exactly the
way the ROIs pipeline is set.

## Process Manager

Click the status indicator in the menu bar to open the process console.

```{image} /_images/gui/readme/11_process_console.png
:width: 70%
:alt: Process Console
```

The status indicator is color-coded:
- **green**: idle or completed
- **orange**: task running (with progress percentage)
- **red**: error

The process console shows:
- **active tasks**: in-app progress (save, z-stats, registration)
- **background processes**: external processes with PID, elapsed time, and status
- per-process log output (color-coded, collapsible)
- kill / dismiss / copy controls

(gui-suite2p)=
## Suite2p Integration

```{figure} /_images/gui/readme/06_suite2p_settings.png
:width: 80%
:alt: Suite2p Processing Settings

The Run tab: dataset summary, output folder, plane/timepoint slicing, scan-phase options, and the pipeline selector. **Parameters and settings > Open** reaches the full parameter dialog.
```

Available when `suite2p` is installed. Access via the processing pipeline panel.

Suite2p always runs on the dataset currently loaded in the viewer. If you used Save As to export a processed file and want to run Suite2p on that file, you must open it first with **File > Open File**.

- run suite2p on selected z-planes
- all suite2p parameters exposed with descriptions
- output directory selection
- scan-phase correction options for processing

### Parameters

**Parameters and settings > Open** opens every Suite2p and LBM-Suite2p-Python parameter in one dialog, grouped by Registration, ROI Detection, Signal Extraction, Deconvolution, and Classification.

```{figure} /_images/gui/readme/07_suite2p_parameters.png
:width: 100%
:alt: Suite2p Parameters

Every Suite2p and LBM-Suite2p-Python parameter in one dialog. Values changed
from the Suite2p default are tinted orange (here `tau` and the cell diameters),
so non-default settings stand out at a glance.
```

The parameter columns are colour-coded. The **Legend** button (next to **Defaults**) explains the conventions:

```{figure} /_images/gui/readme/13_suite2p_legend.png
:width: 55%
:alt: Suite2p parameter legend

The colour/box legend, opened from the settings dialog.
```

| Style | Meaning |
|-------|---------|
| yellow name | Suite2p parameter |
| teal name | LBM-Suite2p-Python parameter |
| orange value | modified from the Suite2p default |
| boxed label | important parameter (look at these first) |

The Run tab also lists every changed value under **Modified parameters**, with a copy button that emits a paste-ready Python dict.

### Spatial Crop

1. click "Add Crop Selector"
2. drag the yellow rectangle on the image
3. only the cropped region is processed

### External Tools

The GUI can launch external tools when installed:
- **suite2p GUI** with rastermap integration
- **cellpose GUI** for cell segmentation
- window polling detects when external tools close

### Results and Diagnostics

- suite2p results viewer with trace quality stats
- diagnostics viewer for signal quality analysis
- grid search viewer for parameter exploration
