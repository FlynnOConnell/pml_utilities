(cli_usage)=

# Command Line Interface

The `mbo` command provides tools for viewing, converting, and analyzing imaging data.

| Command | Description |
|---------|-------------|
| `mbo` | Launch GUI with file dialog |
| `mbo convert` | Convert between formats |
| `mbo info` | Show array info |
| `mbo init` | Create starter notebooks |
| `mbo formats` | List supported formats |
| `mbo shortcut` | Create a desktop icon |
| `mbo gpu` | Show render/compute GPU and memory |

## GUI Mode

```bash
mbo                          # file dialog
mbo /path/to/data            # open specific file
mbo /path/to/data --metadata # show only metadata
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
```{image} /_images/gui/readme/01_step_file_dialog.png
:alt: File selection dialog
:width: 100%
```
:::

:::{grid-item}
```{image} /_images/gui/readme/02_step_data_view.png
:alt: Data viewer
:width: 100%
```
:::
::::

## Convert

Convert between formats, optionally selecting planes/timepoints and applying phase correction.

```bash
mbo convert /data/raw output/ -e .zarr               # tiff to zarr
mbo convert /data/raw output/ -e .bin                # tiff to suite2p binary
mbo convert /data/volume.zarr output/ -e .tiff       # zarr to tiff
mbo convert /data/raw output/ -p 1 -p 7              # specific planes (repeat -p)
mbo convert /data/raw output/ -t 1 -t 50             # specific timepoints (repeat -t)
mbo convert /data/raw output/ -c 1 -c 2              # specific channels (repeat -c)
mbo convert /data/raw output/ --num-timepoints 500   # first 500 timepoints
mbo convert /data/raw output/ --roi 0                # split ROIs into roiNN/ subdirs
mbo convert /data/raw output/ -e .zarr --compressor zstd --pyramid   # compressed pyramid
mbo convert /data/raw output/ --fix-phase            # with phase correction
```

| Option | Description |
|--------|-------------|
| `-e, --ext` | Output format: `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy` (leading dot required) |
| `-p, --planes` | Z-plane to export (1-based); repeat for multiple: `-p 4 -p 5` |
| `-t, --timepoints` | Timepoint to export (1-based); repeat for multiple: `-t 1 -t 50` |
| `-c, --channels` | Color channel to export (1-based); repeat for multiple: `-c 1 -c 2` |
| `--num-timepoints` | Export first N timepoints |
| `--num-zplanes` | Export first N z-planes |
| `--roi` | ROI: `None`=stitch, `0`=split, `N`=specific, `"1,3"`=multiple. Split/multiple write each ROI to a `roiNN/` subdir |
| `--fix-phase` | Bidirectional phase correction |
| `--overwrite` | Replace existing files |

<details>
<summary><b>All Convert Options</b></summary>

| Option | Description |
|--------|-------------|
| `--output-name` | Output filename (binary format) |
| `--output-suffix` | Suffix appended to output filenames |
| `--order` | Reorder planes before writing (0-based indices into `--planes`) |
| `--roi-mode` | `concat_y`=stitch, `separate`=one file per ROI (implied by `--roi`) |
| `--dataset-name` | HDF5 dataset name (`.h5` only, default `mov`) |
| `--register-z` | Z-plane registration (Suite3D) |
| `--reg-max-frames` | register-z: subsample frame count (default 200) |
| `--reg-chunk-frames` | register-z: streaming batch size (default 10) |
| `--reg-max-xy` | register-z: search radius in pixels (default 30) |
| `--phasecorr-method` | `mean`, `median`, or `max` |
| `--border` | Phase correction: border pixels excluded from estimation |
| `--max-offset` | Phase correction: max pixel offset to search |
| `--use-fft/--no-use-fft` | Phase correction: FFT-based 2D correction |
| `--ome/--no-ome` | OME-zarr metadata (zarr only, default on) |
| `--compressor` | Zarr codec: `none`, `gzip`, `zstd`, `blosc-lz4`, `blosc-zstd` (zarr only) |
| `--compression-level` | Zarr compression level (zarr only) |
| `--sharded/--no-sharded` | Zarr v3 sharding (zarr only) |
| `--pyramid/--no-pyramid` | Write multiscale pyramid (zarr only) |
| `--pyramid-max-layers` | Max pyramid resolution levels (zarr only, default 4) |
| `--pyramid-method` | Pyramid downsampling: `mean`, `median`, `mode`, `gaussian`, `nearest`, `local_mean` |
| `--chunk-mb` | Streaming chunk size in MB (default: 100) |
| `-n, --num-frames` | Deprecated alias for `--num-timepoints` |
| `--debug` | Verbose logging |

</details>

Output is named from the timepoint, channel, and plane ranges.

```
$ mbo convert E:/demo/mk355/raw E:/demo/mk355/convert --num-timepoints 500 -p 4
Reading: E:/demo/mk355/raw
  Shape: (1574, 1, 14, 550, 448), dtype: int16
Writing: E:/demo/mk355/convert (format: .tiff)
Writing TIFF: 100%|███████████████████████████████| 500/500 [00:01<00:00, 394.76pg/s]

Done! Output saved to: E:/demo/mk355/convert/tp00001-00500_ch01_zplane04.tif
```

**Note:**
- `-e/--ext` needs the leading dot: `.zarr`, not `zarr`.
- `-p/--planes`, `-t/--timepoints`, and `-c/--channels` are repeatable; pass each value separately (`-p 4 -p 5`), not `4 5` or `4,5,6`.
- `-t/--timepoints` selects specific timepoints; `--num-timepoints` limits to the first N. (`-n/--num-frames` is a deprecated alias for `--num-timepoints`.)
- `--roi 0` (split) and `--roi "1,3"` (multiple) write each ROI to its own `roiNN/` subdirectory. `--roi N` selects one ROI; omitting `--roi` stitches all ROIs into one FOV.
- Zarr-only options (`--compressor`, `--compression-level`, `--sharded`, `--pyramid*`) are ignored for other formats. `--dataset-name` applies to `.h5` only.

## Info

Display shape, dtype, imaging metadata, and any Suite2p results found alongside the data. Nothing is loaded into memory.

```bash
mbo info /data/raw.tiff
mbo info /data/volume.zarr
mbo info /data/suite2p/plane0
mbo info /data/raw --all       # also dump the full raw metadata dict
```

```
$ mbo info E:/demo/mk301/raw
Loading: E:/demo/mk301/raw

E:/demo/mk301/raw
  Type            LBMArray
  Shape           (500, 1, 14, 448, 448)  [T, C, Z, Y, X]
  Dtype           int16

Imaging
  Frame rate      17.07 Hz
  Pixel size      2 x 2 um
  Frame size      448 x 448 px (Y x X)
  FOV             896 x 896 um

Acquisition
  Stack type      lbm
  Timepoints      500
  Z-planes        14
  Color channels  1
  mROIs           2
  Duration        29.3 s
  Value range     [-324, 4511]

Files (2)
  - mk301_03_01_2025_2roi_..._00000.tif
  - mk301_03_01_2025_2roi_..._00001.tif

Results
  none found
```

| Option | Description |
|--------|-------------|
| `--all` | Also dump the full raw metadata dict |
| `--no-metadata` | Skip imaging/acquisition sections |

## Init

Create starter notebooks (mbo + LBM-Suite2p user guides).

```bash
mbo init                       # notebooks in current directory
mbo init /path/to/raw          # notebooks in /path/to/scripts, data path filled in
mbo init /path/to/raw -o ./nb  # custom destination directory
```

| Option | Description |
|--------|-------------|
| `-o, --output` | Destination directory (overrides default location) |
| `--overwrite` | Overwrite existing notebooks |

With a `DATA_PATH` argument, notebooks are written to a `scripts/` directory beside the data and the data path is pre-filled. Without it, notebooks go in the current directory with default paths.

```{image} /_images/cli/jupyter_lab.png
:width: 80%
:align: center
```

## Shortcut

Create a desktop icon that opens the GUI.

```bash
mbo shortcut                  # "Miller Brain Studio"
mbo shortcut --name "MBO"     # custom name
```

```
Created: C:/Users/RBO/Desktop/MBO.lnk
```

Windows creates a `.lnk` (no console window); Linux creates a `.desktop` entry.

## GPU

Show which GPU renders the viewer and which runs compute (suite2p / cellpose / cupy), plus device memory.

```bash
mbo gpu               # render GPU, compute GPU, device memory
mbo gpu --processes   # also per-process VRAM
mbo gpu --watch 2     # refresh every 2s
mbo gpu --json        # machine-readable
```

```
Render GPU  (fastplotlib): NVIDIA RTX A4000 (DiscreteGPU) via Vulkan (wgpu default)
Compute GPU (suite2p/cellpose/cupy): NVIDIA RTX A4000  (cuda:0)

Device memory:
  GPU 0: NVIDIA RTX A4000 - 941/16376 MB used (6%), 15229 MB free, util 3%, 35C
```

## Utilities

```bash
mbo --check-install      # verify installation and GPU config
```

```
mbo_utilities v3.2.0 | Python 3.12.9
==================================================

CUDA Environment:
  Driver CUDA:         12.6
  GPU:                 NVIDIA RTX A4000

Features:
  [OK] PyTorch
  [OK] CuPy
  [ -] Suite2p (not installed)
  [ -] Suite3D (not installed)
  [ -] Rastermap (not installed)

Installation OK
```

## Linescan

Per-ROI traces from the AOD line-scan units of a Femtonics `.mesc` file. Each
line drawn on the reference Z-stack is its own ROI; its kymograph is averaged
over the line (cropped to the line's true, unpadded extent) into one trace per
timepoint. Ribbon, chessboard, Z-stack and other units in the file are skipped.

```bash
mbo linescan scan.mesc                                  # every linescan unit
mbo linescan scan.mesc -o results/linescan              # keep the raw folder untouched
mbo linescan scan.mesc --unit MUnit_3 --channel 1       # one unit, second channel
mbo linescan scan.mesc --dfof-window 2 --no-dfof        # baseline window in seconds
```

Outputs one directory per unit, `rois_linescan/<MUnit_n>/` beside the file (or
`<output>/<MUnit_n>/`):

| file | contents |
|------|----------|
| `F.npy` | K x T mean over each line, in the file's converted counts (zero = no photons; MESc's raw uint16 sit ~1000 counts above that, which left in makes every dF/F several times too small) |
| `F_chan<c>.npy` | the same for every other channel (the red structural channel next to the green functional one) |
| `dfof.npy` | rolling max-min baseline dF/F, window in seconds so kHz line rates get the same seconds as a raster movie; stim frames bridged |
| `kymographs.npy`, `kymographs_chan<c>.npy` | K x bins x W position-along-line x time, 10 ms bins (NaN beyond a narrower line's width) |
| `stim_frames.npy` | frames acquired while a photostimulation pattern was active (from the `PatternSeq_AO1` curve, the rule lab4 uses); the scanner reads low on these, so derived traces bridge them |
| `stat.npy` | per ROI: `roi_index`, `height`, `width`, `npix`, plus `f0`, `peak_dfof`, `time_to_peak_s`, `response_auc`, `noise_dfof`, `snr` (stimulus-aligned when there is a stimulus, whole-run otherwise) |
| `ops.npy` | `roi_workflow` holds fs, channels, stim onsets/durations/pulse count, the paired reference Z-stack and the channel conversion used |
| `Fneu.npy`, `spks.npy`, `iscell.npy`, `rois.json` | suite2p-shaped placeholders so `mbo info` and lsp tools still open the directory |

A ROI whose baseline is at or below zero is reported as unreliable rather than
silently written.

The numbered figures read in order, like the suite2p and masknmf sets:

| figure | what to look for |
|--------|------------------|
| `01a_background_snapshot_lines.png` | every line on the raster snapshot it was drawn on (the unit's `BackgroundImagePath`, taken seconds before the scan), composite plus each channel; lines more than 1 um off that plane are dashed |
| `01b_reference_zstack_lines.png` | the lines on the paired Z-stack, one panel per slice that carries lines; the dot is the line start. The stack is the finest one whose field holds the lines and that puts at least 10 pixels along a line (a whole-cell stack never qualifies while a dendrite stack exists). A line scanned outside the stack's depth range is dashed with a `!`, drawn on the nearest slice, and its offset is in the title |
| `01c_line_zooms.png` | an 8 um crop around every line, each channel, local contrast, from the snapshot or the stack slice at the line's depth: the line should cross a bright spine or shaft. This is the panel to check first when an overlay looks wrong |

The micron-to-pixel convention (translation = the array's corner, rows grow
with +y, no mirror) was verified on real data: 36 of 40 in-plane lines are
brighter than random same-shaped lines nearby, the mirrored mapping is at
chance. `--flip-y` remains for a rig that saves the other way round.
| `02_line_profiles.png` | time-averaged counts along each line, both channels, shared y-axis: a bump is a spine, a flat line at background missed |
| `03a_kymographs_green.png`, `03b_kymographs_red.png` | position x time per ROI, stimulus marked |
| `04a_traces_raw.png`, `04b_traces_dfof.png` | stacked per-ROI traces over the run |
| `05_stim_response.png` | stimulus-aligned dF/F (F0 = 1 s before onset): ROI x time heatmap, every ROI, mean +/- s.e.m. |
| `06_roi_response_metrics.png` | per-ROI F0 (both channels), peak dF/F, latency, noise, SNR, response integral |
| `07_motion_correction.png` | the AOD's real-time motion correction in X/Y/Z over the run |

`--no-figures` skips them.

`mbo scan.mesc` opens the image viewer on the file's first line-scan unit
with no unit prompt. With vnoiser installed the Event Curation widget comes
on by itself: the experiment's `PF` folder when one sits beside the file
(its recordings scoped to the unit on screen), else every line of the file's
line-scan units as a raw recording that the wavelet denoiser runs on when
clicked. The Voltage pipeline is on the Process tab. `mbo <animal>/<expt>`
or `mbo <expt>/PF` opens the folder the same way: `imread` returns a
`PfArray` for a `PF` folder, like a suite2p output folder, whose image is
the line scan it came from (named in `pipeline.json`, or laid out the
notebook's way beside it) or, without that file, a raster of the denoised
traces. The notebook's dashboard on its own, with no image, is
`mbo curate PATH` (`python -m mbo_utilities.gui.curation_viewer PATH` from
a script): the trace and its candidates over the template, focused candidate
and PCA, one recording at a time with arrows to flip, and the recordings
table beside it.

To curate from another machine, serve the dashboard instead of opening a
window: `mbo curate PATH --serve` (or `python -m mbo_utilities.gui.curation_server
PATH`) renders it where the data and the GPU are and streams it to any browser
that opens the printed URL (default `http://localhost:60649/`); pointer and key
events go back the same way, so nothing is installed on the laptop. Every
connected browser sees the same frames and the longest-connected one drives.
There is no login: leave `--host` on localhost and tunnel
(`ssh -L 60649:localhost:60649 server`), or front it with an authenticating
proxy; `--host 0.0.0.0` opens it to the network as is.

To scrub the lines on the stack instead, run `mbo linescan scan.mesc --view`
(or `mbo linescan <animal>/<expt> --view`): three panels on top (the
line-scan itself, the snapshot the lines were drawn on, the paired Z-stack)
and the curation dashboard, scoped to the scan on screen, on the strip
above them. The Z-stack picker shows every
stack's fit (fraction of lines in its field and depth range, pixel size)
and defaults to the paired one; a stack holding none of the lines is
refused. `--dry-run` prints the choice and placement without a window,
`--screenshot out.png` renders the window offscreen.

## Voltage

The spatial JEDI pipeline (Noguchi & Terada) on a `.mesc` with AOD ROI units,
producing the `PF` folder the curation window reads. A unit's ROIs are the
lines of a line scan or the patches of a chessboard or ribbon scan; each unit
is one scan. Each ROI's mean fluorescence per frame is read as above, the ROIs
of a domain (the soma, one branch, one cell's patch) are averaged
pixel-weighted, dF/F and a sign-flipped z-score follow, then vnoiser's wavelet
denoiser with the archive's settings, then the peak detector. The settings
are written for the archive's 1075 Hz line scans; the parameters counted in
samples are scaled to each scan's frame rate and the peak band-pass is capped
below Nyquist, so a 200 Hz chessboard scan keeps the same baseline durations.
One run takes scans of one frame rate.

```bash
mbo voltage stan112_expt12.mesc --init                      # domains.json template beside the file
mbo voltage stan112_expt12.mesc                             # every scan in domains.json
mbo voltage stan112_expt12.mesc --unit MUnit_35 -o PF_new   # one scan, elsewhere
mbo voltage stan112_expt12.mesc --domains PF/scanIDs_ROIs.pkl --overwrite
mbo curate X:/data/asako/stan112/stan112_expt12             # then curate it
```

`domains.json` names the domains and their 0-based ROI indices (lines or
patches, in drawing order), the scans in order and the first scan of each
environment. `--init` groups lines in threes and patches one per domain:

```json
{"domains": {"soma1": [0, 1, 2], "basal1": [3, 4, 5]}, "scans": ["35", "38"], "first_env": ["35"]}
```

Written to `<animal>/<expt>/PF` for the archive layout
(`<expt>/<expt>/<expt>.mesc`), else `PF` beside the file:

| file | contents |
|------|----------|
| `denoised_trace_scans.pkl` | `{scan: {domain: trace}}`, the trace the curation window shows (the masked wavelet sum, no baseline) |
| `fs_scans.pkl`, `scanIDs_ROIs.pkl` | frame rate (rounded down, as the archive stored it) and the scan / domain / ROI tables |
| `detected_events_peaks.pkl`, `param_spike_detect.pkl` | peaks per domain and the thresholds (`--events LO,HI,BP_SD,AMP_SD,DUR_MS`) |
| `denoised_trace_components.pkl`, `test.h5` | the masked sum, 1 Hz / 100 Hz baselines and envelope; per-domain dF/F and z |
| `cwts.h5` | the wavelet coefficients, only with `--save-cwt` (large) |
| `pipeline.json` | provenance: source file and units, every parameter, versions, ROI pixel weights |
| `traces/` | the same results as plain files, see below |

`PF/traces/` needs only numpy and pandas to read
(`demos/voltage_results.ipynb` walks through it). Row `i` of every
`(domain, frame)` array is row `i` of `domains.csv`:

| file | contents |
|------|----------|
| `scans.csv` | scan id, MESc unit, frame rate, frames, ROIs |
| `domains.csv` | row index, domain name, ROI indices |
| `scan<id>_rois.npy` | raw mean fluorescence, `(ROI, frame)` |
| `scan<id>_dfof.npy`, `scan<id>_zscore.npy`, `scan<id>_denoised.npy` | dF/F, z-score and denoised trace per domain, `(domain, frame)` |
| `scan<id>_peaks.csv` | detected events: domain, frame, time in seconds |

Run on `stan112_expt12`'s raw scans with the archive's `scanIDs_ROIs.pkl`, the
output reproduces the archive's PF traces to float precision (the peaks on 7 of
16 domains exactly, the rest with one to nine extra borderline events); the
conversion is left off (`--convert` applies the file's offset so zero means no
photons, which the archive never did).

The same pipeline is the **Voltage** entry of the viewer's Process tab (`mbo
scan.mesc`, or `mbo <expt>/PF` to run it again on a folder's source scan): the dataset block, output folder, slice popup
(a frame window and the channel; every line is used), a Scans block ticking
which units become scans, a Domains table naming which lines make each domain
(loaded from or saved to `domains.json`, seeded from a `PF` folder beside the
file when one exists), the settings popup with the archive's values as
defaults, and Run, which spawns a worker the process console tracks. When the
folder is written, "Open in Curation" loads it. The standalone curation
window (`mbo curate`) only reads results: every scan of the PF folder is
listed, and `mbo scan.mesc` opens on the first scan the folder holds with
the curation following the unit on screen.

## Formats

```bash
mbo formats
```

**Input:** `.tif`, `.tiff`, `.zarr`, `.bin`, `.h5`, `.hdf5`, `.npy`, `.json`
**Output:** `.tiff`, `.zarr`, `.bin`, `.h5`, `.npy`

## Upgrade

| Method | Command |
|--------|---------|
| Install script | Re-run install script |
| CLI only | `uv tool upgrade mbo_utilities` |
| Virtual env | `uv pip install -U mbo_utilities` |
