> **A fork of [mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities) for Program in Memory Longevity (PML)**

<p align="center">
<img src="mbo_utilities/assets/static/logo_utilities.png" height="110" alt="MBO Utilities logo">
</p>

<p align="center">
<a href="https://github.com/FlynnOConnell/pml_utilities/actions/workflows/install-check.yml"><img src="https://github.com/FlynnOConnell/pml_utilities/actions/workflows/install-check.yml/badge.svg" alt="Install from git URL"></a>
</p>

<p align="center">
<a href="#installation"><b>Installation</b></a> ·
<a href="docs/usage/gui_guide.md"><b>The Studio</b></a> ·
<a href="docs/usage/cli.md"><b>CLI</b></a> ·
<a href="docs/file_formats.md"><b>Formats</b></a> ·
<a href="https://gitlab.com/losonczylab/pml_utilities/-/issues"><b>Issues</b></a>
</p>

Readers, pipelines and viewers for the lab's imaging data. It installs the
`mbo_utilities` Python package and the `mbo` command.

- **Lazy reading and writing** of ScanImage and other TIFFs, Femtonics MESc, Bruker
  and lab4 HDF5, Zarr, suite2p folders, NumPy and IsoView trees, each as one
  `(T, C, Z, Y, X)` array
- **The Studio**, a GPU viewer built on [fastplotlib](https://fastplotlib.org/user_guide/guide.html#what-is-fastplotlib):
  hand-drawn ROIs, traces, motion correction and behavior on one time axis; `mbo serve`
  shows it in a browser
- **Pipelines** from the Studio or the command line: suite2p, MaskNMF, vnoiser and
  pre-registration denoising, each writing one results zarr
- **MESc line-scan tools**: vnoiser domain tables, trace figures (`mbo dashboard`) and
  event curation (`mbo curate`)

<p align="center">
  <img src="docs/_images/gui/readme/01_step_file_dialog.png" height="280" alt="File Selection" />
  <img src="docs/_images/gui/readme/02_step_data_view.png" height="280" alt="Data Viewer" />
  <img src="docs/_images/gui/readme/03_metadata_viewer.png" height="280" alt="Metadata Viewer" />
  <br/>
  <em>Select data, visualize, and inspect metadata</em>
</p>

## Installation

Python 3.12.7 to 3.13. The distribution is `pml_utilities`; it installs the
`mbo_utilities` package and the `mbo` command, so it cannot share an environment
with the PyPI `mbo_utilities`.

```bash
uv venv --python 3.12.9
uv pip install "pml_utilities[all] @ git+https://github.com/FlynnOConnell/pml_utilities.git"
```

Without extras you get the readers, The Studio and MaskNMF. The extras add:

| Extra | Adds |
|-------|------|
| `vnoiser` | the vnoiser pipeline (`mbo vnoiser`) and its curation methods |
| `suite2p` | suite2p and rastermap; the suite2p pipeline also needs `lbm-suite2p-python`, below |
| `server` | `mbo serve` |
| `hpc` | submitit, to run suite2p on SLURM |
| `napari` | the napari viewer mode |
| `notebooks` | JupyterLab and jupyter_rfb, for the notebooks `mbo init` writes |
| `docs` | Sphinx, to build `docs/` |
| `all` | every extra but `docs` |

The suite2p pipeline also needs `lbm-suite2p-python`. Install it without its
dependencies: it asks for the PyPI `mbo_utilities`, which this package replaces, and
`[all]` already brings the rest.

```bash
uv pip install --no-deps lbm-suite2p-python
```

Upgrade to the newest `main`:

```bash
uv pip install --reinstall-package pml-utilities "pml_utilities[all] @ git+https://github.com/FlynnOConnell/pml_utilities.git"
```

Add `@<branch>` or `@<commit>` to the URL for anything other than `main`. The GitHub
repository is a public mirror of the lab's GitLab one. A branch that hasn't reached
the mirror installs from GitLab over SSH, with your SSH key added to GitLab:

```bash
uv pip install "pml_utilities[all] @ git+ssh://git@gitlab.com/losonczylab/pml_utilities.git@<branch>"
```

The GitLab `https://` URL does not work for an install: the repository is private and
uv cannot ask for a login.

Quick viewer without installing:

```bash
uvx --from "git+https://github.com/FlynnOConnell/pml_utilities.git" mbo /path/to/data
```

On a minimal Ubuntu/Debian install the viewer's window needs Qt's system libraries:

```bash
sudo apt install --no-install-recommends libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-render0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libxkbcommon0 libx11-xcb1 libegl1 libgl1 libfontconfig1 libdbus-1-3 ffmpeg
```

Suite2p on a GPU needs the CUDA build of pytorch; see the pytorch
[Get Started](https://pytorch.org/get-started/locally/) page for your CUDA version:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`mbo --check-install` reports every package, GPU availability and the install
command for anything missing.

## Usage

`mbo init path/to/data` writes the starter notebooks with your data path filled in.
The [CLI guide](docs/usage/cli.md), [Studio guide](docs/usage/gui_guide.md),
[formats guide](docs/file_formats.md) and [HPC guide](docs/usage/hpc.md) cover the rest.

### Supported Formats

| Format | Read | Write |
|--------|:----:|:-----:|
| ScanImage, ImageJ, OME and plain TIFF | ✓ | ✓ |
| Femtonics MESc (`.mesc`) | ✓ | |
| HDF5, including Bruker and lab4 | ✓ | ✓ |
| Zarr (OME-NGFF 0.5) | ✓ | ✓ |
| Suite2p folders and `.bin` | ✓ | ✓ |
| NumPy (`.npy` or in memory) | ✓ | ✓ |
| IsoView light-sheet trees | ✓ | |
| suite2p, MaskNMF and vnoiser results | ✓ | ✓ (`mbo results`) |

### Commands

| Command | Description |
|---------|-------------|
| `mbo [PATH]` | open The Studio, on a file or folder if given |
| `mbo info PATH` | shape and metadata of a dataset |
| `mbo convert IN OUT` | convert between formats |
| `mbo formats` | list the formats read and written |
| `mbo scanphase PATH` | bidirectional scan-phase analysis |
| `mbo vnoiser FILE.mesc` | the vnoiser pipeline on a `.mesc`'s line scans |
| `mbo curate [PATH]` | event curation of a vnoiser run or a line-scan `.mesc` |
| `mbo dashboard [PATH]` | trace figures of a `.mesc` unit's ROIs, for slides |
| `mbo results PATH` | write a suite2p, MaskNMF or vnoiser folder as one results zarr |
| `mbo reg-denoise PATH` | pre-registration denoising: compress, register, compress again |
| `mbo spike-average PATH` | spike-triggered averages of one ROI of a run |
| `mbo asap7 PATH` | the ASAP7 spine pipeline |
| `mbo serve` | serve The Studio to browsers |
| `mbo init [DATA_PATH]` | write the starter notebooks |
| `mbo processes` | list or stop background runs |
| `mbo gpu` | show which GPU renders and which computes |
| `mbo download-models` | download the cellpose model suite2p detection uses |
| `mbo shortcut` | add a desktop icon that opens The Studio |
| `pollen` | pollen calibration |

### The Studio

Launch an interactive GPU-accelerated viewer for exploring large imaging datasets. Supports all MBO file formats with real-time visualization.

```bash
mbo                    # launch GUI
mbo /path/to/data      # open file directly
mbo --check-install    # verify GPU configuration
mbo shortcut           # add a desktop icon (Windows/Linux)
```

→ [GUI Guide](docs/usage/gui_guide.md)

### Scan-Phase Analysis

Measure and correct bidirectional scan-phase offset in resonant scanning microscopy data. Generates diagnostic figures showing temporal stability, spatial variation, and recommended corrections.

```bash
mbo scanphase /path/to/data.tiff -o ./output
```

### Axial (Z-plane) Registration

Compute per-plane rigid shifts that align z-planes to each other. Shifts are
stored in metadata, **not** baked into the saved pixels, so they can be applied
or removed non-destructively at read time.

```python
import mbo_utilities as mbo

# compute shifts on save; they are written to metadata["plane_shifts"]
mbo.imwrite(arr, "registered.zarr", ext=".zarr", register_z=True)

# apply them on read (reversible, source never modified)
data = mbo.imread("registered.zarr")
aligned = mbo.with_axial_shifts(data)  # reads metadata["plane_shifts"]
aligned.enabled = False  # back to the raw frames

# or supply your own shifts (one (dy, dx) row per z-plane)
shifts = [[0, 0], [2, -1], [3, -2]]  # len must equal the Z size
aligned = mbo.with_axial_shifts(data, plane_shifts=shifts)
```

In The Studio, datasets that carry valid `plane_shifts` are aligned
automatically on load. Saving the aligned view bakes the shifts into the output
pixels.

## ScanImage Acquisition Modes

`mbo_utilities` automatically detects and parses metadata from these ScanImage acquisition modes:

| Configuration | Detection | Result |
|---------------|-----------|--------|
| LBM single channel | `channelSave=[1..N]`, AI0 only | `lbm=True`, `colors=1` |
| LBM dual channel | `channelSave=[1..N]`, AI0+AI1 | `lbm=True`, `colors=2` |
| Piezo (single frame/slice) | `hStackManager.enable=False`, `framesPerSlice=1` | `piezo=True` |
| Piezo multi-frame (with avg) | `hStackManager.enable=False`, `logAvgFactor>1` | `piezo=True`, averaged frames |
| Piezo multi-frame (no avg) | `hStackManager.enable=False`, `framesPerSlice>1`, `logAvg=1` | `piezo=True`, raw frames |
| Single plane | `hStackManager.enable=False` | `zplanes=1` |

> **Note:** Frame-averaging (`logAverageFactor > 1`) is only available for non-LBM acquisitions.

## Uninstall

```bash
uv pip uninstall pml_utilities
rm -rf ~/.mbo          # settings, logs and caches
```

## Troubleshooting

<details>
<summary><b>Error: "Failed to auto-detect CUDA root directory"</b></summary>

This occurs when using GPU-accelerated features and CuPy cannot find your CUDA Toolkit.

**Check if CUDA is installed:**

```powershell
# Windows
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue
$env:CUDA_PATH
```

```bash
# Linux/macOS
nvcc --version
echo $CUDA_PATH
```

**Set CUDA_PATH:**

```powershell
# Windows (replace v12.6 with your version)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', $env:CUDA_PATH, 'User')
```

```bash
# Linux/macOS (add to ~/.bashrc or ~/.zshrc)
export CUDA_PATH=/usr/local/cuda-12.6
```

If CUDA is not installed, download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

</details>

<details>
<summary><b>Error: "libcudnn.so.9: cannot open shared object file"</b></summary>

PyTorch can't load a CUDA library that `pip list` still reports as installed — its files were removed, usually when a conflicting CUDA build was uninstalled. Reinstall every package:

```bash
uv pip install --reinstall --no-cache "pml_utilities[all] @ git+https://github.com/FlynnOConnell/pml_utilities.git"
```

In a checkout, `uv sync --reinstall --no-cache --all-extras` does the same.

</details>

<details>
<summary><b>Git LFS Download Errors</b></summary>

There is a [bug in fastplotlib](https://github.com/fastplotlib/fastplotlib/issues/861) causing `git lfs` errors when installed from a git branch.

Set `GIT_LFS_SKIP_SMUDGE=1` and restart your terminal:

```powershell
# Windows
[System.Environment]::SetEnvironmentVariable('GIT_LFS_SKIP_SMUDGE', '1', 'User')
```

```bash
# Linux/macOS
echo 'export GIT_LFS_SKIP_SMUDGE=1' >> ~/.bashrc
source ~/.bashrc
```

</details>

## Built With

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Integration support
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Visualization
- **[masknmf](https://github.com/apasarkar/masknmf-toolbox)** - MaskNMF pipeline and curation viewers
- **[fastplotlib](https://github.com/fastplotlib/fastplotlib)** - The Studio's rendering

## Issues & Support

Report problems on the lab's [GitLab](https://gitlab.com/losonczylab/pml_utilities/-/issues) or on Slack.
