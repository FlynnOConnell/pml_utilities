> [!NOTE]
> Install `pml_utilities` in a virtual environment with [uv](https://docs.astral.sh/uv/):
>
> ```bash
> uv venv --python 3.12.9
> .venv\Scripts\activate.ps1        # Windows PowerShell
> source .venv/bin/activate         # Linux, macOS
> uv pip install "pml_utilities[gui]"
> ```

<p align="center">
<img src="https://raw.githubusercontent.com/FlynnOConnell/pml_utilities/main/mbo_utilities/assets/static/logo_utilities.png" height="110" alt="MBO Utilities logo">
</p>

# pml_utilities

Readers, pipelines and viewers for imaging data, for the Program in Memory Longevity
(PML). It installs the `mbo_utilities` Python package and the `mbo` command.

> **Beta channel.** `pml_utilities` is
> [mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities) plus
> features that are still in beta. A feature lands here first and moves into the core
> `mbo_utilities` once it settles, so expect changes between releases. The
> documentation is [mbo_utilities'](https://millerbrainobservatory.github.io/mbo_utilities/).

- **Lazy reading and writing** of ScanImage and other TIFFs, Femtonics MESc, Bruker
  and lab4 HDF5, Zarr, suite2p folders, NumPy and IsoView trees, each as one
  `(T, C, Z, Y, X)` array
- **The Studio**, a GPU viewer built on [fastplotlib](https://fastplotlib.org):
  hand-drawn ROIs, traces, motion correction and behavior on one time axis
- **Pipelines** from the Studio or the command line: suite2p and, installed
  separately, MaskNMF and the voltage imaging pipelines, each writing one results zarr
- **MESc line-scan tools**: named ROIs, trace figures (`mbo dashboard`) and event
  curation (`mbo curate`)

## Extras

Python 3.12.7 to 3.13. The base install reads and writes every format and has no
window. `pml_utilities` and the PyPI `mbo_utilities` install the same package, so they
cannot share an environment.

```bash
uv pip install "pml_utilities[gui]"    # The Studio
uv pip install "pml_utilities[all]"    # every extra
```

| Extra | Adds |
|-------|------|
| `gui` | The Studio and the ROI, trace and curation tools; the ROI tool also needs masknmf, installed separately |
| `suite2p` | suite2p and rastermap; then `uv pip install --no-deps lbm-suite2p-python` |
| `server` | `mbo serve`, The Studio in a browser |
| `hpc` | submitit, to run suite2p on SLURM |
| `napari` | the napari viewer mode |
| `notebooks` | JupyterLab, for the notebooks `mbo init` writes |
| `video` | ffmpeg, for mp4 export |
| `codecs` | imagecodecs, for compressed TIFFs |
| `pyramid` | numba, for mode-downsampled zarr pyramids |
| `dask` | `files_to_dask` |
| `qt` | PyQt6 window backend; glfw is the default |

`mbo --check-install` lists every package, the GPU, and the install command for
anything missing, including Qt's system libraries on a minimal Linux install.

## Use

```bash
mbo                        # open The Studio
mbo /path/to/data          # open a file or folder
mbo info /path/to/data     # shape and metadata
mbo convert IN OUT         # convert between formats
mbo dashboard scan.mesc    # trace figures of a .mesc unit's ROIs
mbo curate scan.mesc       # event curation of a line scan
mbo --help                 # every command
```

```python
import mbo_utilities as mbo

arr = mbo.imread("scan.mesc", unit="MUnit_3")   # lazy, (T, C, Z, Y, X)
mbo.imwrite(arr, "out", ext=".zarr")
```

## Uninstall

```bash
uv pip uninstall pml_utilities
rm -rf ~/.mbo          # settings, logs and caches
```

## Issues

Report problems on [GitLab](https://gitlab.com/losonczylab/pml_utilities/-/issues).
