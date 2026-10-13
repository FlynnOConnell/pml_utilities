# ASAP7 spine pipeline on BioHPC

`asap7_spine.py` runs `asap7_spine_pipeline_detrend.ipynb` without a notebook, on one unit of a `.mesc` file.
`run_asap7.sh` is its SLURM worker. The Lab4 window (`imgui_lab4_interface`) submits one job per unit after it
transfers the `.mesc`; this folder is what those jobs run.

## Set up (once, on a BioHPC login node)

The window expects this checkout at `~/repos/pml_utilities` and the env at `~/envs/asap7`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv is missing
export TMPDIR=$HOME/.cache/uv-tmp && mkdir -p "$TMPDIR"  # see ../README.md: /tmp is too small for torch wheels

git clone https://github.com/FlynnOConnell/pml_utilities ~/repos/pml_utilities
mkdir -p ~/envs/asap7
cp ~/repos/pml_utilities/hpc/asap7/pyproject.toml ~/envs/asap7/
cd ~/envs/asap7 && uv sync
```

`pyproject.toml` installs this checkout editable (`../../repos/pml_utilities`), masknmf-toolbox at a pinned commit,
and CUDA 12.8 torch. A `git pull` in `~/repos/pml_utilities` updates the script and `mbo_utilities` together; to move
masknmf, change its `rev` and `uv sync` again.

CUDA 12.8 wheels need an NVIDIA driver of 570 or newer on the GPU nodes. `run_asap7.sh` prints the driver in each
job's log; if it is older, switch the index to `cu126` and `uv sync`.

## Check it

```bash
~/envs/asap7/.venv/bin/python ~/repos/pml_utilities/hpc/asap7/asap7_spine.py --help
sbatch --partition=<gpu partition> --gres=gpu:1 --output=$HOME/asap7_test_%j.out \
    ~/repos/pml_utilities/hpc/asap7/run_asap7.sh \
    ~/repos/pml_utilities/hpc/asap7/asap7_spine.py /path/to/scan.mesc MSession_0/MUnit_14 0 200 0.6 0.3 \
    /path/to/masknmf_asap7/MUnit_14_test
```

## What a run writes

```text
<dir_save>/
├── asap7.json                              # the arguments
├── logs/run.log                            # this script's log; the window points SLURM's at logs/slurm_<job>.out
└── <timestamp>_glutamate-calcium-spine/    # masknmf's run folder: config.json, results.glutamate.hdf5, its log
```

Open the results with `masknmf view <run folder>/results.glutamate.hdf5`, or double-click the run folder in the
window's Browse storage.
