# ASAP7 spine runs on BioHPC

`mbo asap7` runs `asap7_spine_pipeline_detrend.ipynb` on one unit of a `.mesc` file:

1. compress the inverted raw movie
2. register the raw movie with shifts estimated on that compression
3. compress and denoise the registered movie with a maximin spline detrender
4. demix it with masknmf's glutamate spine pipeline

```bash
mbo asap7 scan.mesc                    # list the units
mbo asap7 scan.mesc --unit MUnit_10    # run one; mbo asap7 --help for the settings
masknmf view <run folder>/results.glutamate.hdf5
```

It runs anywhere `mbo` and masknmf are installed with a GPU. `run_asap7.sh` is the SLURM worker for it; the Lab4
window (`imgui_lab4_interface`) submits one job per unit after it transfers the `.mesc`.

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
and CUDA 12.8 torch. A `git pull` in `~/repos/pml_utilities` updates `mbo asap7` and this worker together; to move
masknmf, change its `rev` and `uv sync` again.

CUDA 12.8 wheels need an NVIDIA driver of 570 or newer on the GPU nodes. `run_asap7.sh` prints the driver in each
job's log; if it is older, switch the index to `cu126` and `uv sync`.

## Check it

```bash
~/envs/asap7/.venv/bin/mbo asap7 /path/to/scan.mesc
sbatch --partition=<gpu partition> --gres=gpu:1 --output=$HOME/asap7_test_%j.out \
    ~/repos/pml_utilities/hpc/asap7/run_asap7.sh /path/to/scan.mesc MSession_0/MUnit_10 1 200 0.6 0.3 \
    /path/to/masknmf_asap7/MUnit_10_test
```

## What a run writes

```text
<out>/<timestamp>_glutamate-calcium-spine/
├── config.json, <timestamp>_glutamate-calcium-spine.log   # masknmf's
├── results.glutamate.hdf5                                  # compression and demixing
├── registration.hdf5                                       # shifts estimated on the compressed raw movie
└── asap7.json                                              # settings, source, unit, timings
```
