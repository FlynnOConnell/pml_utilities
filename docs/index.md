# PML Utilities

Readers, pipelines and viewers for the lab's imaging data: a fork of
[mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities) for Program in
Memory Longevity (PML).

## Contents

```{toctree}
---
maxdepth: 1
---
User Guide <user_guide>
Command Line Interface <usage/cli>
HPC / SLURM <usage/hpc>
The Studio <usage/gui_guide>
Supported File Formats <file_formats>
DF/F Analysis <dff>
API <api/index>
development
glossary
```

## Quick Start

```bash
# install
uv pip install "pml_utilities[all] @ git+https://github.com/FlynnOConnell/pml_utilities.git"

# launch gui
mbo

# convert data
mbo convert /data/raw output/ -e .zarr
```

## Resources

- [GUI User Guide (PDF)](_static/mbo_gui_user_guide.pdf)
- [Virtual Environments Guide](https://millerbrainobservatory.github.io/guides/venvs.html)
- [uv cheatsheet](https://www.saaspegasus.com/guides/uv-deep-dive/#cheatsheet-common-operations-in-uvs-workflows)
