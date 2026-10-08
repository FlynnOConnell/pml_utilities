"""masknmf pipeline integration (optional; requires the `masknmf` package).

Stage-gated registration -> PMD compression -> demixing per z-plane, each
into a masknmf run folder (``results.hdf5``, ``config.json``, the log) with a
results zarr for the shared tooling. See `runner.run_plane` / `runner.run_volume`.
"""

_LAZY = {
    "MasknmfSettings": ("mbo_utilities.masknmf.params", "MasknmfSettings"),
    "MasknmfRegistrationSettings": (
        "mbo_utilities.masknmf.params",
        "MasknmfRegistrationSettings",
    ),
    "MasknmfCompressionSettings": (
        "mbo_utilities.masknmf.params",
        "MasknmfCompressionSettings",
    ),
    "MasknmfDemixingSettings": (
        "mbo_utilities.masknmf.params",
        "MasknmfDemixingSettings",
    ),
    "MasknmfRuntimeSettings": (
        "mbo_utilities.masknmf.params",
        "MasknmfRuntimeSettings",
    ),
    "stage_action": ("mbo_utilities.masknmf.params", "stage_action"),
    "run_plane": ("mbo_utilities.masknmf.runner", "run_plane"),
    "run_volume": ("mbo_utilities.masknmf.runner", "run_volume"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
