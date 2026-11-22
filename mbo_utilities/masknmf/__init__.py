"""masknmf from pml_utilities (optional; requires the `masknmf` package).

pml_utilities hands each z-plane to masknmf's own pipeline as a ``(T, Y, X)``
view and writes a results zarr beside masknmf's run folder. See
`pipeline.run_planes` and `pipeline.plane_view`.
"""

_LAZY = {
    "pipeline_class": ("mbo_utilities.masknmf.pipeline", "pipeline_class"),
    "plane_view": ("mbo_utilities.masknmf.pipeline", "plane_view"),
    "run_planes": ("mbo_utilities.masknmf.pipeline", "run_planes"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
