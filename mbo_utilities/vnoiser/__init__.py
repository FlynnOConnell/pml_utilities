"""vnoiser integration (optional; requires the ``vnoiser`` package).

Wavelet denoising and event curation of single voltage traces. The curation
logic, data loading and the saved JSON format all live in vnoiser; this
package only adapts them to the imgui viewer. See ``session.CurationSession``.
``MODES`` is importable without vnoiser, so the curation window can list them.
"""

MODES = ("fast", "slow", "manual")

_LAZY = {
    "LABEL_RGBA": ("mbo_utilities.vnoiser.session", "LABEL_RGBA"),
    "PC1_SIDES": ("mbo_utilities.vnoiser.session", "PC1_SIDES"),
    "CurationSession": ("mbo_utilities.vnoiser.session", "CurationSession"),
    "hex_rgba": ("mbo_utilities.vnoiser.session", "hex_rgba"),
    "recording_id": ("mbo_utilities.results", "recording_id"),
    "trace_label": ("mbo_utilities.results", "trace_label"),
    "vnoiser_run_for_mesc": ("mbo_utilities.results", "vnoiser_run_for_mesc"),
    "vnoiser_unit_for_mesc": (
        "mbo_utilities.vnoiser.session",
        "vnoiser_unit_for_mesc",
    ),
}

__all__ = ["MODES", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
