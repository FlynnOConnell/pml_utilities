"""vnoiser integration (optional; requires the ``vnoiser`` package).

Wavelet denoising and event curation of single voltage traces. The curation
logic, data loading and the saved JSON format all live in vnoiser; this
package only adapts them to the imgui viewer. See ``session.CurationSession``.
"""

from mbo_utilities.vnoiser.session import (
    LABEL_RGBA,
    MODES,
    PC1_SIDES,
    CurationSession,
    PfScan,
    hex_rgba,
    pf_dir_for_mesc,
    pf_scan_for_mesc,
)

__all__ = [
    "LABEL_RGBA",
    "MODES",
    "PC1_SIDES",
    "CurationSession",
    "PfScan",
    "hex_rgba",
    "pf_dir_for_mesc",
    "pf_scan_for_mesc",
]
