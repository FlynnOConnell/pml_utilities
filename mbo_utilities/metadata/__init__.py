"""
mbo_utilities.metadata - metadata handling for calcium imaging data.

this package provides:
- standardized parameter definitions and aliases
- scanimage-specific metadata parsing
- stack type detection (lbm, piezo, pollen, single plane)
- voxel size extraction and normalization
- file I/O for extracting metadata from TIFF files
"""

from mbo_utilities.arrays.features._roi import RoiMode

# filename metadata parsing
from ._filename_parser import (
    FilenameMetadata,
    get_filename_suggestions,
    parse_filename_metadata,
)
from .base import (
    ALIAS_MAP,
    EXPORT_DENYLIST,
    IMAGING_METADATA_KEYS,
    METADATA_PARAMS,
    MetadataParameter,
    VoxelSize,
    get_canonical_name,
    normalize_ops_arrays,
    repair_ops_file,
    repair_ops_tree,
    strip_for_export,
)

# file I/O functions
from .io import (
    _build_ome_metadata,
    clean_scanimage_metadata,
    default_ops,
    get_metadata,
    get_metadata_batch,
    get_metadata_single,
    is_raw_scanimage,
    query_tiff_pages,
)

# output metadata for subsetted data
from .output import OutputMetadata
from .params import (
    get_param,
    get_voxel_size,
    normalize_metadata,
    normalize_resolution,
    scale_frame_rate,
)
from .scanimage import (
    StackType,
    detect_stack_type,
    extract_roi_slices,
    get_beamlets_per_port,
    get_color_channel_ports,
    get_frame_rate,
    get_frames_per_slice,
    get_frames_per_volume,
    get_log_average_factor,
    get_num_color_channels,
    get_num_slices,
    get_num_volumes,
    get_num_zplanes,
    get_roi_info,
    get_saved_channel_ports,
    get_z_step_size,
    is_lbm_stack,
    is_piezo_stack,
)

__all__ = [
    "ALIAS_MAP",
    # imaging metadata (core params for display/editing)
    "IMAGING_METADATA_KEYS",
    # writer denylist + helper (suite2p-only fields stripped from non-suite2p output)
    "EXPORT_DENYLIST",
    "normalize_ops_arrays",
    "repair_ops_file",
    "repair_ops_tree",
    "strip_for_export",
    "METADATA_PARAMS",
    # base types
    "MetadataParameter",
    "RoiMode",
    # scanimage detection
    "StackType",
    "VoxelSize",
    "_build_ome_metadata",
    "clean_scanimage_metadata",
    "default_ops",
    "detect_stack_type",
    "extract_roi_slices",
    "get_canonical_name",
    "get_frame_rate",
    "get_frames_per_slice",
    "get_saved_channel_ports",
    "get_color_channel_ports",
    "get_beamlets_per_port",
    "get_log_average_factor",
    "get_metadata",
    "get_metadata_batch",
    "get_metadata_single",
    "get_num_color_channels",
    "get_num_slices",
    "get_num_volumes",
    "get_num_zplanes",
    "get_frames_per_volume",
    # parameter access
    "get_param",
    "scale_frame_rate",
    "get_roi_info",
    "get_voxel_size",
    "get_z_step_size",
    # file I/O
    "is_lbm_stack",
    "is_piezo_stack",
    "is_raw_scanimage",
    "normalize_metadata",
    "normalize_resolution",
    "query_tiff_pages",
    # output metadata
    "OutputMetadata",
    # filename parsing
    "FilenameMetadata",
    "parse_filename_metadata",
    "get_filename_suggestions",
]
