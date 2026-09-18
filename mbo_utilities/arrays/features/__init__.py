"""
Array features for mbo_utilities.

Features are composable properties that can be attached to array classes.
Following the fastplotlib pattern, each feature is a self-contained class
that manages its own state and events.

Available features:
- DimensionSpecs: dimension structure (names, roles, scales) + OME axes
- RoiFeatureMixin: multi-ROI handling
- PhaseCorrectionFeature: bidirectional scan correction
- FrameAverageFeature: temporal binning, plus apply_read_features() which
  turns read-option kwargs (fix_phase, use_fft, frame_average, ...) into the
  array a worker should read from
"""

from __future__ import annotations

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent
from mbo_utilities.arrays.features._chunks import (
    CHUNKS_2D,
    CHUNKS_3D,
    CHUNKS_4D,
    estimate_chunk_memory,
    normalize_chunks,
)
from mbo_utilities.arrays.features._dim_labels import (
    DEFAULT_DIMS,
    DIM_DESCRIPTIONS,
    KNOWN_ORDERINGS,
    find_slider_name,
    get_dim_index,
    get_dims,
    get_num_planes,
    get_slider_dims,
    infer_dims,
    parse_dims,
)
from mbo_utilities.arrays.features._dim_tags import (
    DIM_ALIASES,
    DimensionTag,
    OutputFilename,
    SPATIAL_DIMS,
    TAG_REGISTRY,
    TagDefinition,
    dim_to_ome_axis,
    dims_to_ome_axes,
    get_ome_axis_type,
    get_ome_axis_unit,
    normalize_dims,
)
from mbo_utilities.arrays.features._dim_spec import (
    DimRole,
    DimensionSpec,
    DimensionSpecs,
)
from mbo_utilities.arrays.features._frame_average import (
    FRAME_AVERAGE_KEY,
    PHASE_FEATURE_KEYS,
    READ_FEATURE_KEYS,
    FrameAverageFeature,
    apply_read_features,
)
from mbo_utilities.arrays.features._motion import MOTION_AXES, MotionCorrection
from mbo_utilities.arrays.features._phase_correction import (
    PhaseCorrectionFeature,
    PhaseCorrectionMixin,
    PhaseCorrMethod,
)
from mbo_utilities.arrays.features._roi import RoiFeatureMixin
from mbo_utilities.arrays.features._segmentation import (
    masks_to_stat,
    stat_to_masks,
)
from mbo_utilities.arrays.features._stats import (
    PlaneStats,
    SliceStats,
)
from mbo_utilities.arrays.features._summary_stats import (
    DEFAULT_BUDGET,
    DEFAULT_METRICS,
    StatsDim,
    StatsDimRole,
    StatsMetric,
    SubsampleBudget,
    SummaryStatsSpec,
    build_summary_stats_spec,
    canonical_axis,
    default_dim_role,
    subsample_indices,
)
from mbo_utilities.arrays.features._slicing import (
    ArraySlicing,
    ChunkInfo,
    DimSelection,
    TimeSelection,
    normalize_dim_key,
    parse_selection,
    parse_timepoint_selection,
    read_chunk,
)
from mbo_utilities.arrays.features._selection import (
    canonical_axis_sizes,
    selection_to_canonical,
    to_isoview_kwargs,
    to_lsp_kwargs,
)
from mbo_utilities.arrays.features._pyramid import (
    DownsampleMethod,
    PyramidConfig,
    PyramidLevel,
    compute_pyramid_shapes,
    downsample_block,
)

__all__ = [
    "CHUNKS_2D",
    "CHUNKS_3D",
    "CHUNKS_4D",
    "DEFAULT_DIMS",
    "DIM_ALIASES",
    "DIM_DESCRIPTIONS",
    "KNOWN_ORDERINGS",
    "MOTION_AXES",
    "SPATIAL_DIMS",
    "TAG_REGISTRY",
    # base
    "ArrayFeature",
    "ArrayFeatureEvent",
    # dim tags
    "DimensionTag",
    "OutputFilename",
    "TagDefinition",
    "dim_to_ome_axis",
    "dims_to_ome_axes",
    "get_ome_axis_type",
    "get_ome_axis_unit",
    "normalize_dims",
    # motion correction
    "MotionCorrection",
    # dim specs
    "DimRole",
    "DimensionSpec",
    "DimensionSpecs",
    "PhaseCorrMethod",
    # phase correction
    "PhaseCorrectionFeature",
    "PhaseCorrectionMixin",
    # frame averaging / read-time features
    "FRAME_AVERAGE_KEY",
    "PHASE_FEATURE_KEYS",
    "READ_FEATURE_KEYS",
    "FrameAverageFeature",
    "apply_read_features",
    # stats
    "PlaneStats",
    "SliceStats",
    # summary stats
    "DEFAULT_BUDGET",
    "DEFAULT_METRICS",
    "StatsDim",
    "StatsDimRole",
    "StatsMetric",
    "SubsampleBudget",
    "SummaryStatsSpec",
    "build_summary_stats_spec",
    "canonical_axis",
    "default_dim_role",
    "subsample_indices",
    # roi
    "RoiFeatureMixin",
    "estimate_chunk_memory",
    "find_slider_name",
    "get_dim_index",
    "get_dims",
    "get_num_planes",
    "get_slider_dims",
    "infer_dims",
    "masks_to_stat",
    "normalize_chunks",
    "parse_dims",
    "stat_to_masks",
    # slicing
    "ArraySlicing",
    "ChunkInfo",
    "DimSelection",
    "TimeSelection",
    "normalize_dim_key",
    "parse_selection",
    "parse_timepoint_selection",
    "read_chunk",
    # canonical selection conversion
    "canonical_axis_sizes",
    "selection_to_canonical",
    "to_isoview_kwargs",
    "to_lsp_kwargs",
    # pyramid
    "DownsampleMethod",
    "PyramidConfig",
    "PyramidLevel",
    "compute_pyramid_shapes",
    "downsample_block",
]
