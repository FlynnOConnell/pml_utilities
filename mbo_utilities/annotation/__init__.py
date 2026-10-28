"""ROI annotation model + persistence, GUI-free.

The manual ROI widget (``gui/manual_roi.py``) is a thin imgui/pygfx shell
over this package: the ROI store, the trace table, the session model that
binds them and the rois zarr they are saved to. See README.md for
the extraction seam shared with masknmf-toolbox's ``visualization/imgui``
layer.
"""

from mbo_utilities.annotation.display import (
    DISPLAY_KINDS,
    SUBTRACTED,
    TRACE_PROFILES,
    DffSettings,
    TraceProfile,
    available_kinds,
    deflect,
    display_trace,
    displayed_kind,
    register_trace_profile,
    trace_profile,
    y_label,
)
from mbo_utilities.annotation.events import ModelEvent, Observable
from mbo_utilities.annotation.imagej import (
    export_footprints_npz,
    export_label_tiff,
    export_roiset,
    export_traces,
    import_roiset,
)
from mbo_utilities.annotation.model import COLUMNS, RoiModel, RunTarget
from mbo_utilities.annotation.ngff import (
    SAVED_SOURCES,
    RoiFile,
    RoiFileContents,
    write_label_image,
)
from mbo_utilities.annotation.store import (
    CLASS_COLORS,
    ROI_COLORS,
    UNLABELED,
    RoiLabelStore,
    RoiRecord,
    class_color,
)
from mbo_utilities.annotation.traces import (
    ENGINES,
    RoiTrace,
    RoiTraceTable,
    trace_key,
)

__all__ = [
    "CLASS_COLORS",
    "COLUMNS",
    "DISPLAY_KINDS",
    "ENGINES",
    "ROI_COLORS",
    "SAVED_SOURCES",
    "SUBTRACTED",
    "TRACE_PROFILES",
    "UNLABELED",
    "DffSettings",
    "ModelEvent",
    "Observable",
    "RoiFile",
    "RoiFileContents",
    "RoiLabelStore",
    "RoiModel",
    "RoiRecord",
    "RoiTrace",
    "RoiTraceTable",
    "RunTarget",
    "TraceProfile",
    "available_kinds",
    "deflect",
    "class_color",
    "display_trace",
    "displayed_kind",
    "export_footprints_npz",
    "export_label_tiff",
    "export_roiset",
    "export_traces",
    "import_roiset",
    "register_trace_profile",
    "trace_key",
    "trace_profile",
    "write_label_image",
    "y_label",
]
