"""ROI annotation model + persistence, GUI-free.

The manual ROI widget (``gui/manual_roi.py``) and the Process tab's ROI
pipeline are thin imgui/pygfx shells over this package: the label store,
the trace table and the session model that binds them. See README.md for
the extraction seam shared with masknmf-toolbox's ``visualization/imgui``
layer.
"""

from mbo_utilities.annotation.events import ModelEvent, Observable
from mbo_utilities.annotation.model import COLUMNS, RoiModel, RunTarget
from mbo_utilities.annotation.ngff import LabelsZarr
from mbo_utilities.annotation.store import (
    CLASS_COLORS,
    ROI_COLORS,
    UNLABELED,
    RoiLabelStore,
    RoiRecord,
    class_color,
)
from mbo_utilities.annotation.traces import ENGINES, FULL_IMAGE, RoiTrace, RoiTraceTable, trace_key

__all__ = [
    "CLASS_COLORS",
    "COLUMNS",
    "ENGINES",
    "FULL_IMAGE",
    "ROI_COLORS",
    "UNLABELED",
    "LabelsZarr",
    "ModelEvent",
    "Observable",
    "RoiLabelStore",
    "RoiModel",
    "RoiRecord",
    "RoiTrace",
    "RoiTraceTable",
    "RunTarget",
    "class_color",
    "trace_key",
]
