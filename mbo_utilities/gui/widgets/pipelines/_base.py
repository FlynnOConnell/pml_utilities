"""``PipelineWidget``: the one class that registers a pipeline with the Run tab (AGENTS.md §7.2)."""

from abc import ABC, abstractmethod
from typing import Any

from mbo_utilities.pipeline_registry import PipelineInfo

AXIS_MODES = (
    "range",
    "all",
    "none",
    "select-one",
)  # a start:stop range, the whole axis (row disabled), hidden, one index
DEFAULT_AXES_CONSUMED: dict[str, str] = {"T": "range", "Z": "range", "C": "range"}


class PipelineWidget(ABC):
    """One pipeline's Run-tab widget: its selector label, availability, ``PipelineInfo`` and per-axis consumption."""

    name: str = "Pipeline"
    is_available: bool = False
    install_command: str = 'uv pip install "pml_utilities @ git+https://github.com/FlynnOConnell/pml_utilities.git"'
    info: PipelineInfo | None = None
    axes_consumed: dict[str, str] = DEFAULT_AXES_CONSUMED

    def __init__(self, parent: Any):
        self.parent = parent

    @classmethod
    def axis_mode(cls, axis: str) -> str:
        """The ``AXIS_MODES`` entry for ``axis`` ("T", "Z" or "C"); "range" for an axis the pipeline does not declare."""
        mode = cls.axes_consumed.get(axis.upper(), "range")
        if mode not in AXIS_MODES:
            raise ValueError(
                f"{cls.__name__}.axes_consumed[{axis!r}] is {mode!r}, "
                f"expected one of {AXIS_MODES}"
            )
        return mode

    def draw(self) -> None:
        self.draw_config()

    @abstractmethod
    def draw_config(self) -> None: ...

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """True when the pipeline runs on ``arr``; the Run tab asks before instantiation, and ``arr`` is None with no data loaded."""
        return True

    extracts_traces: bool = (
        False  # the manual-ROI widget offers "Extract trace" only when set
    )

    @classmethod
    def extract_traces(cls, movie: Any, labels: Any) -> dict | None:
        """One trace per hand-drawn mask through the pipeline's own extraction step.

        ``movie`` is ``(T, Y, X)``, indexable by int or slice on T; ``labels`` is a
        ``(Y, X)`` uint16 label image with mask ``i`` at ``i + 1``. Returns
        ``{"F": (n_masks, T) float32}`` plus what the pipeline adds (``"Fneu"``
        for suite2p), or None when it cannot extract.
        """
        return None

    seeds_from_view: bool = (
        False  # the MESc tab offers "open set to what is on screen" only when set
    )

    def seed_from_view(self) -> None:
        """Set the selection to the recording on screen and the slice its sliders are on; everything else stays seeded."""

    def poll(self) -> None:
        """Runs every frame whatever tab is showing; a widget watching a worker it started does it here, never in ``draw_config``."""

    def cleanup(self) -> None:
        """Release windows, threads and file handles when the widget is destroyed."""
