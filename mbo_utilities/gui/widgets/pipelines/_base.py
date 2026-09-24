"""
base class for pipeline widgets.

each pipeline is self-contained with its own settings dataclass and config ui.
"""

from abc import ABC, abstractmethod
from typing import Any

from mbo_utilities.gui._availability import HAS_SUITE2P
from mbo_utilities.pipeline_registry import PipelineInfo

# how a pipeline consumes each non-spatial axis: a start:stop "range", "all" of
# it (row disabled), "none" (row hidden) or "select-one" index
AXIS_MODES = ("range", "all", "none", "select-one")

# Reproduces the pre-existing behaviour for every widget that does not
# override it, so suite2p / masknmf / isoview are unaffected.
DEFAULT_AXES_CONSUMED: dict[str, str] = {"T": "range", "Z": "range", "C": "range"}


class PipelineWidget(ABC):
    """base class for pipeline widgets."""

    # human-readable name shown in pipeline selector
    name: str = "Pipeline"

    # whether this pipeline's dependencies are installed
    is_available: bool = False

    # install command to show when not available
    install_command: str = "uv pip install mbo_utilities"

    # file patterns / marker files, registered when the widget is discovered
    # through the ``mbo_utilities.pipelines`` entry-point group
    info: PipelineInfo | None = None

    # per-axis consumption mode; see AXIS_MODES
    axes_consumed: dict[str, str] = DEFAULT_AXES_CONSUMED

    def __init__(self, parent: Any):
        self.parent = parent

    @classmethod
    def axis_mode(cls, axis: str) -> str:
        """
        How this pipeline consumes ``axis`` ("T", "Z" or "C").

        Returns
        -------
        str
            One of :data:`AXIS_MODES`; "range" for an axis the pipeline
            does not declare.
        """
        mode = cls.axes_consumed.get(axis.upper(), "range")
        if mode not in AXIS_MODES:
            raise ValueError(
                f"{cls.__name__}.axes_consumed[{axis!r}] is {mode!r}, "
                f"expected one of {AXIS_MODES}"
            )
        return mode

    def draw(self) -> None:
        """Draw the pipeline widget."""
        self.draw_config()

    @abstractmethod
    def draw_config(self) -> None:
        """Draw the configuration/processing ui."""
        ...

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """True iff this pipeline can be run against ``arr``.

        Called by the Run-tab selector BEFORE instantiation, so it
        must be safe to call without spinning up the widget. Override
        for pipelines tied to a specific array type (e.g. Isoview
        consolidator only applies to ``IsoviewArray`` instances).

        ``arr`` may be ``None`` when no data is loaded.

        Default: returns ``True`` (pipeline works on any data).
        """
        return True

    #: whether :meth:`extract_traces` is implemented. The manual-ROI widget
    #: offers "Extract trace" only for pipelines that set this.
    extracts_traces: bool = False

    @classmethod
    def extract_traces(cls, movie: Any, labels: Any) -> dict | None:
        """Extract one trace per mask, using this pipeline's own extraction.

        Called with masks the user drew by hand rather than ones the
        pipeline detected, so a pipeline only needs its extraction step,
        not its detection step.

        Parameters
        ----------
        movie
            ``(T, Y, X)`` array-like; indexable by an int or a slice on the
            first axis, as ``suite2p``'s extractor expects.
        labels
            ``(Y, X)`` uint16 label image; 0 is background, mask ``i`` is
            ``i + 1``.

        Returns
        -------
        dict or None
            ``{"F": (n_masks, T) float32}`` plus whatever else the pipeline
            produces (``"Fneu"`` for suite2p). None when the pipeline cannot
            extract, which is the default.
        """
        return None

    def cleanup(self) -> None:
        """Clean up resources when widget is destroyed.

        override in subclasses to release resources like open windows,
        background threads, file handles, etc.
        """


class Suite2pState:
    """The suite2p run settings a window keeps for its Process tab, made on first use.

    Building them imports the suite2p schema, and doing that while the
    first frame paints freezes the window for seconds, so ``s2p``,
    ``s2p_db`` and ``s2p_extras`` stay None until read, and None for good
    when suite2p is not installed.
    """

    def __init__(self):
        self._s2p = None
        self._s2p_db = None
        self._s2p_extras = None
        self._s2p_savepath_flash_start = None
        self._s2p_savepath_flash_count = 0
        self._s2p_show_savepath_popup = False
        self._s2p_folder_dialog = None

    @property
    def s2p(self):
        """Suite2p processing settings (upstream schema)."""
        if self._s2p is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import Suite2pSettings
            from mbo_utilities.preferences import get_s2p_torch_device

            self._s2p = Suite2pSettings()
            # the persisted torch device, unless a dataset's settings.npy overrides it
            self._s2p.torch_device = get_s2p_torch_device()
        return self._s2p

    @s2p.setter
    def s2p(self, value):
        self._s2p = value

    @property
    def s2p_db(self):
        """Suite2p input/output db (paths, plane counts), upstream schema."""
        if self._s2p_db is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import Suite2pDB

            self._s2p_db = Suite2pDB()
        return self._s2p_db

    @s2p_db.setter
    def s2p_db(self, value):
        self._s2p_db = value

    @property
    def s2p_extras(self):
        """Mbo-only suite2p helper fields (dff_*, accept_all_cells, etc.)."""
        if self._s2p_extras is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import MboSuite2pExtras

            self._s2p_extras = MboSuite2pExtras()
        return self._s2p_extras

    @s2p_extras.setter
    def s2p_extras(self, value):
        self._s2p_extras = value
