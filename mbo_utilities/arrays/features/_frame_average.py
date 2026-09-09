"""
Frame averaging (temporal binning) feature for arrays.

Scan-phase correction changes pixels in place, so it lives on the reader as a
mutable ``PhaseCorrectionFeature``. Frame averaging changes ``T``, so it is a
read-time view (``FrameAveragedView``) wrapped around the reader instead. This
module gives that view the same kind of surface the phase feature has, and one
place - :func:`apply_read_features` - where every consumer that re-opens a
dataset from its path (save-as workers, pipeline tasks, ``imwrite``, the CLI)
turns a bag of option kwargs into the array they should actually read.

Classes
-------
FrameAverageFeature
    Standalone feature object holding the binning factor.

Functions
---------
apply_read_features
    Apply ``fix_phase`` / ``use_fft`` / ``frame_average`` / ... kwargs to an
    array, returning the array to read from and the kwargs left over.
"""

from __future__ import annotations

from typing import Any

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent

# kwargs that describe how a dataset is read rather than how it is written.
# ``imwrite`` and the workers pop exactly these before anything else sees the
# kwargs, so passing them alongside writer options is always safe.
PHASE_FEATURE_KEYS: tuple[str, ...] = (
    "fix_phase",
    "use_fft",
    "phasecorr_method",
    "border",
    "max_offset",
)
FRAME_AVERAGE_KEY = "frame_average"
READ_FEATURE_KEYS: tuple[str, ...] = (
    *PHASE_FEATURE_KEYS,
    "mean_subtraction",
    FRAME_AVERAGE_KEY,
)


class FrameAverageFeature(ArrayFeature):
    """
    Temporal binning settings for an array.

    Parameters
    ----------
    factor : int
        Source frames averaged into each output frame. 1 means off.
    dtype : {"source", "float32"}
        ``"source"`` rounds the mean back to the source dtype (drop-in for the
        int16 writers), ``"float32"`` keeps the fractional means.

    Examples
    --------
    >>> fa = FrameAverageFeature(factor=4)
    >>> fa.enabled
    True
    >>> fa.to_reader_kwargs()
    {'frame_average': 4}
    """

    def __init__(
        self,
        factor: int = 1,
        dtype: str = "source",
        property_name: str = FRAME_AVERAGE_KEY,
    ):
        super().__init__(property_name=property_name)
        self._factor = self._validate_factor(factor)
        self._dtype = self._validate_dtype(dtype)

    @staticmethod
    def _validate_factor(value: Any) -> int:
        try:
            factor = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"frame_average must be an integer, got {value!r}") from e
        if factor < 1:
            raise ValueError(f"frame_average must be >= 1, got {factor}")
        return factor

    @staticmethod
    def _validate_dtype(value: str) -> str:
        if value not in ("source", "float32"):
            raise ValueError(f"dtype must be 'source' or 'float32', got {value!r}")
        return value

    @property
    def value(self) -> int:
        """The binning factor (alias of :attr:`factor`)."""
        return self._factor

    @property
    def factor(self) -> int:
        """Source frames averaged into each output frame (1 = off)."""
        return self._factor

    @factor.setter
    def factor(self, value: int) -> None:
        old = self._factor
        self._factor = self._validate_factor(value)
        if old != self._factor:
            self._call_event_handlers(
                ArrayFeatureEvent(
                    type=self._property_name,
                    info={"value": self._factor, "old_value": old},
                )
            )

    @property
    def enabled(self) -> bool:
        """Whether binning does anything (factor > 1)."""
        return self._factor > 1

    @property
    def dtype(self) -> str:
        """Output dtype policy, ``"source"`` or ``"float32"``."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: str) -> None:
        self._dtype = self._validate_dtype(value)

    def apply(self, arr):
        """``arr`` binned by this factor; ``arr`` itself when disabled.

        Never stacks: an already-averaged view is re-based on its source, so
        applying a feature twice, or applying factor 1, is always safe.
        """
        from mbo_utilities.arrays._average_view import average_frames

        return average_frames(arr, self._factor, dtype=self._dtype)

    def to_reader_kwargs(self) -> dict:
        """The ``imread`` kwargs that re-create this binning from a path."""
        return {FRAME_AVERAGE_KEY: self._factor} if self.enabled else {}

    def __repr__(self) -> str:
        if not self.enabled:
            return "FrameAverageFeature(disabled)"
        return f"FrameAverageFeature(factor={self._factor}, dtype={self._dtype!r})"


def apply_read_features(arr, kwargs: dict | None = None, **extra) -> tuple[Any, dict]:
    """Apply read-time feature kwargs to ``arr``.

    Phase settings (``fix_phase``, ``use_fft``, ``phasecorr_method``,
    ``border``, ``max_offset``), ``mean_subtraction`` and ``frame_average``
    are popped from ``kwargs`` (merged with ``extra``) and applied. Phase and
    mean-subtraction keys are set as attributes when the array has them, the
    way the save-as worker always did; ``frame_average`` wraps the array in a
    ``FrameAveragedView`` (or unwraps one when the factor is 1).

    Keys whose value is ``None`` are skipped, so a caller can forward a whole
    options dict without deciding per key. Unknown keys are returned
    untouched.

    Returns
    -------
    (array, remaining_kwargs)
        The array to read from and the kwargs that were not read features.
    """
    opts = dict(kwargs or {})
    opts.update(extra)

    for key in (*PHASE_FEATURE_KEYS, "mean_subtraction"):
        if key not in opts:
            continue
        value = opts.pop(key)
        if value is not None and hasattr(arr, key):
            setattr(arr, key, value)

    if FRAME_AVERAGE_KEY in opts:
        factor = opts.pop(FRAME_AVERAGE_KEY)
        if factor is not None:
            arr = FrameAverageFeature(factor).apply(arr)

    return arr, opts
