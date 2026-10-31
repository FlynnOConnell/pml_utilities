"""Bidirectional scan-phase correction settings.

``PhaseCorrectionFeature`` holds them for one array; ``PhaseCorrectionMixin``
exposes them as array properties (``fix_phase``, ``use_fft``, ``offset``, ...).
"""

from __future__ import annotations

from enum import Enum

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent


class PhaseCorrMethod(str, Enum):
    """Phase correction methods."""

    none = "none"
    frame = "frame"  # per-frame 2D correlation
    mean = "mean"  # mean projection
    max = "max"  # max projection
    std = "std"  # std projection
    mean_sub = "mean-sub"

    @classmethod
    def from_string(cls, value: str | None) -> PhaseCorrMethod:
        """Case-insensitive lookup."""
        if value is None:
            return cls.none
        value_lower = value.lower().strip()
        if value_lower in ("", "none", "false"):
            return cls.none
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"unknown phase correction method: {value}")


class PhaseCorrectionFeature(ArrayFeature):
    """Bidirectional scan-phase correction settings for one array; toggling ``enabled`` emits an event.

    Parameters
    ----------
    enabled : bool
        whether correction is enabled
    method : PhaseCorrMethod | str
        correction method
    shift : float | None
        fixed shift value (None for auto-compute)
    use_fft : bool
        use FFT-based subpixel correction
    border : int
        border pixels to exclude from phase estimation
    max_offset : int
        maximum phase offset to search

    Examples
    --------
    >>> pc = PhaseCorrectionFeature(enabled=True, method="mean")
    >>> pc.enabled
    True
    >>> pc.method
    <PhaseCorrMethod.mean: 'mean'>
    """

    def __init__(
        self,
        enabled: bool = False,
        method: PhaseCorrMethod | str = PhaseCorrMethod.mean,
        shift: float | None = None,
        use_fft: bool = False,
        border: int = 10,
        max_offset: int = 4,
        property_name: str = "phase_correction",
    ):
        super().__init__(property_name=property_name)
        self._enabled = enabled

        if isinstance(method, str):
            self._method = PhaseCorrMethod.from_string(method)
        else:
            self._method = method

        self._shift = shift
        self._use_fft = use_fft
        self._border = border
        self._max_offset = max_offset

    @property
    def enabled(self) -> bool:
        """Whether correction is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        old_enabled = self._enabled
        self._enabled = bool(value)
        if old_enabled != self._enabled:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"enabled": self._enabled, "old_enabled": old_enabled},
            )
            self._call_event_handlers(event)

    @property
    def method(self) -> PhaseCorrMethod:
        """Correction method."""
        return self._method

    @method.setter
    def method(self, value: PhaseCorrMethod | str) -> None:
        if isinstance(value, str):
            self._method = PhaseCorrMethod.from_string(value)
        else:
            self._method = value

    @property
    def shift(self) -> float | None:
        """Fixed shift value (None for auto)."""
        return self._shift

    @shift.setter
    def shift(self, value: float | None) -> None:
        self._shift = value

    @property
    def effective_shift(self) -> float | None:
        """Fixed shift value (None means compute on read)."""
        return self._shift

    @property
    def use_fft(self) -> bool:
        """Use FFT-based subpixel correction."""
        return self._use_fft

    @use_fft.setter
    def use_fft(self, value: bool) -> None:
        self._use_fft = bool(value)

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase estimation."""
        return self._border

    @border.setter
    def border(self, value: int) -> None:
        self._border = max(0, int(value))

    @property
    def max_offset(self) -> int:
        """Maximum phase offset to search."""
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value: int) -> None:
        self._max_offset = max(1, int(value))

    def __repr__(self) -> str:
        if not self._enabled:
            return "PhaseCorrectionFeature(disabled)"
        shift_str = f"{self.effective_shift:.2f}" if self.effective_shift else "auto"
        return f"PhaseCorrectionFeature({self._method.value}, shift={shift_str})"


class PhaseCorrectionMixin:
    """Array properties over the instance's ``phase_correction`` feature.

    The host sets ``self.phase_correction`` (a ``PhaseCorrectionFeature``) in
    ``__init__`` and defines ``_invalidate_offset_cache``, which the setters of
    ``fix_phase``, ``use_fft``, ``border`` and ``max_offset`` call. An array
    supports phase correction when it has a ``phase_correction`` attribute.
    """

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        return self.phase_correction.enabled

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self.phase_correction.enabled = value
        self._invalidate_offset_cache()

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        return self.phase_correction.use_fft

    @use_fft.setter
    def use_fft(self, value: bool):
        self.phase_correction.use_fft = value
        self._invalidate_offset_cache()

    @property
    def phasecorr_method(self) -> str:
        """Current phase correction method name."""
        return self.phase_correction.method.value

    @phasecorr_method.setter
    def phasecorr_method(self, value: str | None):
        if value is None:
            self.phase_correction.enabled = False
        else:
            self.phase_correction.method = value

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase estimation."""
        return self.phase_correction.border

    @border.setter
    def border(self, value: int):
        self.phase_correction.border = value
        self._invalidate_offset_cache()

    @property
    def max_offset(self) -> int:
        """Maximum phase offset to search."""
        return self.phase_correction.max_offset

    @max_offset.setter
    def max_offset(self, value: int):
        self.phase_correction.max_offset = value
        self._invalidate_offset_cache()

    @property
    def offset(self) -> float | None:
        """The fixed phase shift, else the last one a read computed (None before any)."""
        shift = self.phase_correction.effective_shift
        if shift is not None:
            return shift
        # only ScanImageArray's reads set it
        return getattr(self, "_last_offset", None)

    @offset.setter
    def offset(self, value: float):
        self.phase_correction.shift = value
