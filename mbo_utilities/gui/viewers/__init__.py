"""Viewers that replace the standard panels for one kind of data.

The pollen calibration viewer is the one: the app's pollen panel builds it
on the viewer and draws it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["BaseViewer"]


class BaseViewer(ABC):
    """Base class for all viewer applications."""

    name: str = "Base Viewer"

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        self.image_widget = image_widget
        self.fpath = fpath
        self.parent = parent

    @property
    def data(self):
        """Access the loaded data arrays."""
        if self.image_widget is None:
            return None
        return self.image_widget.data

    @property
    def logger(self):
        """Access the logger (from parent if available)."""
        if self.parent is not None and hasattr(self.parent, "logger"):
            return self.parent.logger
        import logging

        return logging.getLogger("mbo_utilities.gui")

    @abstractmethod
    def draw(self) -> None:
        """Main render callback. Must be implemented by subclasses."""
        ...

    def on_data_loaded(self) -> None:
        """Called by _dialogs when new data is loaded. Override as needed."""

    def cleanup(self) -> None:
        """Clean up resources when the viewer closes. Override as needed."""
