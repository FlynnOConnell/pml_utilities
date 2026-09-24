"""A panel widget: what it draws, and whether the open data gives it anything."""

from abc import ABC, abstractmethod
from typing import Any


class Widget(ABC):
    """One panel over the open data, drawn by the app that hosts it.

    ``parent`` is what the widget reads the open data through (the app
    host's window context).
    """

    # human-readable name
    name: str = "Widget"

    def __init__(self, parent: Any):
        self.parent = parent

    @classmethod
    @abstractmethod
    def is_supported(cls, parent: Any) -> bool:
        """Whether the data ``parent`` shows gives this widget anything to draw."""

    @abstractmethod
    def draw(self) -> None:
        """Draw the widget."""

    def cleanup(self) -> None:
        """Release windows, threads and files the widget holds."""
