"""Observable models, after fastplotlib's ``GraphicFeature`` events.

A model emits one :class:`ModelEvent` per mutation. A view subscribes to the
event types it cares about (``model.add_event_handler(fn, "rois")``, like
``graphic.add_event_handler(fn, "data")``) and redraws from the event instead
of re-deriving everything every frame. Handlers run synchronously inside the
mutation, so a view's state is consistent by the time the mutating call
returns.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["ModelEvent", "Observable"]


class ModelEvent:
    """One change: ``type`` names the feature, ``info`` says what changed,
    ``source`` is the model that emitted it."""

    __slots__ = ("type", "info", "source")

    def __init__(self, type: str, info: dict, source=None):
        self.type = str(type)
        self.info = dict(info)
        self.source = source

    def __repr__(self) -> str:
        return f"ModelEvent({self.type!r}, {self.info!r})"


class Observable:
    """Event source. Subclasses list the types they emit in ``events``."""

    events: tuple[str, ...] = ()

    def __init__(self):
        self._handlers: dict[str, list[Callable[[ModelEvent], None]]] = {}
        self._events_blocked = False

    def add_event_handler(self, handler: Callable[[ModelEvent], None], *types: str) -> None:
        """Call ``handler(event)`` on every event of ``types`` (all of them
        when none are named). Registering the same handler twice is a no-op."""
        if not callable(handler):
            raise TypeError("event handler must be callable")
        for kind in types or self.events:
            if kind not in self.events:
                raise ValueError(f"{type(self).__name__} emits {self.events}, not {kind!r}")
            handlers = self._handlers.setdefault(kind, [])
            if handler not in handlers:
                handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[ModelEvent], None], *types: str) -> None:
        """Forget ``handler`` for ``types`` (every type when none are named);
        a handler that was never registered is ignored."""
        for kind in types or tuple(self._handlers):
            handlers = self._handlers.get(kind)
            if handlers and handler in handlers:
                handlers.remove(handler)

    def clear_event_handlers(self) -> None:
        self._handlers.clear()

    def block_events(self, on: bool) -> None:
        """Silence every event while ``on``; a bulk edit unblocks and emits
        one event of its own."""
        self._events_blocked = bool(on)

    @property
    def events_blocked(self) -> bool:
        return self._events_blocked

    def _emit(self, kind: str, **info) -> ModelEvent | None:
        if self._events_blocked:
            return None
        event = ModelEvent(kind, info, self)
        for handler in list(self._handlers.get(kind, ())):
            handler(event)
        return event
