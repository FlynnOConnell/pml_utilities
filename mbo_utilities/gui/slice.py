"""The slice on screen: where the viewer's sliders sit, shared by every view.

One :class:`Slice` per host, fed by one handler on the viewer's indices and
read by anything that depends on the channel or z-plane shown (the ROI
widget's overlays and tables, Signal Quality's active plane, the summary
images), the way :class:`~mbo_utilities.gui.playhead.Playhead` is the one
time on screen. A view subscribes to it; it never reads the sliders itself.
"""

from __future__ import annotations

from mbo_utilities.annotation.events import Observable
from mbo_utilities.arrays.features._dim_labels import slider_roles

__all__ = ["Slice", "viewer_positions", "viewer_roles"]


class Slice(Observable):
    """Where the viewer's sliders sit.

    ``positions`` is every slider's 0-based index by the viewer's dim name, T
    included; ``roles`` says which array axis each slider scrolls (``t``,
    ``c``, ``z``, else ``dimN``: :func:`viewer_roles`), and ``t``, ``c`` and
    ``z`` are the indices by role. :meth:`move` emits ``slice`` when anything
    but T changed; T is the playhead's.
    """

    events = ("slice",)

    def __init__(self):
        super().__init__()
        self.positions: dict[str, int] = {}
        self.roles: dict[str, str] = {}

    def move(self, positions, roles, source=None) -> bool:
        """Take the sliders' new positions; True when a non-T one changed.

        ``positions`` maps each dim name to its 0-based index, ``roles`` each
        dim name to its axis, and ``source`` is who moved them, so a view can
        ignore its own moves.
        """
        positions = {str(k): int(v) for k, v in positions.items()}
        roles = {str(k): str(v) for k, v in roles.items()}
        previous = {
            k: v for k, v in self.positions.items() if self.roles.get(k) != "t"
        }
        now = {k: v for k, v in positions.items() if roles.get(k) != "t"}
        self.positions, self.roles = positions, roles
        if now == previous:
            return False
        self._emit("slice", positions=dict(positions), previous=previous, source=source)
        return True

    def of(self, which: str) -> int | None:
        """The index of slider ``which``: a role (``t``, ``c``, ``z``), else a
        dim name; None when there is no such slider.
        """
        for name, role in self.roles.items():
            if role == which:
                return self.positions.get(name)
        return self.positions.get(which)

    @property
    def t(self) -> int:
        return self.of("t") or 0

    @property
    def c(self) -> int:
        return self.of("c") or 0

    @property
    def z(self) -> int:
        return self.of("z") or 0


def viewer_roles(iw) -> dict[str, str]:
    """Which array axis each of a viewer's sliders scrolls, by dim name.

    The viewer drops size-1 T, C and Z axes before it makes sliders, and the
    wrapper it does that with records the axes it kept; those say which
    slider is which whatever the sliders are labelled (``ROI`` on an AOD
    unit, ``Cam`` on IsoView). A viewer over a plain array has no such record
    and its sliders are read by position: T, C, Z in order (AGENTS.md §7.6).
    """
    names = tuple(iw.dim_names)
    data = getattr(iw, "data", None)
    arr = data[0] if data else None
    while hasattr(arr, "_wrapped"):
        arr = arr._wrapped
    kept = getattr(arr, "_kept", None)
    if kept is not None:
        axes = [a for a in kept if a < 3]
        if len(axes) == len(names):
            return {name: "tcz"[a] for name, a in zip(names, axes, strict=True)}
    return slider_roles(names)


def viewer_positions(iw) -> tuple[dict[str, int], dict[str, str]]:
    """``(positions, roles)`` of a viewer's sliders, what :meth:`Slice.move` takes."""
    names = tuple(iw.dim_names)
    return {name: int(iw.indices[name]) for name in names}, viewer_roles(iw)
