"""
pipeline widget registry.

pipelines are processing workflows (suite2p, masknmf, etc) that can be
run on imaging data. each pipeline has config and results views.

imports are done in a background thread to avoid blocking the GUI.
"""

import contextlib
import threading
import time
from pathlib import Path
from typing import Any

from imgui_bundle import hello_imgui, imgui, imgui_ctx

from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.lazy_array import base_array

# registry of available pipeline classes
_PIPELINE_CLASSES: list[type[PipelineWidget]] = []
_REGISTRATION_LOCK = threading.Lock()
_REGISTRATION_STARTED = False
_REGISTRATION_COMPLETE = False

# delay before the bg thread starts heavy imports. lets the main thread
# finish painting the first frame and initializing fastplotlib/qt
# without GIL contention from a multi-second suite2p import. tune via
# start_preload(delay_s=...) if needed.
_PRELOAD_DELAY_S = 1.0


def _register_pipelines_sync() -> None:
    """Register pipeline widgets (called from background thread)."""
    global _PIPELINE_CLASSES, _REGISTRATION_COMPLETE

    with _REGISTRATION_LOCK:
        if _PIPELINE_CLASSES:
            _REGISTRATION_COMPLETE = True
            return

        # preload settings module first (it's imported by Suite2pPipelineWidget.__init__).
        # _s2p_schema is now lazy, so this import is cheap — it does NOT
        # transitively pull in suite2p. the actual suite2p load happens
        # in the warm_up step below.
        try:
            from mbo_utilities.gui.widgets.pipelines import settings as _  # noqa: F401
        except Exception:
            pass

        # import pipeline widgets - they register themselves based on availability
        try:
            from mbo_utilities.gui.widgets.pipelines.suite2p import (
                Suite2pPipelineWidget,
            )

            _PIPELINE_CLASSES.append(Suite2pPipelineWidget)
        except Exception:
            pass

        try:
            from mbo_utilities.gui.widgets.pipelines.isoview import (
                IsoviewPipelineWidget,
            )

            _PIPELINE_CLASSES.append(IsoviewPipelineWidget)
        except Exception:
            pass

        try:
            from mbo_utilities.gui.widgets.pipelines.masknmf import (
                MaskNMFPipelineWidget,
            )

            _PIPELINE_CLASSES.append(MaskNMFPipelineWidget)
        except Exception:
            pass

        try:
            from mbo_utilities.gui.widgets.pipelines.voltage import (
                VoltagePipelineWidget,
            )

            _PIPELINE_CLASSES.append(VoltagePipelineWidget)
        except Exception:
            pass

        try:
            from mbo_utilities.gui.widgets.pipelines.rois import RoiPipelineWidget

            _PIPELINE_CLASSES.append(RoiPipelineWidget)
        except Exception:
            pass

        # third-party pipelines from the "mbo_utilities.pipelines"
        # entry-point group. Loaded last so a plugin never shadows a
        # built-in in the selector order.
        try:
            from mbo_utilities.pipeline_registry import load_entry_point_pipelines

            for cls in load_entry_point_pipelines():
                if issubclass(cls, PipelineWidget) and cls not in _PIPELINE_CLASSES:
                    _PIPELINE_CLASSES.append(cls)
        except Exception:
            pass

        _REGISTRATION_COMPLETE = True

    # suite2p.parameters.SETTINGS is loaded out-of-process by
    # `_s2p_schema`'s module-level daemon (cached to
    # `~/.mbo/cache/s2p_settings_<version>.json`). Importing the widget
    # classes above triggers that daemon transitively; the subprocess
    # never enters this interpreter's address space.


def _delayed_preload(delay_s: float) -> None:
    """Sleep, then run the heavy preload. Runs in a daemon thread."""
    if delay_s > 0:
        time.sleep(delay_s)
    _register_pipelines_sync()


def start_preload(delay_s: float | None = None) -> None:
    """Start background preloading of pipeline widgets + suite2p schema.

    Call this early (e.g., on GUI startup) to warm up imports
    before the user clicks the Run tab. The bg thread sleeps briefly
    before doing heavy work so it doesn't contend with the main
    thread during first paint / fastplotlib init.
    """
    global _REGISTRATION_STARTED

    if _REGISTRATION_STARTED:
        return

    _REGISTRATION_STARTED = True
    delay = _PRELOAD_DELAY_S if delay_s is None else delay_s
    thread = threading.Thread(target=_delayed_preload, args=(delay,), daemon=True)
    thread.start()


def is_ready() -> bool:
    """Check if pipeline registration is complete."""
    return _REGISTRATION_COMPLETE


def _register_pipelines() -> None:
    """Register available pipeline widgets (blocking if not preloaded)."""
    if not _REGISTRATION_STARTED:
        start_preload()

    # if already complete, return immediately
    if _REGISTRATION_COMPLETE:
        return

    # wait for background thread to complete (blocking)
    _register_pipelines_sync()


def get_available_pipelines() -> list[type[PipelineWidget]]:
    """Get list of all registered pipeline classes."""
    _register_pipelines()
    return _PIPELINE_CLASSES.copy()


def get_pipeline_names() -> list[str]:
    """Get names of all registered pipelines."""
    _register_pipelines()
    return [p.name for p in _PIPELINE_CLASSES]


def any_pipeline_available() -> bool:
    """Check if any pipeline is available (installed)."""
    _register_pipelines()
    return any(p.is_available for p in _PIPELINE_CLASSES)


def get_trace_extractors() -> list[type[PipelineWidget]]:
    """Installed pipelines that can extract traces from supplied masks.

    See :meth:`PipelineWidget.extract_traces`; used by the manual-ROI
    widget's "Extract trace" action.
    """
    _register_pipelines()
    return [
        p for p in _PIPELINE_CLASSES if p.extracts_traces and _is_pipeline_available(p)
    ]


def _active_array(parent: Any) -> Any:
    """Return the currently-loaded array (or ``None``).

    Used to filter pipelines via :meth:`PipelineWidget.applies_to`.
    """
    iw = getattr(parent, "image_widget", None)
    if iw is None or not iw.data:
        return None
    return iw.data[0]


def shown_name(parent: Any) -> str:
    """What the array on screen is called on a button: its recording
    (``MUnit_3`` of a .mesc), else its file's name, else "".
    """
    arr = _active_array(parent)
    key = getattr(base_array(arr), "unit_key", None) if arr is not None else None
    if key:
        return str(key).rsplit("/", 1)[-1]
    fpath = getattr(parent, "fpath", None)
    if isinstance(fpath, (list, tuple)):
        fpath = fpath[0] if fpath else None
    return Path(str(fpath)).name if fpath else ""


def _is_pipeline_available(cls: type) -> bool:
    """Resolve ``is_available`` whether it's a class attr or a property.

    Suite2p declares ``is_available`` as a ``@property`` (instance-bound),
    so reading it off the class returns the descriptor (truthy) and not
    the value. We instantiate temporarily if needed — Suite2p widgets
    are heavy to construct so we cache the result on the class.
    """
    cached = getattr(cls, "_is_available_cached", None)
    if cached is not None:
        return cached
    val = cls.__dict__.get("is_available")
    if isinstance(val, property):
        try:
            result = bool(val.fget(cls.__new__(cls)))
        except Exception:
            # property reads parent state — give up and assume available
            result = True
    else:
        result = bool(getattr(cls, "is_available", True))
    cls._is_available_cached = result  # type: ignore[attr-defined]
    return result


def _applies_by_class(parent: Any) -> dict[type[PipelineWidget], bool]:
    """``applies_to`` of every registered class for the array on screen,
    evaluated once per array: it can open the source file.
    """
    arr = _active_array(parent)
    cache = getattr(parent, "_pipeline_applies_cache", None)
    if cache is None or cache[0] is not arr:
        applies_by_cls = {}
        for cls in _PIPELINE_CLASSES:
            try:
                applies_by_cls[cls] = bool(cls.applies_to(arr))
            except Exception:
                applies_by_cls[cls] = False
        cache = (arr, applies_by_cls)
        parent._pipeline_applies_cache = cache
    return cache[1]


def quick_pipelines(parent: Any) -> list[type[PipelineWidget]]:
    """Installed pipelines that apply to the array on screen and seed
    themselves from the view (``seeds_from_view``). Empty until the
    background registration is done, so a tab drawing them never blocks.
    """
    if not _REGISTRATION_COMPLETE:
        return []
    applies = _applies_by_class(parent)
    return [
        cls
        for cls in _PIPELINE_CLASSES
        if cls.seeds_from_view
        and applies.get(cls, False)
        and _is_pipeline_available(cls)
    ]


def pipeline_instance(parent: Any, cls: type[PipelineWidget]) -> PipelineWidget:
    """The one widget of ``cls`` for this host, built on first use."""
    if not hasattr(parent, "_pipeline_instances"):
        parent._pipeline_instances = {}
    if cls.name not in parent._pipeline_instances:
        parent._pipeline_instances[cls.name] = cls(parent)
    return parent._pipeline_instances[cls.name]


def open_pipeline(
    parent: Any, name: str, where: str = "window", seed: bool = False
) -> PipelineWidget | None:
    """Show the pipeline called ``name``: in a floating window
    (``where="window"``, drawn by :func:`draw_pipeline_windows`) or in the
    Process tab (``"tab"``, selecting it there). It is the same widget either
    way. ``seed`` first sets its selection to what the viewer shows
    (:meth:`PipelineWidget.seed_from_view`). None when no registered
    pipeline has that name.
    """
    _register_pipelines()
    cls = next((c for c in _PIPELINE_CLASSES if c.name == name), None)
    if cls is None:
        return None
    pipeline = pipeline_instance(parent, cls)
    if seed:
        pipeline.seed_from_view()
    if where == "tab":
        parent._selected_pipeline_name = name
        parent._force_run_tab = True
        return pipeline
    if not hasattr(parent, "_pipeline_windows"):
        parent._pipeline_windows = []
    if name not in parent._pipeline_windows:
        parent._pipeline_windows.append(name)
    parent._pipeline_window_focus = name
    return pipeline


def draw_pipeline_windows(parent: Any) -> None:
    """Draw every pipeline opened as a floating window (:func:`open_pipeline`):
    the widget the Process tab draws, under an imgui id of its own so both can
    show at once. Runs from the top strip's frame hook, so the windows stay up
    whatever tab is selected; closing one takes it off the list.
    """
    names = list(getattr(parent, "_pipeline_windows", None) or [])
    if not names:
        return
    focus = getattr(parent, "_pipeline_window_focus", None)
    parent._pipeline_window_focus = None
    viewport = imgui.get_main_viewport()
    for name in names:
        pipeline = parent._pipeline_instances.get(name)
        if pipeline is None:
            parent._pipeline_windows.remove(name)
            continue
        # a fixed first size: the widget sizes its columns from the window,
        # so an auto-resizing window would never settle
        imgui.set_next_window_size(
            imgui.ImVec2(
                min(hello_imgui.em_size(52), viewport.size.x * 0.9),
                min(hello_imgui.em_size(44), viewport.size.y * 0.9),
            ),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_pos(
            viewport.get_center(), imgui.Cond_.first_use_ever, imgui.ImVec2(0.5, 0.5)
        )
        if name == focus:
            imgui.set_next_window_focus()
        expanded, keep = imgui.begin(f"{name}###pipeline_window_{name}", True)
        if expanded:
            with imgui_ctx.push_id("window"):
                try:
                    pipeline.draw()
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")
        imgui.end()
        if not keep:
            parent._pipeline_windows.remove(name)


def draw_run_tab(parent: Any) -> None:
    """Draw the run tab content.

    Renders a pipeline selector at the top (when more than one
    applicable pipeline is available), then the selected widget's
    config UI. Pipelines are filtered by ``is_available`` (deps
    installed) AND ``applies_to(active_array)`` (data type matches).
    """
    _register_pipelines()

    # Persist selection by pipeline NAME, not list index — the list of
    # applicable pipelines changes between datasets, and an int index
    # silently shifts to a different pipeline when the list shrinks.
    if not hasattr(parent, "_selected_pipeline_name"):
        parent._selected_pipeline_name = None
    if not hasattr(parent, "_pipeline_instances"):
        parent._pipeline_instances = {}

    applies_by_cls = _applies_by_class(parent)

    if not _PIPELINE_CLASSES:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            "No pipelines registered.",
        )
        imgui.text("Install a pipeline package:")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            "uv pip install mbo_utilities",
        )
        return

    # partition: applicable to current data (and installed) vs. not.
    applicable: list[type[PipelineWidget]] = []
    not_applicable: list[type[PipelineWidget]] = []
    not_installed: list[type[PipelineWidget]] = []
    for cls in _PIPELINE_CLASSES:
        installed = _is_pipeline_available(cls)
        applies = applies_by_cls.get(cls, False)
        if installed and applies:
            applicable.append(cls)
        elif not installed:
            not_installed.append(cls)
        else:
            not_applicable.append(cls)

    # Isoview before Suite2p in the selector when both apply (Suite2p
    # applies to any array, so it would otherwise lead by registration
    # order). Stable: only Isoview is hoisted; the rest keep their order.
    applicable.sort(key=lambda c: 0 if c.name == "Isoview" else 1)

    # selector lists EVERY registered pipeline — runnable ones first,
    # then installed-but-not-applicable, then not-installed. Selecting a
    # non-runnable entry explains why instead of drawing a config UI.
    entries: list[tuple[type[PipelineWidget], str, str]] = []
    for cls in applicable:
        entries.append((cls, cls.name, "ok"))
    for cls in not_applicable:
        entries.append((cls, f"{cls.name} (not applicable)", "na"))
    for cls in not_installed:
        entries.append((cls, f"{cls.name} (not installed)", "missing"))

    if not entries:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            "No pipelines registered.",
        )
        return

    # Resolve persisted name → index each frame so a user's choice
    # survives switching between datasets where the runnable set changes.
    idx = 0
    for i, (cls, _label, _state) in enumerate(entries):
        if cls.name == parent._selected_pipeline_name:
            idx = i
            break
    if len(entries) > 1:
        labels = [e[1] for e in entries]
        imgui.set_next_item_width(220)
        changed, new_idx = imgui.combo("Pipeline##run_tab", idx, labels)
        if changed:
            idx = new_idx
    pipeline_cls, _label, state = entries[idx]
    parent._selected_pipeline_name = pipeline_cls.name
    if state == "ok":
        if len(entries) > 1:
            imgui.same_line()
        if imgui.small_button("Pop out##run_tab_window"):
            open_pipeline(parent, pipeline_cls.name, "window")
        imgui.set_item_tooltip(
            "Open this pipeline in a floating window: the same configuration, "
            "kept up while you use the other tabs."
        )
    if len(entries) > 1 or state == "ok":
        imgui.separator()

    if state == "missing":
        imgui.text(f"{pipeline_cls.name} is not installed.")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            pipeline_cls.install_command,
        )
        return
    if state == "na":
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            f"{pipeline_cls.name} does not apply to the loaded data.",
        )
        return

    pipeline = pipeline_instance(parent, pipeline_cls)

    try:
        pipeline.draw()
    except Exception as e:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
            f"Error: {e}",
        )


def cleanup_pipelines(parent: Any) -> None:
    """Clean up all pipeline instances when gui is closing.

    calls cleanup() on each pipeline to release resources like
    open windows, background threads, etc.
    """
    if not hasattr(parent, "_pipeline_instances"):
        return

    for pipeline in parent._pipeline_instances.values():
        with contextlib.suppress(Exception):
            pipeline.cleanup()

    parent._pipeline_instances.clear()


# lazy imports for settings - use __getattr__ for module-level lazy loading
_settings_cache = {}


def __getattr__(name: str):
    """Lazy load settings module exports on first access."""
    lazy_names = (
        "Suite2pSettings",
        "Suite2pDB",
        "MboSuite2pExtras",
        "draw_suite2p_settings_panel",
        "draw_section_suite2p",
    )
    if name in lazy_names:
        if name not in _settings_cache:
            from mbo_utilities.gui.widgets.pipelines import settings

            for attr in lazy_names:
                _settings_cache[attr] = getattr(settings, attr)
        return _settings_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineWidget",
    "Suite2pSettings",
    "Suite2pDB",
    "MboSuite2pExtras",
    "any_pipeline_available",
    "cleanup_pipelines",
    "draw_pipeline_windows",
    "draw_run_tab",
    "draw_section_suite2p",
    "draw_suite2p_settings_panel",
    "get_available_pipelines",
    "get_pipeline_names",
    "get_trace_extractors",
    "is_ready",
    "open_pipeline",
    "pipeline_instance",
    "quick_pipelines",
    "shown_name",
    "start_preload",
]
