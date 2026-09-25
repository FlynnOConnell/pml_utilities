"""
pipeline widget registry.

pipelines are processing workflows (suite2p, masknmf, etc) that can be
run on imaging data. each pipeline has config and results views.

imports are done in a background thread to avoid blocking the GUI.
"""

import threading
import time

from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget

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
        p for p in _PIPELINE_CLASSES if p.extracts_traces and pipeline_installed(p)
    ]


def pipeline_installed(cls: type) -> bool:
    """Whether ``cls``'s dependencies are installed: its ``is_available``,
    a class attr or a property.

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
    "draw_section_suite2p",
    "draw_suite2p_settings_panel",
    "get_available_pipelines",
    "get_pipeline_names",
    "get_trace_extractors",
    "is_ready",
    "pipeline_installed",
    "start_preload",
]
