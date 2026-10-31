"""Sets up Qt, the wgpu backend and the imgui assets for the GUI; importing the module runs it once."""

import importlib.util
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

_initialized = False


def get_package_assets_path() -> Path:
    """Return path to the bundled assets folder in the installed package."""
    return Path(str(resources.files("mbo_utilities").joinpath("assets")))


def _copy_assets():
    """Copy package assets to user config directory."""
    import imgui_bundle

    from mbo_utilities.preferences import get_mbo_dirs

    package_assets = get_package_assets_path()
    user_assets = Path(get_mbo_dirs()["base"]) / "imgui" / "assets"

    user_assets.mkdir(parents=True, exist_ok=True)
    if package_assets.is_dir():
        shutil.copytree(package_assets, user_assets, dirs_exist_ok=True)

    # imgui_bundle's fonts fill in whatever the package assets lack
    fonts_dst = user_assets / "fonts"
    fonts_dst.mkdir(parents=True, exist_ok=True)
    (user_assets / "static").mkdir(parents=True, exist_ok=True)

    fonts_src = Path(imgui_bundle.__file__).parent / "assets" / "fonts"
    for p in fonts_src.rglob("*"):
        if p.is_file():
            d = fonts_dst / p.relative_to(fonts_src)
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)

    # markdown rendering loads these by name; a missing one gets any roboto face
    roboto_dir = fonts_dst / "Roboto"
    roboto_dir.mkdir(parents=True, exist_ok=True)
    required = [
        roboto_dir / "Roboto-Regular.ttf",
        roboto_dir / "Roboto-Bold.ttf",
        roboto_dir / "Roboto-RegularItalic.ttf",
        fonts_dst / "fontawesome-webfont.ttf",
    ]
    fallback = next(roboto_dir.glob("*.ttf"), None)
    for need in required:
        if not need.exists() and fallback is not None:
            shutil.copy2(fallback, need)

    return user_assets


def get_default_ini_path(name: str = "imgui_settings") -> str:
    """Returns ``~/.mbo/imgui/assets/app_settings/<name>.ini``, creating the folder.

    Pass it as ``RunnerParams.ini_filename`` before ``immapp.run()``.
    """
    from mbo_utilities.preferences import get_mbo_dirs

    settings_dir = Path(get_mbo_dirs()["base"]) / "imgui" / "assets" / "app_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return str(settings_dir / f"{name}.ini")


def _configure_qt_backend():
    """Selects the Qt backend for rendercanvas when PyQt6 is installed.

    Must run before fastplotlib is imported, or rendercanvas picks glfw.
    """
    if os.environ.get("RENDERCANVAS_FORCE_OFFSCREEN"):
        # offscreen/http canvases never touch Qt; importing QtWidgets here
        # would need libGL even on a headless box that doesn't have it
        return
    if importlib.util.find_spec("PyQt6") is None:
        return
    if os.environ.get("RENDERCANVAS_BACKEND", "qt").lower() not in ("qt", "pyqt6"):
        # the user chose another backend (glfw on a box without qt's xcb libs)
        return

    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")

    from PyQt6.QtWidgets import QSlider

    # suite2p's GUI uses the PyQt5 spelling
    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks


def _configure_wgpu_backend():
    """Configure wgpu instance to skip opengl backend and avoid egl warnings."""
    if sys.platform == "emscripten":
        return

    from wgpu.backends.wgpu_native.extras import set_instance_extras

    try:
        if sys.platform == "win32":
            set_instance_extras(backends=["Vulkan", "DX12"])
        elif sys.platform == "darwin":
            set_instance_extras(backends=["Metal"])
        else:
            set_instance_extras(backends=["Vulkan"])
    except RuntimeError:
        # the wgpu instance already exists, so its backends are fixed
        pass


def setup_imgui():
    """Configures the Qt and wgpu backends and the imgui assets folder, once per process."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    _configure_qt_backend()
    _configure_wgpu_backend()
    user_assets = _copy_assets()

    from imgui_bundle import hello_imgui

    hello_imgui.set_assets_folder(str(user_assets))


setup_imgui()
