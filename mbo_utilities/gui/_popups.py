"""
Popup windows and dialogs.

This module contains popup windows for tools, scope inspector,
metadata viewer, and process console.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from imgui_bundle import ImVec2, imgui

from mbo_utilities.gui._options_popup import (
    _ensure_gpu_list,
    apply_compute_gpu,
    compute_gpu_current_index,
    compute_gpu_options,
)
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.preferences import (
    get_gpu_index,
    get_mem_monitor_interval,
    get_mem_warn_pct,
    set_gpu_index,
)

_SYS_TITLE = imgui.ImVec4(0.5, 0.8, 1.0, 1.0)
_SYS_LABEL = imgui.ImVec4(0.7, 0.7, 0.72, 1.0)
_SYS_ACCENT = imgui.ImVec4(0.95, 0.85, 0.45, 1.0)

_LOG_INFO = imgui.ImVec4(0.86, 0.86, 0.86, 1.0)
_LOG_ORANGE = imgui.ImVec4(1.0, 0.6, 0.2, 1.0)
_LOG_RED = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
_LOG_LEVEL_COLORS = {
    "DEBUG": _LOG_ORANGE,
    "INFO": _LOG_INFO,
    "WARNING": _LOG_ORANGE,
    "ERROR": _LOG_RED,
    "CRITICAL": _LOG_RED,
}

_MIN_LOG_BOX_H = 80.0


def _log_line_style(line: str) -> tuple[Any, str]:
    """Color and display text for a worker log line.

    Worker logs are ``asctime | name | levelname | message``. Color by
    level and drop the level field; lines without a level render white.
    """
    parts = line.split(" | ", 3)
    if len(parts) == 4:
        color = _LOG_LEVEL_COLORS.get(parts[2].strip())
        if color is not None:
            return color, f"{parts[0]} | {parts[1].strip()} | {parts[3]}"
    return _LOG_INFO, line


def _meter_color(frac: float) -> Any:
    """Green / amber / red fill for a usage meter by fill fraction."""
    if frac >= 0.85:
        return imgui.ImVec4(0.85, 0.30, 0.30, 1.0)
    if frac >= 0.60:
        return imgui.ImVec4(0.90, 0.70, 0.25, 1.0)
    return imgui.ImVec4(0.32, 0.66, 0.38, 1.0)


def _draw_meter(label: str, frac: float, pct: str, caption_s: str = "") -> None:
    """A slim usage bar: dim label, then bar + percentage + optional caption.

    The bar is a fixed half of the panel width so every meter (CPU and each
    GPU) lines up at the same width regardless of caption length.
    """
    imgui.text_colored(_SYS_LABEL, label)
    frac = min(max(frac, 0.0), 1.0)
    bar_w = max(80.0, imgui.get_content_region_avail().x * 0.5)
    bar_h = imgui.get_text_line_height() * 0.7
    imgui.push_style_color(imgui.Col_.plot_histogram, _meter_color(frac))
    imgui.progress_bar(frac, ImVec2(bar_w, bar_h), "")
    imgui.pop_style_color()
    imgui.same_line()
    imgui.text(pct)
    if caption_s:
        imgui.same_line()
        imgui.text_disabled(caption_s)


def _draw_gpu_meter(d: dict) -> None:
    """One physical GPU as a utilization meter (VRAM fill when util is N/A)."""
    total = d.get("total_mb") or 0
    used = d.get("used_mb") or 0
    util = d.get("util_pct")
    temp = d.get("temp_c")

    mem_frac = (used / total) if total else 0.0
    if util is not None:
        frac, pct = util / 100.0, f"{util:.0f}%"
    else:
        frac, pct = mem_frac, f"{mem_frac * 100:.0f}%"

    caption = []
    if total:
        caption.append(f"{used / 1024:.1f}/{total / 1024:.1f} GB")
    if temp is not None:
        caption.append(f"{temp:.0f}C")
    dm = d.get("driver_model")
    if dm:
        dm_pending = d.get("driver_model_pending")
        # pending differs -> driver-model switch staged until reboot
        caption.append(f"{dm}→{dm_pending}" if dm_pending and dm_pending != dm else dm)
    caption_s = ("  " + " · ".join(caption)) if caption else ""

    _draw_meter(
        f"GPU {d.get('index', '?')}: {d.get('name', '?')}", frac, pct, caption_s
    )


def _cpu_temp() -> float | None:
    """Best-effort CPU package temperature in C, or None.

    psutil.sensors_temperatures is Linux/FreeBSD only; on Windows it is
    absent and this returns None, so the meter simply omits the temp like
    a GPU that doesn't report one.
    """
    try:
        import psutil

        fn = getattr(psutil, "sensors_temperatures", None)
        if fn is None:
            return None
        temps = fn() or {}
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            entries = temps.get(key)
            if entries:
                return float(entries[0].current)
        for entries in temps.values():
            if entries:
                return float(entries[0].current)
    except Exception:
        return None
    return None


def _draw_cpu_meter(cpu_pct: float) -> None:
    """Total CPU as a usage meter, formatted like the GPU meters (temperature
    in the caption). RAM has its own meter below.
    """
    temp = _cpu_temp()
    caption_s = f"  {temp:.0f}C" if temp is not None else ""
    _draw_meter("CPU", cpu_pct / 100.0, f"{cpu_pct:.0f}%", caption_s)


def _draw_ram_meter(snap: dict, warn_pct: float) -> None:
    """System RAM as a usage meter, with this process tree's share.

    The bar is the Task-Manager headline (all processes); the caption adds
    used/total GB and what the gui plus its running jobs account for, which
    is the number that matters when a pipeline is the thing eating the box.
    """
    pct = snap.get("sys_pct", 0.0)
    caption = [f"{snap['used_gb']:.1f}/{snap['total_gb']:.1f} GB"]
    proc_gb = snap.get("proc_gb")
    if proc_gb is not None:
        caption.append(f"mbo {proc_gb:.1f} GB/{snap.get('nproc', 0)} proc")
    _draw_meter("RAM", pct / 100.0, f"{pct:.0f}%", "  " + " · ".join(caption))
    if warn_pct and pct >= warn_pct:
        imgui.same_line()
        imgui.text_colored(
            imgui.ImVec4(0.95, 0.45, 0.45, 1.0), f"! over {warn_pct:.0f}%"
        )


def _draw_gpu_selectors(parent: Any, devices: list, show_render: bool) -> None:
    """Inline GPU pickers for multi-GPU systems.

    Render adapter change applies on next launch (wgpu can't hot-swap).
    Compute device is applied live via CUDA_VISIBLE_DEVICES so newly
    started jobs use it, and persisted for future launches.
    """
    labels = getattr(parent, "_options_gpu_labels", []) or []
    multi_render = show_render and len(labels) > 1
    multi_compute = len(devices) > 1
    if not (multi_render or multi_compute):
        return

    imgui.spacing()
    bar_w = max(80.0, imgui.get_content_region_avail().x * 0.5)

    if multi_render:
        if not hasattr(parent, "_options_gpu_idx"):
            parent._options_gpu_idx = get_gpu_index()
        imgui.text_colored(_SYS_LABEL, "Render GPU")
        imgui.set_next_item_width(bar_w)
        ui_idx = parent._options_gpu_idx + 1  # 0 == auto
        changed, new_ui = imgui.combo("##sys_render_gpu", ui_idx, labels)
        if changed:
            parent._options_gpu_idx = new_ui - 1
            set_gpu_index(parent._options_gpu_idx)
        imgui.same_line()
        imgui.text_disabled("next launch")

    if multi_compute:
        values, labels = compute_gpu_options(devices)
        sel = compute_gpu_current_index(values)
        imgui.text_colored(_SYS_LABEL, "Compute GPU")
        imgui.set_next_item_width(bar_w)
        changed, new_sel = imgui.combo("##sys_compute_gpu", sel, labels)
        if changed and 0 <= new_sel < len(values):
            apply_compute_gpu(values[new_sel])
        imgui.same_line()
        imgui.text_disabled("new jobs")


def _draw_system_info_header(parent: Any) -> None:
    """Compact system-capacity header for the Process Console.

    Shows: CPU cores, a CPU usage meter, a RAM meter (system use plus this
    process tree's share), one live-usage meter per physical GPU, and — on
    multi-GPU systems — render/compute GPU pickers. Per-GPU usage and CPU%
    come from a ~1.5s throttle (nvidia-smi is a subprocess); RAM follows the
    memory-monitor tick rate.
    """
    if not imgui.collapsing_header("System", imgui.TreeNodeFlags_.default_open):
        return

    _ensure_gpu_list(parent)

    # per-GPU usage (nvidia-smi) + total CPU%, throttled. nvidia-smi is a
    # subprocess; cpu_percent(interval=None) needs a gap between calls to be
    # meaningful — both want the ~1.5s window, not per-frame.
    now = time.monotonic()
    if (
        not hasattr(parent, "_sys_gpu_devices")
        or now - getattr(parent, "_sys_gpu_last_refresh", 0.0) >= 1.5
    ):
        from mbo_utilities.gpu import gpu_devices

        try:
            parent._sys_gpu_devices = gpu_devices()
        except Exception:
            parent._sys_gpu_devices = []
        try:
            import psutil

            parent._sys_cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            parent._sys_cpu_pct = None
        parent._sys_gpu_last_refresh = now
    gpu_devices_live = parent._sys_gpu_devices
    cpu_pct = getattr(parent, "_sys_cpu_pct", None)

    # RAM on its own throttle: a snapshot costs a children walk (~8 ms), so it
    # follows the user's tick rate rather than the frame rate or the slower
    # nvidia-smi window.
    tick = max(0.25, min(get_mem_monitor_interval(), 2.0))
    if (
        not hasattr(parent, "_sys_mem")
        or now - getattr(parent, "_sys_mem_last_refresh", 0.0) >= tick
    ):
        from mbo_utilities._sysmem import mem_snapshot

        try:
            parent._sys_mem = mem_snapshot()
        except Exception:
            parent._sys_mem = None
        parent._sys_mem_last_refresh = now
    mem_snap = parent._sys_mem

    # CPU core counts via psutil (static). Live RAM moves to the CPU meter.
    try:
        import psutil

        cpu_phys = psutil.cpu_count(logical=False)
        cpu_log = psutil.cpu_count(logical=True)
        cpu_str = f"{cpu_phys}p / {cpu_log}t" if cpu_phys else f"{cpu_log or '?'}"
    except ImportError:
        import os as _os

        cpu_str = f"{_os.cpu_count() or '?'}"

    adapters = getattr(parent, "_options_gpu_adapters", []) or []

    # Selected adapter resolved from persisted preferences. -1 == auto.
    sel_idx = get_gpu_index()
    if 0 <= sel_idx < len(adapters):
        info_d = getattr(adapters[sel_idx], "info", {}) or {}
        sel_name = info_d.get("device", "?")
        sel_backend = info_d.get("backend_type", "")
        selected_str = f"{sel_name} [{sel_backend}]" if sel_backend else sel_name
    else:
        selected_str = "auto (wgpu picks)"

    # Deduplicate physical GPUs by device name; skip software/CPU adapters
    # so the list shows only real hardware the user can render with.
    seen: set[str] = set()
    gpu_names: list[str] = []
    for a in adapters:
        info_d = getattr(a, "info", {}) or {}
        name = info_d.get("device", "?")
        adapter_type = info_d.get("adapter_type", "?")
        if adapter_type in ("CPU", "Unknown"):
            continue
        if name not in seen:
            seen.add(name)
            gpu_names.append(name)
    gpus_str = ", ".join(gpu_names) if gpu_names else "none detected"

    imgui.spacing()

    if imgui.begin_table("##sysinfo_table", 2, imgui.TableFlags_.sizing_fixed_fit):
        imgui.table_setup_column("k", imgui.TableColumnFlags_.width_fixed, 140)
        imgui.table_setup_column("v", imgui.TableColumnFlags_.width_stretch)

        def _row(k: str, v: str, value_color: Any = None) -> None:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_colored(_SYS_LABEL, k)
            imgui.table_next_column()
            if value_color is not None:
                imgui.text_colored(value_color, v)
            else:
                imgui.text(v)

        _row("CPU cores:", cpu_str)
        # render adapter shown as text only when there's nothing to pick;
        # the multi-GPU picker below replaces it otherwise.
        if len(gpu_names) <= 1:
            _row("GPU (selected):", selected_str, value_color=_SYS_ACCENT)
        if not gpu_devices_live:
            _row("GPU (available):", gpus_str)

        imgui.end_table()

    if cpu_pct is not None or mem_snap or gpu_devices_live:
        imgui.spacing()
        if cpu_pct is not None:
            _draw_cpu_meter(cpu_pct)
        if mem_snap:
            _draw_ram_meter(mem_snap, get_mem_warn_pct())
        for d in gpu_devices_live:
            _draw_gpu_meter(d)

    _draw_gpu_selectors(parent, gpu_devices_live, len(gpu_names) > 1)

    imgui.spacing()
    imgui.spacing()


def draw_console_body(obj: Any, progress_items: list) -> float | None:
    """The Process Console's contents, in whatever window the caller opened.

    The System meters, the active tasks in ``progress_items``, in-process
    jobs and background processes, then a button dismissing finished ones.
    ``obj`` keeps the meters' throttles and the log boxes' layout between
    frames. Returns the height the window should grow to so nothing
    scrolls, or None when it already fits.
    """
    # the meters stay above the list, however long it scrolls
    _draw_system_info_header(obj)

    pm = get_process_manager()
    pm.cleanup_finished()
    running = pm.get_running()
    jobs = pm.get_jobs()

    avail = imgui.get_content_region_avail()
    # a resize can leave no client area, and begin_child needs a positive size
    content_height = max(1.0, avail.y - 35)

    # expanded log boxes share the leftover height, from last frame's layout
    n_boxes = getattr(obj, "_proc_log_count", 0)
    if n_boxes > 0:
        log_fill_h = max(
            _MIN_LOG_BOX_H,
            (content_height - getattr(obj, "_proc_log_fixed_h", 0.0)) / n_boxes,
        )
    else:
        log_fill_h = _MIN_LOG_BOX_H
    expanded_boxes = 0
    sum_box_h = 0.0
    content_h = 0.0

    # end_child runs even when begin_child clips, or the window stack unwinds wrong
    if imgui.begin_child(
        "##ProcessContent", ImVec2(0, content_height), imgui.ChildFlags_.none
    ):
        if progress_items:
            imgui.text_colored(_SYS_TITLE, f"Active Tasks ({len(progress_items)})")
            imgui.separator()
            imgui.spacing()

            for item in progress_items:
                pct = int(item["progress"] * 100)
                imgui.push_text_wrap_pos(0.0)
                if item.get("done", False):
                    imgui.text_colored(
                        imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
                        f"[Done] {item['text']}",
                    )
                else:
                    imgui.text(f"{item['text']}")
                imgui.pop_text_wrap_pos()
                imgui.progress_bar(item["progress"], ImVec2(-1, 0), f"{pct}%")
                imgui.spacing()

            if running or jobs:
                imgui.spacing()

        # in-process jobs run on a gui thread: no pid, log file or kill button
        if jobs:
            imgui.text_colored(_SYS_TITLE, f"In-Process Jobs ({len(jobs)})")
            imgui.separator()
            imgui.spacing()

            for job in jobs:
                if job.status == "error":
                    color = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
                    label = f"[Failed] {job.description}"
                elif job.status == "completed":
                    color = imgui.ImVec4(0.4, 1.0, 0.4, 1.0)
                    label = f"[Done] {job.description}"
                else:
                    color = imgui.ImVec4(1.0, 0.75, 0.3, 1.0)
                    label = job.description
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(color, label)
                if job.status_message:
                    imgui.text_disabled(f"    {job.status_message}")
                imgui.pop_text_wrap_pos()
                imgui.same_line()
                imgui.text_disabled(f"({job.elapsed_str()})")
                if job.is_alive():
                    imgui.progress_bar(job.progress, ImVec2(-1, 0))
                imgui.spacing()

            if running:
                imgui.spacing()

        if running:
            imgui.text_colored(_SYS_TITLE, f"Background Processes ({len(running)})")
            imgui.separator()
            imgui.spacing()

            for i, proc in enumerate(running):
                if i > 0:
                    imgui.separator()
                    imgui.spacing()
                box_h = _draw_process_entry(pm, proc, log_fill_h)
                if box_h > 0.0:
                    expanded_boxes += 1
                    sum_box_h += box_h

        if not running and not progress_items and not jobs:
            imgui.spacing()
            imgui.text_disabled("No active tasks or background processes.")

        # natural height of everything drawn, for next-frame window grow
        content_h = imgui.get_cursor_pos_y()
    imgui.end_child()

    # remember the non-box height and box count for next frame's split
    obj._proc_log_fixed_h = content_h - sum_box_h
    obj._proc_log_count = expanded_boxes

    imgui.separator()
    imgui.spacing()
    finished = [p for p in running if not p.is_alive()]
    if finished:
        dismiss_w = 150.0
        imgui.set_cursor_pos_x((imgui.get_window_width() - dismiss_w) * 0.5)
        if imgui.button(f"Dismiss finished ({len(finished)})", ImVec2(dismiss_w, 0)):
            for p in finished:
                pm._processes.pop(p.pid, None)
            pm._save()

    # a log box toggled this frame leaves the measured height off by one box
    win_h = imgui.get_window_height()
    max_h = max(300.0, imgui.get_main_viewport().work_size.y - 40.0)
    target_h = min(win_h - content_height + content_h, max_h)
    if target_h > win_h + 1.0 and expanded_boxes == n_boxes:
        return target_h
    return None


def _draw_process_entry(pm: Any, proc: Any, log_fill_h: float = 0.0) -> float:
    """Draw a single process entry in the console.

    ``log_fill_h`` is the height for an expanded log box. Returns the box
    height used (0.0 when the log is collapsed or absent).
    """
    imgui.push_id(f"proc_{proc.pid}")
    box_h = 0.0

    # status indicator + description
    if proc.status == "error":
        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), "[ERR]")
    elif proc.status == "completed":
        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), "[OK]")
    else:
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "[...]")

    imgui.same_line()

    # description wraps to the full content width
    imgui.push_text_wrap_pos(0.0)
    imgui.text(proc.description)
    imgui.pop_text_wrap_pos()

    # meta line: PID/elapsed on the left, actions right-aligned
    has_copy = bool(proc.output_path and Path(proc.output_path).is_file())
    action_label = "Kill" if proc.is_alive() else "Dismiss"
    style = imgui.get_style()

    def _btn_w(label: str) -> float:
        return imgui.calc_text_size(label).x + style.frame_padding.x * 2.0

    actions_w = _btn_w(action_label)
    if has_copy:
        actions_w += style.item_spacing.x + _btn_w("Copy")

    imgui.text_disabled(f"PID {proc.pid} · {proc.elapsed_str()}")
    imgui.same_line()
    right_x = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - actions_w
    imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), right_x))

    if proc.is_alive():
        if imgui.small_button("Kill"):
            pm.kill(proc.pid)
    else:
        if imgui.small_button("Dismiss"):
            pm._processes.pop(proc.pid, None)
            pm._save()

    if has_copy:
        imgui.same_line()
        if imgui.small_button("Copy"):
            try:
                with Path(proc.output_path).open(encoding="utf-8") as f:
                    imgui.set_clipboard_text(f.read())
            except Exception:
                pass

    # error message
    if proc.status == "error" and proc.status_message:
        imgui.push_text_wrap_pos(0)
        imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.6, 1.0), f"  {proc.status_message}")
        imgui.pop_text_wrap_pos()

    # collapsible log output
    if proc.output_path and Path(proc.output_path).is_file():
        tree_open = imgui.tree_node(f"Log Output##proc_{proc.pid}")
        if tree_open:
            try:
                lines = proc.tail_log(500)
                box_h = max(_MIN_LOG_BOX_H, log_fill_h)

                child_flags = imgui.ChildFlags_.borders
                # begin_child always needs end_child, regardless of return value
                imgui.begin_child(f"##log_{proc.pid}", ImVec2(-1, box_h), child_flags)
                # only auto-scroll when user is already pinned to the bottom;
                # otherwise scrolling up would be overridden every frame
                at_bottom = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 1.0
                line_h = imgui.get_text_line_height()
                imgui.push_text_wrap_pos(0.0)
                for line in lines:
                    color, display = _log_line_style(line.strip())
                    imgui.text_colored(color, display)
                    # wrapped line spans >1 row: add a small gap before the next
                    if imgui.get_item_rect_size().y > line_h + 1.0:
                        imgui.spacing()
                imgui.pop_text_wrap_pos()
                if at_bottom:
                    imgui.set_scroll_here_y(1.0)
                imgui.end_child()
            finally:
                imgui.tree_pop()

    imgui.spacing()
    imgui.pop_id()
    return box_h
