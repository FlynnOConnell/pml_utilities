"""Mean traces of drawn ROIs and pixel averages, measured on background threads."""

from __future__ import annotations

import queue
import threading

import numpy as np

from mbo_utilities.annotation import RoiTrace
from mbo_utilities.annotation.measure import measure_mean
from mbo_utilities.annotation.paths import slice_name
from mbo_utilities.arrays.features._slicing import index_window
from mbo_utilities.arrays.plane import PlaneMovie
from mbo_utilities.gui.roi._shared import PIXEL_RADIUS, _frames_of
from mbo_utilities.gui.widgets.process_manager import get_process_manager


class JobsMixin:
    """Mean traces of drawn ROIs and pixel averages, measured on background threads."""

    def _channel(self) -> int:
        return int(self.iw.indices[self.cdim]) if self.cdim is not None else 0

    def movie(
        self, plane: int | None = None, *, z: int | None = None, c: int | None = None
    ) -> PlaneMovie | None:
        """``(T, Y, X)`` view of the viewer's array behind store ``plane`` (the one on screen when None).

        ``z`` / ``c`` read another z-plane or channel with the same mask (a
        cell drawn on the structural channel, traced on the functional
        one). None when the array has no (T, Y, X) movie to wrap.
        """
        pos = self._plane_pos(self.z if plane is None else int(plane))
        zname, cname = self.store.axis_name("z"), self.store.axis_name("c")
        if z is None:
            z = pos.get(zname, 0) if zname is not None else 0
        if c is None:
            c = pos.get(cname, 0) if cname is not None else self._channel()
        try:
            arr = self.iw.data[0]
            nz = PlaneMovie(arr).nz
            return PlaneMovie(arr, z=(int(z) if nz > 1 else 0), c=int(c))
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def _coords(self, z=None, c=None, frames=None) -> tuple:
        """``(z, c, frames)`` a trace reads: the explicit ones, else ``run_where``'s (None reads where each ROI was drawn) and ``run_tp``."""
        if self.run_where == "screen":
            base_z, base_c = self.model.z, self.model.c
        else:
            base_z = base_c = None
        return (
            base_z if z is None else z,
            base_c if c is None else c,
            self.run_tp if frames is None else frames,
        )

    def _where_label(self) -> str:
        """The place a trace reads its pixels from, worded as the tooltips say it."""
        if self.run_where == "screen":
            where = f"slice on screen ({self._plane_label(self.z)})"
        else:
            where = "as drawn"
        if self.run_tp is not None:
            window = index_window(self.run_tp)
            where += (
                f", frames {window[0] + 1}-{window[1]}"
                + (f"-{window[2]}" if window[2] > 1 else "")
                if window
                else f", {len(self.run_tp)} frames"
            )
        return where

    def _frame_select(self, movie: PlaneMovie | None, tp) -> PlaneMovie | None:
        """``movie`` cut to the 0-based frames ``tp`` it has; the movie itself for None or every frame."""
        if movie is None or tp is None:
            return movie
        nt = int(movie.shape[0])
        kept = [int(t) for t in tp if 0 <= int(t) < nt]
        if not kept or kept == list(range(nt)):
            return movie
        return movie.select(kept)

    @property
    def trace_busy(self) -> bool:
        self._trace_threads = [t for t in self._trace_threads if t.is_alive()]
        return bool(self._trace_threads)

    @property
    def busy(self) -> bool:
        return self.trace_busy

    def has_traces(self) -> bool:
        return bool(self.traces)

    def trace_disabled(self, index: int) -> str | None:
        """The reason drawn ROI ``index`` cannot be traced now, or None."""
        movie = self.movie()
        if movie is None or int(movie.shape[0]) < 2:
            return "no (T, Y, X) movie behind this view"
        return None

    def quick_trace(self, index: int):
        """Mean of the ROI's pixels per frame, on a thread tracked as a job."""
        self.trace_rois([index])

    def trace_rois(
        self,
        indices: list[int],
        *,
        z: int | None = None,
        c: int | None = None,
        frames=None,
    ):
        """Mean-trace drawn ROIs on one background thread.

        Each mask is read at the coordinates given (else ``_coords``) and
        lands as a ``mean`` row of the trace table, replacing an earlier row
        at the same coordinates.
        """
        z, c, tp = self._coords(z, c, frames)
        work = []
        for target in self.model.targets(indices, z=z, c=c):
            movie = self._frame_select(
                self.movie(target.plane, z=target.z, c=target.c), tp
            )
            if movie is None:
                continue
            exclude = self.store.label_image(target.plane) > 0
            work.append((target, self.store.mask(target.index), exclude, movie))
        if not work:
            if indices:
                self.status = "nothing to trace"
            return
        averaged = int(getattr(self.host, "frame_average", 1) or 1)
        description = (
            f"quick trace - ROI {work[0][0].index}"
            if len(work) == 1
            else f"quick trace - {len(work)} ROIs"
        )
        job = get_process_manager().start_job("roi_trace", description)
        self.status = f"{description} started"
        thread = threading.Thread(
            target=self._run_quick,
            args=(work, job, averaged),
            name=f"roi-trace-{work[0][0].index}",
            daemon=True,
        )
        self._trace_threads.append(thread)
        thread.start()

    def _run_quick(self, work: list[tuple], job, averaged: int) -> None:
        """The quick-trace thread: one ``quick`` row per ROI in ``work``, queued for the draw thread."""
        frames_done = 0
        for n, (target, mask, exclude, movie) in enumerate(work):
            try:
                F, Fneu = measure_mean(movie, [mask], exclude)
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"quick trace - ROI {target.index} failed")
                job.fail(f"{type(error).__name__}: {error}")
                self._trace_results.put((target, None, str(error)))
                return
            frames_done = int(F.shape[1])
            trace = RoiTrace(
                uid=target.uid,
                z=target.z,
                c=target.c,
                engine="mean",
                source="quick",
                F=np.asarray(F[0], np.float32),
                Fneu=None if Fneu is None else np.asarray(Fneu[0], np.float32),
                frame_average=averaged,
                **_frames_of(movie),
            )
            self._trace_results.put((target, trace, None))
            job.set_progress((n + 1) / len(work), f"ROI {target.index}")
        job.done(f"{frames_done} frames" if len(work) == 1 else f"{len(work)} traces")

    def _traces_changed(self):
        """Trace rows moved: drop stale stats and picks, redraw the lines, refresh the peaks."""
        self._trace_stats.clear()
        self._trace_display.clear()
        self.trace_sel &= set(self.traces.keys)
        if not self.trace_picked:
            self.trace_sel = set(self._selection_trace_keys())
        self._plot_lines_key = None
        if len(self.order.columns.get("peak", ())) == len(self.rows):
            self.order.columns["peak"] = self._row_peaks()
        self._footprints = None

    def selection_indices(self) -> list[int]:
        """Drawn ROIs the selection covers: the group if there is one, else the selected ROI."""
        grouped = [k for si, k in self.buffer if si < 0]
        return grouped or ([self.selected] if self.selected >= 0 else [])

    def listed_drawn(self) -> list[int]:
        """Store indices of the drawn ROIs the table currently lists."""
        return [
            self.rows[int(r)][1] for r in self.order.order if self.rows[int(r)][0] < 0
        ]

    def trace_in_view(self):
        """Quick trace every drawn ROI the table currently lists."""
        self.trace_rois(self.listed_drawn())

    def _poll_jobs(self):
        """Drain finished traces, once per frame from the panel."""
        while True:
            try:
                target, trace, error = self._trace_results.get_nowait()
            except queue.Empty:
                break
            index = self.store.uid_index(target.uid) if target is not None else None
            if error is not None:
                if target is None:
                    self.status = f"trace failed: {error}"
                else:
                    shown = index if index is not None else f"uid {target.uid}"
                    self.status = f"trace for ROI {shown} failed: {error}"
                continue
            if target is not None and index is None:
                continue  # deleted while the trace ran
            if trace.source == "pixel" and trace.key not in self.pixels.values():
                continue  # dropped (esc, d) while the average ran
            self.traces.add(trace)
            if trace.stands_for_roi:
                self.trace_uid = trace.uid
            self.status = (
                f"{trace.label or 'ROI ' + str(index)}: {trace.n_frames} frames"
            )

    def add_pixel(self, row: int, col: int, toggle: bool = False):
        """A pixel average (p): the movie's 5x5 mean around ``(row, col)`` on the slice on screen joins the group and the plot; ctrl on a grouped one takes it out."""
        pid = (self.z, int(row), int(col))
        if toggle and pid in self.pixel_group:
            self.pixel_group.remove(pid)
            self.active_pixel = None
            self._plot_lines_key = None
            self.refresh_overlay()
            return
        if pid not in self.pixels:
            key = self._trace_pixel(int(row), int(col))
            if key is None:
                return
            self._snapshot()
            self.pixels[pid] = key
            self.pixels.move_to_end(pid, last=False)
        self._seed_buffer()
        if pid not in self.pixel_group:
            self.pixel_group.append(pid)
        self.selected = -1  # the ROI stays in the group
        self.selected_derived = None
        self._note_buf = ""
        self.active_pixel = pid
        self._plot_lines_key = None
        self.refresh_overlay()

    def _trace_pixel(self, row: int, col: int) -> tuple | None:
        """Start the 5x5 average around one pixel on a background job; returns the row's key."""
        movie = self.movie()
        if movie is None or int(movie.shape[0]) < 2:
            self.status = "no (T, Y, X) movie behind this view"
            return None
        mask = np.zeros(movie.shape[1:], bool)
        mask[
            max(row - PIXEL_RADIUS, 0) : row + PIXEL_RADIUS + 1,
            max(col - PIXEL_RADIUS, 0) : col + PIXEL_RADIUS + 1,
        ] = True
        zz, cc = int(movie.z), int(movie.c)
        label = f"px {row},{col}"
        trace = RoiTrace(
            uid=0,
            member=f"{row},{col} {slice_name(zz, cc)}",
            source="pixel",
            label=label,
            z=zz,
            c=cc,
            engine="mean",
            frame_average=int(getattr(self.host, "frame_average", 1) or 1),
            extra={"pixel": (row, col)},
        )
        job = get_process_manager().start_job("roi_trace", f"pixel average {label}")
        thread = threading.Thread(
            target=self._run_pixel,
            args=(movie, mask, trace, job),
            name="roi-trace-pixel",
            daemon=True,
        )
        self._trace_threads.append(thread)
        thread.start()
        return trace.key

    def _run_pixel(self, movie, mask, trace: RoiTrace, job):
        try:
            F, _fneu = measure_mean(movie, [mask], neuropil=False)
        except Exception as error:  # noqa: BLE001 - reported on the job
            self.logger.exception(f"{trace.label} failed")
            job.fail(f"{type(error).__name__}: {error}")
            self._trace_results.put((None, None, f"{trace.label}: {error}"))
            return
        trace.F = np.asarray(F[0], np.float32)
        trace.frames = _frames_of(movie)["frames"]
        self._trace_results.put((None, trace, None))
        job.done(f"{int(F.shape[1])} frames")

    def drop_pixel(self, pid: tuple):
        """Take one pixel average off the plot, the table and the trace rows."""
        key = self.pixels.pop(pid, None)
        if pid in self.pixel_group:
            self.pixel_group.remove(pid)
        if self.active_pixel == pid:
            self.active_pixel = None
        if key is not None:
            self.traces.remove(key)
        self._plot_lines_key = None

    def set_pixel_traces(self, on: bool):
        """Pixel traces (p); turning them off drops every pixel average."""
        self.pixel_traces = bool(on)
        if on:
            return
        for pid in list(self.pixels):
            self.drop_pixel(pid)
        self.refresh_overlay()
