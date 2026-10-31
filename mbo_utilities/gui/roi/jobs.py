"""Measuring traces and running ROIs.

Quick and pixel traces, the mean / suite2p / masknmf runs and their adoption.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from mbo_utilities.annotation import (
    ENGINES,
    RoiTrace,
)
from mbo_utilities.arrays.features._dim_tags import (
    TAG_REGISTRY,
    DimensionTag,
)
from mbo_utilities.arrays.features._slicing import index_window
from mbo_utilities.gui import roi_runs
from mbo_utilities.gui.roi._shared import (
    _STAGE_NAMES,
    ADOPT_INTERVAL_S,
    DEFAULT_RUN_TAG,
    PIXEL_RADIUS,
    _frames_of,
    _RunSpec,
)
from mbo_utilities.gui.roi_runs import (
    RoiRun,
    finished_dirs,
    run_dir_complete,
    unit_trace,
)
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.results import (
    pipeline_files,
)
from mbo_utilities.roi_workflow import (
    PlaneMovie,
    demix_rois,
    extract_rois,
    measure_mean,
    roi_folder,
    run_path,
    slice_name,
    write_run,
)


class JobsMixin:
    """Quick and pixel traces, the mean / suite2p / masknmf runs, their adoption."""

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
        """``(z, c, frames)`` a run reads: the explicit ones, else ``run_where``'s (None reads where each ROI was drawn) and ``run_tp``."""
        if self.run_where == "screen":
            base_z, base_c = self.model.z, self.model.c
        elif self.run_where == "fixed":
            base_z, base_c = self.run_z, self.run_c
        else:
            base_z = base_c = None
        return (
            base_z if z is None else z,
            base_c if c is None else c,
            self.run_tp if frames is None else frames,
        )

    def _where_label(self) -> str:
        """The place a run reads its pixels from, worded as the tooltips say it."""
        if self.run_where == "screen":
            where = f"slice on screen ({self._plane_label(self.z)})"
        elif self.run_where == "fixed":
            parts = []
            if self.run_z is not None:
                parts.append(self._axis_tag("z", self.run_z))
            if self.run_c is not None:
                parts.append(self._axis_tag("c", self.run_c))
            where = " ".join(parts) if parts else "as drawn"
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
        return self.trace_busy or self.manager.busy

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

    def _merge_run_traces(self, unit, name: str, path) -> int:
        """Add a run unit's rows to the trace table under source ``name``, keyed by the store uids its ``uids`` attr names.

        Rows whose ROI is gone are skipped; a run that did not record where
        it read gets the ROI's own z-plane and channel. Returns how many landed.
        """
        uids = unit.attrs.get("uids")
        if not uids or len(uids) != unit.n_rois:
            return 0
        merged = 0
        for row, uid in enumerate(uids):
            index = self.store.uid_index(int(uid))
            if index is None:
                continue
            z, c = unit.attrs.get("z"), unit.attrs.get("c")
            trace = unit_trace(
                unit,
                row,
                uid=int(uid),
                member=None,
                source=name,
                path=path,
                z=self.store.roi_z(index) if z is None else int(z),
                c=self.store.roi_c(index) if c is None else int(c),
                engine=str(unit.attrs.get("engine") or "mean"),
            )
            if trace is None:
                continue
            self.traces.add(trace)
            self.trace_uid = trace.uid
            merged += 1
        return merged

    def pipeline_for(self, engine: str | None = None) -> str | None:
        """The pipeline whose parameters ``engine`` runs on; None for the numpy mean extractor."""
        engine = self.engine if engine is None else engine
        return engine if engine in ("masknmf", "suite2p") else None

    def masknmf_settings(self) -> dict | None:
        """The masknmf parameters set in the Process tab, or None for defaults."""
        return roi_runs.masknmf_settings(self.host)

    def _pipeline_summary(self, kind: str) -> tuple[str, str]:
        """``(label, tooltip)`` for what a run of ``kind`` would use."""
        if kind == "masknmf":
            instances = getattr(self.host, "_pipeline_instances", None) or {}
            settings = getattr(instances.get("MaskNMF"), "settings", None)
            if settings is None:
                return "masknmf: defaults", (
                    "The Process tab has not built masknmf yet, so this runs on "
                    "its defaults. Open it to set registration, compression "
                    "and demixing parameters."
                )
            from mbo_utilities.gui.widgets.pipelines.masknmf import _collect_modified

            stages = " · ".join(
                f"{name}:{_STAGE_NAMES[int(getattr(section, attr, 1)) % 3]}"
                for name, section, attr in (
                    ("reg", settings.registration, "do_registration"),
                    ("pmd", settings.compression, "do_compression"),
                    ("demix", settings.demixing, "do_demixing"),
                )
            )
            changed = _collect_modified(settings)
            label = (
                f"masknmf: {len(changed)} changed" if changed else "masknmf: defaults"
            )
            detail = "\n".join(f"{n} = {v}  (default {d})" for n, v, d in changed[:12])
            return label, f"{stages}\n{detail}" if detail else stages
        s2p = getattr(self.host, "s2p", None)
        if s2p is None:
            return "suite2p: defaults", (
                "The Process tab has not been opened yet, so this runs on "
                "suite2p's defaults."
            )
        from mbo_utilities.gui.widgets.pipelines.settings import collect_modified_params

        stages = " · ".join(
            f"{name}:{_STAGE_NAMES[int(getattr(s2p, attr, 1)) % 3]}"
            for name, attr in (("reg", "do_registration"), ("detect", "do_detection"))
        )
        changed = collect_modified_params(
            s2p,
            getattr(self.host, "s2p_db", None),
            getattr(self.host, "s2p_extras", None),
        )
        label = f"suite2p: {len(changed)} changed" if changed else "suite2p: defaults"
        detail = "\n".join(f"{row[0]} = {row[1]}" for row in changed[:12])
        return label, f"{stages}\n{detail}" if detail else stages

    def open_pipeline_params(self, kind: str):
        """Bring the Process tab forward with ``kind`` selected, where its parameters are edited."""
        if self.host is None:
            self.status = "no Process tab to open"
            return
        self.host._selected_pipeline_name = (
            "MaskNMF" if kind == "masknmf" else "Suite2p"
        )
        self.host._force_run_tab = True
        self.status = f"{kind} parameters are in the Process tab"

    def _run_out_path(self, tag: str) -> Path | None:
        """``<stem>.<tag>.<stamp>.manual.zarr`` in the ROI folder (the recording's own tag first); None, with the reason in the status."""
        if self.fpath is None:
            self.status = "no data path to write beside"
            return None
        if any(
            r.tag == tag and r.job is not None and not r.finished
            for r in self.manager.runs
        ):
            self.status = f"{tag} is still being written"
            return None
        tags = "_".join(t for t in (self.tag, tag) if t)
        return run_path(self.fpath, tags, self.folder)

    def run_rois(
        self,
        indices: list[int],
        tag: str,
        *,
        z: int | None = None,
        c: int | None = None,
        frames=None,
        engine: str | None = None,
    ):
        """Run drawn ROIs through an extraction engine on the viewer's array, into one results zarr in the ROI folder.

        Each mask stays on the plane it was drawn on; its pixels come from
        ``z`` / ``c`` / ``frames`` (else ``_coords``). Each z-plane and
        channel read is one unit (``zplane02``, ``ch01_zplane02``). The run
        closes over a store snapshot, so drawing on is safe while it works.
        """
        indices = [int(i) for i in indices if 0 <= int(i) < self.n_rois]
        if not indices:
            self.status = "nothing to run"
            return
        tag = (tag or "").strip() or DEFAULT_RUN_TAG
        out_path = self._run_out_path(tag)
        if out_path is None:
            return
        engine = self.engine if engine is None else str(engine)
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; one of {ENGINES}")
        z, c, tp = self._coords(z, c, frames)
        store = self.store.snapshot()
        groups: dict[tuple[int, int, int], list[int]] = {}
        for target in self.model.targets(indices, z=z, c=c):
            groups.setdefault((target.plane, target.z, target.c), []).append(
                target.index
            )
        movies = {}
        for key in groups:
            plane, zz, cc = key
            movie = self._frame_select(self.movie(plane, z=zz, c=cc), tp)
            if movie is None or int(movie.shape[0]) < 2:
                self.status = (
                    "no (T, Y, X) movie behind this view: nothing to trace over"
                )
                return
            movies[key] = movie
        spec = _RunSpec(
            groups=groups,
            movies=movies,
            store=store,
            engine=engine,
            tag=tag,
            out_path=out_path,
            source=self.iw.data[0],
            settings=self.masknmf_settings() if engine == "masknmf" else None,
            many_c=len({cc for _p, _z, cc in groups}) > 1,
            work=out_path.with_name(out_path.name + ".work"),
        )
        run = RoiRun(
            kind="demix" if engine == "masknmf" else "extract",
            tag=tag,
            description=f"{engine}: {len(indices)} ROI(s) -> {out_path.name}",
            out_root=out_path,
            planes=sorted({zz + 1 for _p, zz, _c in groups}),
        )
        self.manager.submit(
            run, lambda job: self._run_job(job, spec), heavy=(engine == "masknmf")
        )
        self._run_error = None
        self.status = f"{run.description} started"

    def _run_job(self, job, spec: _RunSpec) -> Path | None:
        """The run thread: one unit per slice read, written as one results zarr; masknmf's native files move into its ``manual/`` folder.

        Returns the file, or None when no ROI was on any plane read.
        """
        units = []
        for i, (key, on_plane) in enumerate(spec.groups.items()):
            plane, zz, cc = key
            name = slice_name(zz, cc if spec.many_c else None)
            job.set_progress(i / len(spec.groups), name)
            if spec.engine == "masknmf":
                unit = demix_rois(
                    spec.movies[key],
                    spec.store,
                    on_plane,
                    z=plane,
                    c=cc,
                    work_dir=spec.work / name,
                    settings=spec.settings,
                    logger=self.logger,
                )
            else:
                unit = extract_rois(
                    spec.movies[key],
                    spec.store,
                    on_plane,
                    z=plane,
                    c=cc,
                    engine=spec.engine,
                    logger=self.logger,
                )
            if unit is None:
                continue
            unit.name = name
            while any(u.name == unit.name for u in units):
                unit.name += (
                    f"_p{plane}"  # masks drawn on two channels, traced on one slice
                )
            units.append(unit)
        if not units:
            return None
        write_run(
            spec.out_path,
            units,
            source=spec.source,
            store=spec.store,
            tag=spec.tag,
            settings=spec.settings or {},
        )
        if spec.work.exists():
            native = pipeline_files(spec.out_path)
            native.mkdir(parents=True, exist_ok=True)
            for child in spec.work.iterdir():
                shutil.move(str(child), str(native / child.name))
            shutil.rmtree(spec.work)
        return spec.out_path

    def selection_indices(self) -> list[int]:
        """Drawn ROIs the selection covers: the group if there is one, else the selected ROI."""
        grouped = [k for si, k in self.buffer if si < 0]
        return grouped or ([self.selected] if self.selected >= 0 else [])

    def run_selection(self, indices: list[int] | None = None):
        """Run the selection through the widget's engine: one ROI into its own ``roiNN`` run, a group into one run under the tag."""
        indices = self.selection_indices() if indices is None else indices
        if not indices:
            self.status = "select an ROI first"
            return
        if len(indices) == 1:
            self.run_roi(indices[0])
        else:
            self.run_rois(indices, self.effective_tag)

    def run_roi(self, index: int):
        """Run one drawn ROI into ``rois_roiNN/`` (the R tag, 1-based)."""
        self.run_rois(
            [index], DimensionTag(TAG_REGISTRY["R"], index + 1, None).to_string()
        )

    @property
    def effective_tag(self) -> str:
        """The typed run tag, else the default."""
        return (self.run_tag or "").strip() or DEFAULT_RUN_TAG

    def listed_drawn(self) -> list[int]:
        """Store indices of the drawn ROIs the table currently lists."""
        return [
            self.rows[int(r)][1] for r in self.order.order if self.rows[int(r)][0] < 0
        ]

    def run_in_view(self):
        """Run every drawn ROI the table currently lists."""
        self.run_rois(self.listed_drawn(), self.effective_tag)

    def trace_in_view(self):
        """Quick trace every drawn ROI the table currently lists."""
        self.trace_rois(self.listed_drawn())

    def _poll_jobs(self):
        """Drain finished traces and runs, once per frame from the panel."""
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
        for run, payload in self.manager.poll(get_process_manager()):
            if run.error is not None:
                self._run_error = f"{run.description} failed: {run.error}"
                continue
            if run.job is not None:
                if isinstance(payload, (str, Path)):
                    run.out_dirs = [Path(payload)]
                elif payload:
                    run.out_dirs = [Path(o) for o in payload]
                outs = [d for d in run.out_dirs if run_dir_complete(d)]
            else:
                # a spawned pipeline may suffix the plane dir name, so the guessed dirs are replaced from disk
                outs = finished_dirs(run.out_root, run.planes) if run.out_root else []
                if outs:
                    guessed = {str(d) for d in run.out_dirs}
                    self._registry_extra = [
                        e for e in self._registry_extra if str(e["path"]) not in guessed
                    ]
                    run.out_dirs = outs
            loaded = sum(self.load_run(d) for d in outs)
            run.loaded = bool(loaded)
            if outs and loaded == len(outs):
                self._run_error = None
                names = ", ".join(d.name for d in outs)
                took = run.job.elapsed_str() if run.job is not None else ""
                self.logger.info(
                    f"roi run done: {names}" + (f" in {took}" if took else "")
                )
                self.status = f"done: {names}"
                if run.job is not None:
                    run.job.status_message = f"{names} · {took}"
            elif not outs:
                self.status = f"{run.description}: nothing written"
                if run.job is not None:
                    run.job.status_message = "nothing written"
        self._adopt_finished_runs()

    def _adopt_finished_runs(self):
        """Load the runs the Process tab's workers finish beside this data, with their own iscell."""
        now = time.monotonic()
        if now - self._adopt_checked < ADOPT_INTERVAL_S:
            return
        self._adopt_checked = now
        if self.fpath is None:
            return
        roots = {roi_folder(self.fpath), self.folder}
        mine = {r.pid for r in self.manager.runs if r.pid is not None}
        loaded = {str(s.path) for s in self.derived} | self._results_loaded
        for info in get_process_manager().get_running():
            if (
                info.pid in mine
                or info.pid in self._adopted
                or info.status != "completed"
                or info.task_type not in ("suite2p", "masknmf", "vnoiser")
            ):
                continue
            self._adopted.add(info.pid)
            args = info.args or {}
            out = args.get("output_dir")
            if not out:
                continue
            out = Path(out)
            if not any(
                root in (out, *out.parents) or out in root.parents for root in roots
            ):
                continue  # another dataset's run
            if info.task_type == "vnoiser":
                # one results file beside the data, or (--pkl) a PF folder
                from mbo_utilities.results import newest_results, results_dir_of

                found = newest_results(out, "vnoiser") or results_dir_of(out)
                dirs = [found] if found is not None and str(found) not in loaded else []
            else:
                dirs = [
                    d
                    for d in finished_dirs(out, args.get("planes"))
                    if str(d) not in loaded
                ]
            # a volume run's other planes are not this widget's error
            held = self._run_error
            took = [d for d in dirs if self.load_run(d)]
            self._run_error = held
            if took:
                names = ", ".join(d.name for d in took)
                self.logger.info(f"adopted {info.task_type} run: {names}")
                self.status = f"loaded {info.task_type}: {names}"
            elif dirs:
                self.logger.debug(
                    f"{info.task_type} run at {out} has no dir for this plane"
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
