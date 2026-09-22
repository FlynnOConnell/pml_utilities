"""One curation mode on one data path, driven without a notebook.

``EventCurationDashboard`` holds every rule of the curation notebook: which
candidates a threshold yields, how the seed template and cosine calls are
made, what the saved JSON looks like. ipywidgets and plotly build fine
without a frontend, so the session owns a dashboard and steers it through
the same handlers its widgets call; the imgui panels read arrays off it and
never re-implement a rule. Each call still redraws the dashboard's plotly
figures, which is why label and threshold changes cost tens of ms.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from vnoiser.curation import (
    LABEL_COLORS,
    PC1_SIDES,
    PIPELINE_CACHE_VERSION,
    EventCurationDashboard,
)
from vnoiser.dataset import RecordingSample

from mbo_utilities.arrays.pf import TRACES_FILE, PfArray, pf_results_in

__all__ = [
    "LABEL_RGBA",
    "MODES",
    "PC1_SIDES",
    "CurationSession",
    "hex_rgba",
    "pf_dir_for_mesc",
    "pf_scan_for_mesc",
]

MODES = ("fast", "slow", "manual")
LABELS = ("yes", "no", "auto_yes", "auto_no", "unlabeled")
VIEW_FILTERS = ("all", "yes", "no", "unlabeled")

_TAGS = re.compile(r"<[^>]+>")


def hex_rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """``#rrggbb`` as floats in 0..1."""
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, float(alpha))


LABEL_RGBA = {name: hex_rgba(value) for name, value in LABEL_COLORS.items()}


def _text(html: str) -> str:
    return _TAGS.sub("", html or "").replace("&lt;", "<").replace("&gt;", ">").strip()


class CurationSession:
    """Curation of one mode over one recording.

    Parameters
    ----------
    data_path : str or Path
        Where the recording comes from: the ``PF`` folder (labels go to its
        ``.curation``), or the file a raw trace was read from (labels go
        beside it).
    mode : str
        ``"fast"``, ``"slow"`` or ``"manual"``.
    slow_cutoff_hz : float
        Low-pass cutoff of the slow mode's analysis trace.
    duration_s : float, optional
        Window length to curate; None curates the whole trace.
    """

    def __init__(self, data_path, mode="fast", slow_cutoff_hz=40.0, duration_s=None, **kwargs):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.dash = EventCurationDashboard(
            data_path=Path(data_path).expanduser(),
            mode=mode,
            duration_s=duration_s,
            slow_cutoff_hz=slow_cutoff_hz,
            auto_load=False,
            **kwargs,
        )
        self.recording_id: str = ""

    # ------------------------------------------------------------------
    # data source
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self.dash.mode

    @property
    def data_path(self) -> Path:
        return self.dash.data_path

    def load_pf(self, pf: PfArray, scan: str, domain: str) -> str:
        """Curate the pipeline's denoised trace of one scan / domain of a
        ``PF`` folder: no denoiser, labels in ``PF/.curation``. Returns the
        status line; raises ``KeyError`` for a scan or domain the folder
        does not hold."""
        scan, domain = str(scan), str(domain)
        if scan not in pf.traces or domain not in pf.traces[scan]:
            raise KeyError(f"{pf.pf_dir} has no trace for scan {scan}, domain {domain}")
        return self.load_trace(
            pf.traces[scan][domain],
            pf.fs_by_scan[scan],
            recording_id=pf.recording_id(domain, scan),
            label=f"scan {scan} / {domain}",
            source_path=pf.results_path or pf.pf_dir / TRACES_FILE,
            curation_dir=pf.pf_dir / ".curation",
            pre_denoised=True,
        )

    def load_trace(
        self,
        trace,
        fs_hz: float,
        *,
        recording_id: str,
        label: str,
        source_path,
        curation_dir=None,
        pre_denoised: bool = False,
    ) -> str:
        """Curate a trace held in memory: an ROI trace pulled from a line
        scan, say. vnoiser's denoiser runs on it, cached under
        ``curation_dir/cache`` by ``recording_id`` and the source file's size
        and mtime; ``pre_denoised`` takes the trace as the pipeline's output
        instead. Labels go to ``curation_dir/<mode>_template_curation.json``
        keyed by ``recording_id``. Returns the status line."""
        trace = np.asarray(trace, dtype=float).ravel()
        if trace.size < 2:
            raise ValueError(f"{label} has fewer than two samples")
        if not np.isfinite(trace).all():
            raise ValueError(f"{label} contains NaN or infinite values")
        source_path = Path(source_path)
        curation_dir = (
            Path(curation_dir) if curation_dir is not None else source_path.parent / ".curation"
        )
        t = np.arange(trace.size, dtype=float) / float(fs_hz)
        full = RecordingSample(
            t=t,
            trace=trace,
            fs_hz=float(fs_hz),
            events_ap_indices=np.array([], dtype=int),
            events_ap_times_s=np.array([], dtype=float),
            path=source_path,
            metadata={
                "recording_id": str(recording_id),
                "label": str(label),
                "fs_hz": float(fs_hz),
                "duration_s": float(t[-1]),
                "n_samples": int(trace.size),
                "curation_dir": str(curation_dir),
                "pre_denoised": bool(pre_denoised),
                "source_format": "pf" if pre_denoised else "trace",
            },
        )
        dash = self.dash
        recording = dash._window_from_recording(full)
        dash._activate_recording_storage(recording)
        cache_path = None if pre_denoised else self._trace_cache_path(recording)
        if pre_denoised:
            dash._run_or_load_pipeline(recording)
        elif cache_path is not None and cache_path.exists():
            dash._load_pipeline_cache(recording, cache_path)
            dash.pipeline_cache_status = f"loaded cache: {cache_path.name}"
        else:
            dash._run_pipeline(recording)
            dash.pipeline_cache_status = "computed pipeline"
            if cache_path is not None:
                dash._save_pipeline_cache(cache_path)
        dash.event_slider.max = max(0, len(dash.candidates.indices) - 1)
        dash.event_slider.value = 0
        dash._set_loaded_controls(True)
        dash._refresh_all()
        dash.status.value = (
            f"<b>Status:</b> loaded {label}; {len(dash.event_keys)} candidates; "
            f"{dash.pipeline_cache_status}."
        )
        self.recording_id = str(recording_id)
        return self.status

    def _trace_cache_path(self, recording) -> Path | None:
        """Like the dashboard's cache path, with the recording id in the key
        so several traces of one source file do not share a cache."""
        if not self.dash.enable_pipeline_cache:
            return None
        stat = Path(recording.path).stat()
        payload = {
            "version": PIPELINE_CACHE_VERSION,
            "recording_id": recording.metadata.get("recording_id"),
            "path": str(Path(recording.path).resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "fs_hz": float(recording.fs_hz),
            "n_samples": int(len(recording.trace)),
            "window_start_index": int(recording.metadata.get("window_start_index", 0)),
            "duration_s": self.dash.duration_s,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(recording.metadata.get("recording_id")))
        return self.dash.curation_dir / "cache" / f"{stem}-{digest}.npz"

    @property
    def loaded(self) -> bool:
        return self.dash.recording is not None

    @property
    def status(self) -> str:
        return _text(self.dash.status.value).removeprefix("Status:").strip()

    @property
    def recording_label(self) -> str:
        rec = self.dash.recording
        if rec is None:
            return ""
        return str(rec.metadata.get("label", rec.path.name))

    @property
    def label_path(self) -> Path:
        return self.dash.label_path

    @property
    def cache_status(self) -> str:
        return self.dash.pipeline_cache_status

    # ------------------------------------------------------------------
    # traces
    # ------------------------------------------------------------------

    @property
    def fs(self) -> float:
        return float(self.dash.recording.fs_hz) if self.loaded else 0.0

    @property
    def t(self) -> np.ndarray:
        return self.dash.recording.t if self.loaded else np.array([], dtype=float)

    @property
    def denoised(self) -> np.ndarray:
        return self.dash.denoised

    @property
    def analysis_trace(self) -> np.ndarray:
        """What candidates are thresholded on: the denoised trace, or its
        low-pass view in slow mode."""
        return self.dash.analysis_trace

    @property
    def second_pass(self) -> np.ndarray:
        return self.dash.second_pass

    # ------------------------------------------------------------------
    # detection settings
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return float(self.dash.candidate_threshold)

    @property
    def threshold_range(self) -> tuple[float, float, float]:
        """``(low, high, step)`` of the data-calibrated threshold slider."""
        s = self.dash.threshold_slider
        return float(s.min), float(s.max), float(s.step)

    def set_threshold(self, value: float) -> None:
        if not self.loaded:
            return
        lo, hi, _ = self.threshold_range
        value = float(np.clip(value, lo, hi))
        if value == self.threshold:
            return
        self.dash._threshold_changed({"new": value})

    @property
    def seeded(self) -> bool:
        """Whether auto calls apply (fast and slow modes)."""
        return self.mode != "manual"

    @property
    def auto_pass(self) -> float | None:
        return self.dash.auto_pass_amplitude

    @property
    def auto_pass_range(self) -> tuple[float, float, float]:
        """``(low, high, step)`` for the auto-pass line.

        A2 is a line on the trace and is compared with the value a
        candidate's peak sits at, so it shares the threshold slider's range,
        whose floor is the trace median. A retained manual event can peak
        under that floor, so the floor drops under the lowest peak and A2 all
        the way down passes every candidate.
        """
        s = self.dash.auto_pass_slider
        lo, hi, step = float(s.min), float(s.max), float(s.step)
        if self.loaded and self.n:
            lowest = float(np.nanmin(self.peak_values))
            if np.isfinite(lowest):
                lo = min(lo, lowest - step)
        return lo, hi, step

    def auto_pass_count(self) -> int:
        """How many candidates the auto-pass line passes now."""
        if not self.loaded or self.auto_pass is None or not self.n:
            return 0
        return int(np.count_nonzero(self.peak_values >= float(self.auto_pass)))

    def set_auto_pass(self, value: float) -> None:
        if not self.loaded or not self.seeded:
            return
        lo, hi, _ = self.auto_pass_range
        value = float(np.clip(value, lo, hi))
        if self.auto_pass is not None and value == self.auto_pass:
            return
        self.dash._auto_pass_changed({"new": value})

    @property
    def waveform_rejection(self) -> bool:
        return bool(self.dash.waveform_rejection)

    def set_waveform_rejection(self, value: bool) -> None:
        if not self.loaded or not self.seeded or bool(value) == self.waveform_rejection:
            return
        self.dash._waveform_rejection_changed({"new": bool(value)})

    # A3: the PC1 line on the candidate PCA

    @property
    def auto_pass_pc1(self) -> float | None:
        """PC1 score of the A3 line; None while the rule is off."""
        return self.dash.auto_pass_pc1

    @property
    def auto_pass_pc1_side(self) -> str:
        """``"right"``: PC1 at or above the line passes; ``"left"``: at or below."""
        return str(self.dash.auto_pass_pc1_side)

    @property
    def auto_pass_pc1_range(self) -> tuple[float, float, float]:
        """``(low, high, step)`` of the A3 line: the candidates' PC1 scores
        padded so the line at either end passes none."""
        lo, hi, step = self.dash.pc1_slider_range()
        return float(lo), float(hi), float(step)

    @property
    def auto_pass_pc1_shown(self) -> float:
        """Where the A3 line is drawn: its value, or parked at the end of
        the passing side (nothing passes) while the rule is off."""
        if self.auto_pass_pc1 is not None:
            return float(self.auto_pass_pc1)
        lo, hi, _ = self.auto_pass_pc1_range
        return lo if self.auto_pass_pc1_side == "left" else hi

    def auto_pass_pc1_count(self) -> int:
        """How many candidates the A3 line passes now."""
        if not self.loaded or self.auto_pass_pc1 is None or not self.n:
            return 0
        return int(sum(self.dash._pc1_passes(i) for i in range(self.n)))

    def set_auto_pass_pc1(self, value: float) -> None:
        if not self.loaded or not self.seeded:
            return
        lo, hi, _ = self.auto_pass_pc1_range
        value = float(np.clip(value, lo, hi))
        if self.auto_pass_pc1 is not None and value == self.auto_pass_pc1:
            return
        self.dash._auto_pass_pc1_changed({"new": value})

    def set_auto_pass_pc1_side(self, side: str) -> None:
        if not self.loaded or not self.seeded or side not in PC1_SIDES:
            return
        if side == self.auto_pass_pc1_side:
            return
        self.dash._auto_pass_pc1_side_changed({"new": side})

    # A4: the seed-template cosine at or above which a candidate passes

    @property
    def auto_template_threshold(self) -> float:
        return float(self.dash.auto_template_threshold)

    @property
    def auto_template_threshold_range(self) -> tuple[float, float, float]:
        s = self.dash.cosine_slider
        return float(s.min), float(s.max), float(s.step)

    def auto_template_count(self) -> int:
        """How many candidates the cosine threshold passes on its own."""
        if not self.loaded or not self.seeded or not self.n:
            return 0
        scores = np.asarray(self.dash.initial_template_scores, dtype=float)
        if len(scores) < self.n:
            return 0
        return int(np.count_nonzero(scores[: self.n] >= self.auto_template_threshold))

    def set_auto_template_threshold(self, value: float) -> None:
        if not self.loaded or not self.seeded:
            return
        lo, hi, _ = self.auto_template_threshold_range
        value = float(np.clip(value, lo, hi))
        if value == self.auto_template_threshold:
            return
        self.dash._auto_template_threshold_changed({"new": value})

    # ------------------------------------------------------------------
    # candidates and labels
    # ------------------------------------------------------------------

    @property
    def candidates(self):
        return self.dash.candidates

    @property
    def n(self) -> int:
        return int(len(self.dash.candidates.indices))

    @property
    def times_s(self) -> np.ndarray:
        return self.dash.candidates.times_s

    @property
    def amplitudes(self) -> np.ndarray:
        """Baseline-subtracted peak of each candidate, the height panel C draws."""
        return self.dash.candidates.amplitudes

    @property
    def peak_values(self) -> np.ndarray:
        """Trace value each candidate peaks at, the height panel A draws it at."""
        return self.dash.candidates.peak_values

    @property
    def current(self) -> int:
        return int(self.dash.current)

    def select(self, index: int) -> None:
        if 0 <= index < self.n:
            self.dash._select_event(int(index))

    def step(self, delta: int) -> None:
        self.dash._step_event(int(delta))

    @property
    def view_filter(self) -> str:
        return str(self.dash.view_filter.value)

    def set_view_filter(self, value: str) -> None:
        if value in VIEW_FILTERS and value != self.view_filter:
            self.dash.view_filter.value = value

    @property
    def visible(self) -> np.ndarray:
        """Candidate indices passing the view filter."""
        return np.asarray(self.dash.visible_indices, dtype=int)

    def label(self, index: int) -> str:
        """The shown label: a manual yes/no, else the auto call as
        ``auto_yes`` / ``auto_no``, else ``unlabeled``."""
        return self.dash._label_for_index(int(index))

    def labels(self) -> list[str]:
        return [self.label(i) for i in range(self.n)]

    def manual_label(self, index: int) -> str:
        return self.dash.labels.get(self.dash.event_keys[int(index)], "unlabeled")

    def set_label(self, label: str) -> None:
        """Label the current candidate ``yes``, ``no`` or ``unlabeled`` and
        save; the template and second pass follow."""
        if label not in ("yes", "no", "unlabeled") or not self.n:
            return
        self.dash._set_label(label)

    def set_labels(self, indices, label: str) -> int:
        """Label several candidates at once (a box selection) and save;
        the template and second pass follow, as after ``set_label``.
        Returns how many were labelled."""
        if label not in ("yes", "no", "unlabeled") or not self.n:
            return 0
        dash = self.dash
        keys = [dash.event_keys[int(i)] for i in indices if 0 <= int(i) < len(dash.event_keys)]
        if not keys:
            return 0
        for key in keys:
            if label == "unlabeled":
                dash.labels.pop(key, None)
            else:
                dash.labels[key] = label
        dash._compute_template_state()
        dash._compute_second_pass()
        dash._save_labels()
        if label == "unlabeled":
            # a cleared manual label may drop a retained sub-threshold event
            dash._rebuild_candidates(preserve_key=dash.event_keys[dash.current])
            dash._set_loaded_controls(True)
        dash._refresh_all()
        return len(keys)

    def clear_labels(self) -> int:
        """Drop every manual label of this recording and save; the auto
        rules decide every candidate again. Returns how many were dropped."""
        if not self.n:
            return 0
        labelled = [i for i in range(self.n) if self.manual_label(i) != "unlabeled"]
        return self.set_labels(labelled, "unlabeled") if labelled else 0

    def counts(self) -> tuple[int, int, int]:
        """``(yes, no, unlabeled)`` over manual labels."""
        manual = [self.manual_label(i) for i in range(self.n)]
        return manual.count("yes"), manual.count("no"), manual.count("unlabeled")

    def colors(self, indices=None) -> np.ndarray:
        """``(n, 4)`` rgba per candidate (or per ``indices``)."""
        if indices is None:
            indices = range(self.n)
        return np.array(
            [LABEL_RGBA[self.label(int(i))] for i in indices], dtype=np.float32
        ).reshape(-1, 4)

    def is_threshold_candidate(self, index: int) -> bool:
        idx = int(self.dash.candidates.indices[int(index)])
        return bool(np.isin(idx, self.dash.threshold_event_indices))

    def event_info(self, index: int) -> dict:
        """What the notebook's decision card prints for one event."""
        i = int(index)
        scores = self.dash.template_scores
        score = float(scores[i]) if len(scores) > i else float("nan")
        return {
            "index": i,
            "key": self.dash.event_keys[i],
            "label": self.manual_label(i),
            "shown": self.label(i),
            "time_s": float(self.times_s[i]),
            "amplitude": float(self.amplitudes[i]),
            "peak": float(self.peak_values[i]),
            "source": "threshold" if self.is_threshold_candidate(i) else "retained manual",
            "template_cosine": score,
            "initial_cosine": float(self.dash._initial_template_score_for_index(i)),
            "auto_call": self.dash._initial_auto_call_for_index(i),
            "pc1": float(self.dash.candidates.pca_scores[i, 0]),
            "pc1_pass": bool(self.dash._pc1_passes(i)),
            "seed": bool(i in set(self.dash.seed_indices.tolist())),
        }

    # ------------------------------------------------------------------
    # template and features
    # ------------------------------------------------------------------

    @property
    def template(self) -> np.ndarray | None:
        return self.dash.template

    @property
    def template_time_ms(self) -> np.ndarray:
        return self.dash.candidates.template_time_ms

    @property
    def template_source(self) -> np.ndarray:
        """Candidate indices averaged into the template."""
        return np.asarray(self.dash.template_source, dtype=int)

    @property
    def template_scores(self) -> np.ndarray:
        return self.dash.template_scores

    @property
    def long_time_ms(self) -> np.ndarray:
        return self.dash.candidates.long_time_ms

    def long_snippet(self, index: int) -> np.ndarray:
        return self.dash.candidates.long_snippets[int(index)]

    def short_snippet(self, index: int) -> np.ndarray:
        return self.dash.candidates.short_snippets[int(index)]

    def aligned_index(self, index: int) -> int:
        return int(self.dash.candidates.aligned_indices[int(index)])

    @property
    def pca_scores(self) -> np.ndarray:
        return self.dash.candidates.pca_scores

    @property
    def pca_explained(self) -> np.ndarray:
        return self.dash.candidates.pca_explained_variance

    @property
    def candidate_window_ms(self) -> float:
        """Half-width of the focused-candidate view."""
        return 500.0 if self.mode == "slow" else 100.0


# ----------------------------------------------------------------------
# the processed PF folder that belongs to a raw line-scan .mesc
# ----------------------------------------------------------------------

def pf_dir_for_mesc(mesc_path) -> Path | None:
    """What the voltage pipeline last left for a line scan: the newest results
    zarr beside the file, else a ``PF`` folder of pickles beside it or one
    folder up (the ``<expt>/<expt>/<expt>.mesc`` layout keeps ``<expt>/PF``);
    None when there is none."""
    mesc_path = Path(mesc_path)
    found = pf_results_in(mesc_path.parent)
    if found is not None:
        return found
    for parent in (mesc_path.parent.parent, mesc_path.parent):
        pf = parent / "PF"
        if (pf / TRACES_FILE).is_file() or pf_results_in(pf) is not None:
            return pf
    return None


def pf_scan_for_mesc(mesc_path, unit_key: str) -> PfArray | None:
    """The line scan's ``PF`` folder opened on that unit's scan (``MUnit_35``
    is scan ``35``; the image is left closed), or None when there is no
    folder or the pipeline never processed the scan."""
    pf_dir = pf_dir_for_mesc(mesc_path)
    if pf_dir is None:
        return None
    try:
        return PfArray(pf_dir, unit=unit_key, source=False)
    except ValueError:
        return None
