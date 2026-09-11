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
from vnoiser.dataset import RecordingSample, SpatialJediDataset

__all__ = [
    "LABEL_RGBA",
    "MODES",
    "PC1_SIDES",
    "CurationSession",
    "PfScan",
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
    """Curation of one mode over one data path.

    Parameters
    ----------
    data_path : str or Path
        A vnoiser ``Data`` folder, animal folder, experiment / ``PF`` folder,
        or a raw ``.mat`` recording.
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
        self.experiment: str = ""
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

    @property
    def dataset(self):
        return self.dash.dataset

    @property
    def has_dataset(self) -> bool:
        return self.dash.dataset is not None

    @property
    def hierarchical(self) -> bool:
        """Whether an animal and experiment must be chosen before a recording."""
        return (
            isinstance(self.dash.dataset, SpatialJediDataset)
            and self.dash.dataset.requires_experiment_selection
        )

    @property
    def animals(self) -> list[tuple[str, str]]:
        """``(label, path)`` per animal folder, for a hierarchical data path."""
        if not self.hierarchical:
            return []
        return list(self.dash.dataset.animal_options())

    def experiments(self, animal: str) -> list[tuple[str, str]]:
        """``(label, path)`` per experiment folder under ``animal``."""
        if not self.hierarchical or not animal:
            return []
        return list(self.dash.dataset.experiment_options(animal))

    def select_experiment(self, experiment: str) -> str:
        """Read one experiment's scan metadata; returns the status line."""
        self.experiment = str(experiment)
        self.recording_id = ""
        self.dash._experiment_changed({"new": self.experiment})
        return self.status

    def scan(self, data_path) -> str:
        """Point the session at another path; returns the status line."""
        self.dash.path_text.value = str(Path(data_path).expanduser())
        self.dash._scan_data_path(None)
        self.experiment = ""
        self.recording_id = ""
        return self.status

    @property
    def recordings(self) -> list[tuple[str, str]]:
        """``(label, id)`` per loadable recording."""
        return [(label, value) for label, value in self.dash._recording_options() if value]

    def load(self, recording_id: str) -> str:
        """Load one recording and run or restore its pipeline; returns the
        status line. Raises when vnoiser cannot load it."""
        options = self.dash._recording_options()
        if recording_id not in {value for _, value in options}:
            raise KeyError(f"unknown recording: {recording_id}")
        self.dash.recording_dropdown.options = options
        self.dash.recording_dropdown.value = recording_id
        self.dash._load_selected_recording(None)
        self.recording_id = recording_id
        return self.status

    def load_trace(
        self,
        trace,
        fs_hz: float,
        *,
        recording_id: str,
        label: str,
        source_path,
        curation_dir=None,
    ) -> str:
        """Curate a trace held in memory: an ROI trace pulled from a line
        scan, say. vnoiser's denoiser runs on it (the same pipeline a raw
        ``.mat`` recording gets), cached under ``curation_dir/cache`` by
        ``recording_id`` and the source file's size and mtime. Labels go to
        ``curation_dir/<mode>_template_curation.json`` keyed by
        ``recording_id``. Returns the status line."""
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
                "pre_denoised": False,
                "source_format": "trace",
            },
        )
        dash = self.dash
        recording = dash._window_from_recording(full)
        dash._activate_recording_storage(recording)
        cache_path = self._trace_cache_path(recording)
        if cache_path is not None and cache_path.exists():
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
        """``(low, high, step)`` for the auto-pass amplitude.

        The notebook's A2 slider shares the threshold slider's range, whose
        floor is the trace median. The amplitude it is compared with is the
        candidate's baseline-subtracted peak, which can sit below that floor
        (a fast-mode candidate near the noise whose peak is under its own
        preceding baseline), so the notebook's slider at its bottom still
        leaves those rejected. Here the floor drops under the lowest
        amplitude, so A2 all the way down passes every candidate.
        """
        s = self.dash.auto_pass_slider
        lo, hi, step = float(s.min), float(s.max), float(s.step)
        if self.loaded and self.n:
            lowest = float(np.nanmin(self.amplitudes))
            if np.isfinite(lowest):
                lo = min(lo, lowest - step)
        return lo, hi, step

    def auto_pass_count(self) -> int:
        """How many candidates the auto-pass amplitude passes now."""
        if not self.loaded or self.auto_pass is None or not self.n:
            return 0
        return int(np.count_nonzero(self.amplitudes >= float(self.auto_pass)))

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
        return self.dash.candidates.amplitudes

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

PF_TRACES = SpatialJediDataset.trace_filename


def pf_dir_for_mesc(mesc_path) -> Path | None:
    """The experiment's ``PF`` folder for a raw line scan laid out as
    ``<animal>/<experiment>/<experiment>/<experiment>.mesc`` with the
    processed traces in ``<animal>/<experiment>/PF``; None when absent."""
    mesc_path = Path(mesc_path)
    for parent in (mesc_path.parent.parent, mesc_path.parent):
        pf = parent / "PF"
        if (pf / PF_TRACES).is_file():
            return pf
    return None


class PfScan:
    """The processed traces of one line-scan unit: the PF scan whose id is
    the unit's number (``MUnit_35`` is scan ``35``), its domains (groups of
    lines the pipeline averaged) and which line ROIs each holds.

    Parameters
    ----------
    pf_dir : Path
        The experiment's ``PF`` folder.
    scan_id : str
        Scan id as the pipeline keys it.
    domains : dict
        ``{domain: [roi, ...]}`` from ``scanIDs_ROIs.pkl``.
    """

    def __init__(self, pf_dir: Path, scan_id: str, domains: dict[str, list[int]]):
        self.pf_dir = Path(pf_dir)
        self.scan_id = str(scan_id)
        self.domains = {str(k): [int(v) for v in rois] for k, rois in domains.items()}
        self.experiment = self.pf_dir.parent.name
        self.animal = self.pf_dir.parent.parent.name

    def domain_for_roi(self, roi: int) -> str | None:
        """The domain that averages line ``roi``, or None."""
        for domain, rois in self.domains.items():
            if int(roi) in rois:
                return domain
        return None

    def recording_id(self, domain: str) -> str:
        """The dataset's id for a domain trace of this scan."""
        return f"{self.animal}/{self.experiment}/scan={self.scan_id}/domain={domain}"


def pf_scan_for_mesc(mesc_path, unit_key: str) -> PfScan | None:
    """The :class:`PfScan` of a line-scan unit, or None when the PF folder
    is missing or the pipeline never processed that scan."""
    import pickle

    pf_dir = pf_dir_for_mesc(mesc_path)
    if pf_dir is None:
        return None
    munit = str(unit_key).rsplit("/", 1)[-1]
    scan_id = munit.rsplit("_", 1)[-1]
    fs_path, roi_path = pf_dir / "fs_scans.pkl", pf_dir / "scanIDs_ROIs.pkl"
    if not fs_path.is_file() or not roi_path.is_file():
        return None
    with fs_path.open("rb") as handle:
        fs_by_scan = {str(k): v for k, v in pickle.load(handle).items()}
    if scan_id not in fs_by_scan:
        return None
    with roi_path.open("rb") as handle:
        scan_metadata = pickle.load(handle)
    domain_map = scan_metadata.get("domain_ROInumber", {})
    domains = {
        str(name): list(rois)
        for name, rois in domain_map.items()
        if str(name) != "All_domains"
    }
    return PfScan(pf_dir, scan_id, domains)
