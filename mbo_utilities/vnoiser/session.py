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

import re
from pathlib import Path

import numpy as np
from vnoiser.curation import (
    AUTO_TEMPLATE_THRESHOLD,
    LABEL_COLORS,
    EventCurationDashboard,
)
from vnoiser.dataset import SpatialJediDataset

__all__ = ["LABEL_RGBA", "MODES", "CurationSession", "hex_rgba"]

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
        s = self.dash.auto_pass_slider
        return float(s.min), float(s.max), float(s.step)

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

    @property
    def auto_template_threshold(self) -> float:
        return float(AUTO_TEMPLATE_THRESHOLD)

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
