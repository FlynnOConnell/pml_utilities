"""Domain traces from a recording: which pixels, what is subtracted, which way up.

A domain is the pixels of one or more ROIs averaged into one trace: the lines
of a soma, a chessboard patch, the part of a patch the cell covers. An ROI is
one Z index of a 5D array; on a MESc AOD unit that is a line, patch or box
(``mesc_z_axis_meaning == "roi_index"``) cropped to its own extent
(``mesc_roi_extents``), anywhere else (a single-patch chessboard, a raster
time series, a TIFF) it is the whole frame of that plane. Pixels are chosen
per domain (:data:`PIXEL_RULES`); the scored rules rank pixels on a bounded
contiguous window, so a long recording is read twice at most. After the
pixel mean come the file's conversion offset, a background term, a reference
channel and the inversion, in that order (:class:`TraceSettings`).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from scipy.ndimage import gaussian_filter1d

from mbo_utilities.roi_workflow import as_movie

__all__ = [
    "BACKGROUNDS",
    "PIXEL_RULES",
    "REFERENCES",
    "Domain",
    "ExtractedTraces",
    "TraceSettings",
    "domains_from_json",
    "extract_traces",
    "invert",
    "roi_geometry",
]

# how a domain picks pixels inside each of its ROIs: every pixel, the top
# fraction by mean brightness, by fast fluctuation (SD after a high-pass) or by
# correlation of that fluctuation with the ROI's mean; "mask" is a given set
PIXEL_RULES = ("all", "bright", "active", "correlated", "mask")
# what is subtracted from a domain's F: nothing, the dimmest pixels of its ROIs
# per frame (a percentile), the mean of its ROIs' pixels outside its mask, or
# the F of another domain (the archive's "bg")
BACKGROUNDS = ("none", "dim", "outside", "domain")
# a second channel recorded beside the indicator (mScarlet beside ASAP7): F is
# divided by it, or the part of F it explains is removed
REFERENCES = ("none", "ratio", "regress")


@dataclass
class Domain:
    """One trace: the ``pixels`` of ``rois`` averaged with equal weight per pixel.

    ``rois`` are 0-based ROI indices (the array's Z). ``pixels`` is one of
    :data:`PIXEL_RULES`; ``fraction`` is the share of each ROI's pixels a
    scored rule keeps; ``mask`` maps an ROI to its flat ROI-local pixel indices
    (row-major over the ROI's own height x width) for ``pixels == "mask"``.
    """

    name: str
    rois: list[int]
    pixels: str = "all"
    fraction: float = 0.3
    mask: dict[int, list[int]] | None = None

    def __post_init__(self):
        self.name = str(self.name)
        self.rois = sorted({int(r) for r in self.rois})
        if not self.rois:
            raise ValueError(f"domain {self.name} has no ROIs")
        if self.pixels not in PIXEL_RULES:
            raise ValueError(
                f"domain {self.name}: pixels must be one of {PIXEL_RULES}, got {self.pixels!r}"
            )
        if not 0.0 < float(self.fraction) <= 1.0:
            raise ValueError(f"domain {self.name}: fraction must be in (0, 1]")
        if self.pixels == "mask":
            mask = {
                int(r): sorted({int(i) for i in v})
                for r, v in (self.mask or {}).items()
            }
            missing = [r for r in self.rois if not mask.get(r)]
            if missing:
                raise ValueError(
                    f"domain {self.name}: no mask pixels for ROI(s) {missing}"
                )
            self.mask = {r: mask[r] for r in self.rois}

    def to_json(self):
        """A bare ROI list for whole ROIs (the archive's ``domain_ROInumber``), else a dict."""
        if self.pixels == "all":
            return list(self.rois)
        out = {"rois": list(self.rois), "pixels": self.pixels}
        if self.pixels == "mask":
            out["mask"] = {str(r): list(v) for r, v in self.mask.items()}
        else:
            out["fraction"] = float(self.fraction)
        return out

    @classmethod
    def from_json(cls, name, value) -> Domain:
        """From a ``domains.json`` entry: an ROI list, or ``{"rois", "pixels", "fraction" | "mask"}``."""
        if isinstance(value, dict):
            return cls(
                name,
                value.get("rois") or [],
                pixels=str(value.get("pixels", "all")),
                fraction=float(value.get("fraction", 0.3)),
                mask=value.get("mask"),
            )
        return cls(name, list(value))


def domains_from_json(spec) -> list[Domain]:
    """``{name: entry}`` (see :meth:`Domain.from_json`) as domains, in order."""
    return [Domain.from_json(name, value) for name, value in dict(spec).items()]


@dataclass
class TraceSettings:
    """What happens to a domain's pixel mean before dF/F.

    ``convert`` applies the file's linear conversion (MESc's
    ``raw * scale + offset``) so zero means no photons; the archive worked
    on raw counts, which leaves its dF/F several times too small on the AOD
    rigs. ``background`` (:data:`BACKGROUNDS`) is subtracted times
    ``background_coeff``: ``dim`` is the ``background_percentile`` of an
    ROI's pixels per frame, ``outside`` the mean of the ROI pixels outside the
    domain's, ``domain`` the F of the domain named ``background_domain``
    (which then gets no trace of its own). ``reference`` (:data:`REFERENCES`)
    uses ``reference_channel`` (default the other of two) read through the
    same pixels: ``ratio`` is ``F / R * mean(R)``, ``regress`` removes the
    least-squares fit of ``R`` from ``F``. ``invert`` turns the trace upside
    down about its mean (``2 * mean - F``), pml_utilities' Invert Deflection
    on the pixels' mean, so a dimming indicator's dF/F and z both rise. The
    scored pixel rules rank pixels on ``score_seconds`` of the middle of the
    frames read, high-passed above ``score_highpass_hz``.
    """

    convert: bool = False
    background: str = "none"
    background_percentile: float = 10.0
    background_domain: str = "bg"
    background_coeff: float = 1.0
    reference: str = "none"
    reference_channel: int | None = None
    invert: bool = False
    score_seconds: float = 20.0
    score_highpass_hz: float = 5.0

    def __post_init__(self):
        if self.background not in BACKGROUNDS:
            raise ValueError(
                f"background must be one of {BACKGROUNDS}, got {self.background!r}"
            )
        if self.reference not in REFERENCES:
            raise ValueError(
                f"reference must be one of {REFERENCES}, got {self.reference!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> TraceSettings:
        d = d or {}
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


@dataclass
class ExtractedTraces:
    """Every domain's trace from one recording, rows in ``names`` order.

    ``pixel_mean`` is the domain's pixel mean (converted); ``measured`` that
    after the background and the reference, the trace in the recording's
    units; ``F`` what dF/F is taken of: ``measured``, inverted when the
    settings say so. ``background`` and ``reference`` are what was
    subtracted or divided by, per domain, when there was one. ``roi_mean`` holds every ROI
    read, whole, converted (the members' raw traces). ``masks`` gives each
    domain's flat ROI-local pixels per ROI, ``shapes`` each ROI's height and
    width, ``images`` its mean, SD and correlation images over the scoring
    window (only for ROIs a scored rule or the preview asked for).
    """

    names: list[str]
    fs: float | None
    frames: tuple[int, int]
    F: np.ndarray
    measured: np.ndarray
    pixel_mean: np.ndarray
    roi_mean: dict[int, np.ndarray]
    masks: dict[str, dict[int, np.ndarray]]
    shapes: dict[int, tuple[int, int]]
    background: np.ndarray | None = None
    reference: np.ndarray | None = None
    images: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def row(self, name: str) -> int:
        return self.names.index(name)


def roi_geometry(arr) -> dict[int, tuple[int, int, int]]:
    """``{roi: (y0, height, width)}``: where each ROI's valid pixels sit in
    the array's frame. A MESc AOD unit's lines, patches or boxes come from
    ``mesc_roi_extents`` (a ragged ROI is padded to the shared frame and, on
    a Y-flipped layout, lands at the bottom); any other array's ROIs are its
    Z planes, each the whole frame.
    """
    md = getattr(arr, "metadata", None) or {}
    extents = md.get("mesc_roi_extents") or []
    if extents and md.get("mesc_z_axis_meaning") == "roi_index":
        return {
            int(e["index"]): (
                int(e.get("y_start", 0)),
                int(e["height"]),
                int(e["width"]),
            )
            for e in extents
        }
    _, _, nz, ny, nx = (int(s) for s in arr.shape)
    return {z: (0, ny, nx) for z in range(nz)}


def invert(F: np.ndarray) -> np.ndarray:
    """``2 * mean - F`` along the last axis: the Invert Deflection of a pixel mean."""
    F = np.asarray(F, dtype=np.float64)
    return 2.0 * F.mean(axis=-1, keepdims=True) - F


def _conversion(arr, channel: int) -> tuple[float, float]:
    conv = (getattr(arr, "metadata", None) or {}).get("mesc_channel_conversion") or []
    if channel < len(conv):
        return float(conv[channel]["scale"]), float(conv[channel]["offset"])
    return 1.0, 0.0


def _top(score: np.ndarray, fraction: float) -> np.ndarray:
    """Flat indices of the top ``fraction`` of ``score``, at least one."""
    flat = np.nan_to_num(score.ravel(), nan=-np.inf)
    k = max(1, int(round(fraction * flat.size)))
    return np.sort(np.argsort(flat, kind="stable")[::-1][:k])


def _score_images(movie, frames, h, w, fs, settings) -> dict[str, np.ndarray]:
    """Mean, high-passed SD and correlation-with-the-ROI-mean images of one
    ROI over the middle ``score_seconds`` of ``frames``.
    """
    t0, t1 = frames
    n = t1 - t0
    if fs:
        n = min(n, max(16, int(round(settings.score_seconds * fs))))
    s0 = t0 + (t1 - t0 - n) // 2
    blk = movie.frames(s0, s0 + n, slice(0, h), slice(0, w)).astype(np.float64)
    mean = blk.mean(axis=0)
    if fs and settings.score_highpass_hz > 0 and n > 8:
        sigma = fs / (2.0 * np.pi * float(settings.score_highpass_hz))
        blk = blk - gaussian_filter1d(blk, sigma=sigma, axis=0)
    blk -= blk.mean(axis=0)
    sd = blk.std(axis=0)
    trace = blk.reshape(n, -1).mean(axis=1)
    denom = sd * trace.std()
    corr = np.zeros_like(mean)
    np.divide(
        np.tensordot(trace, blk, axes=(0, 0)) / n, denom, out=corr, where=denom > 0
    )
    return {"mean": mean, "sd": sd, "corr": corr}


def extract_traces(
    arr,
    domains,
    settings: TraceSettings | None = None,
    *,
    channel: int = 0,
    frames=None,
    batch_size: int = 20000,
    images: bool = False,
    progress=None,
) -> ExtractedTraces:
    """Every domain's trace from ``arr`` (a 5D array), one pass per ROI.

    ``domains`` are :class:`Domain` objects or ``{name: entry}`` (see
    :meth:`Domain.from_json`); ``frames=(start, stop)`` reads that half-open
    window, default every frame. ``images`` computes the scoring images of
    every ROI read, not only those a scored rule needs (the GUI preview).
    ``progress(i, n, seconds)`` is called after each ROI.
    """
    settings = settings or TraceSettings()
    if isinstance(domains, dict):
        domains = domains_from_json(domains)
    domains = list(domains)
    if not domains:
        raise ValueError("no domains")
    geometry = roi_geometry(arr)
    for d in domains:
        bad = [r for r in d.rois if r not in geometry]
        if bad:
            raise ValueError(
                f"domain {d.name}: ROI {bad[0]} is outside 0..{len(geometry) - 1}"
            )
    background_of = None
    if settings.background == "domain":
        background_of = next(
            (d for d in domains if d.name == settings.background_domain), None
        )
        if background_of is None:
            raise ValueError(
                f"background is domain {settings.background_domain!r}, which is not among the domains"
            )
    traced = [d for d in domains if d is not background_of]
    if not traced:
        raise ValueError("every domain is the background")
    nt, nc = int(arr.shape[0]), int(arr.shape[1])
    t0, t1 = (0, nt) if frames is None else (int(frames[0]), int(frames[1]))
    if not 0 <= t0 < t1 <= nt:
        raise ValueError(f"frames ({t0}, {t1}) outside 0..{nt}")
    if not 0 <= channel < nc:
        raise ValueError(f"channel {channel} outside 0..{nc - 1}")
    ref_channel = None
    if settings.reference != "none":
        ref_channel = settings.reference_channel
        if ref_channel is None:
            if nc != 2:
                raise ValueError(f"name the reference channel: the recording has {nc}")
            ref_channel = 1 - channel
        ref_channel = int(ref_channel)
        if ref_channel == channel or not 0 <= ref_channel < nc:
            raise ValueError(
                f"reference channel {ref_channel} must be another of the {nc} channels"
            )
    fs = getattr(arr, "fs", None)
    fs = float(fs) if fs else None
    n_frames = t1 - t0
    rois = sorted({r for d in domains for r in d.rois})
    shapes = {r: (geometry[r][1], geometry[r][2]) for r in rois}

    # each domain sums its pixels in every ROI it has; count is its pixel total
    num = np.zeros((len(domains), n_frames))
    ref_num = np.zeros((len(domains), n_frames)) if ref_channel is not None else None
    bg_num = np.zeros((len(domains), n_frames))
    bg_weight = np.zeros(len(domains))
    count = np.zeros(len(domains))
    roi_mean, masks, imgs = {}, {d.name: {} for d in domains}, {}
    started = time.time()
    for i, r in enumerate(rois):
        y0, h, w = geometry[r]
        rows = slice(y0, y0 + h)
        movie = as_movie(arr, z=r, c=channel)
        users = [k for k, d in enumerate(domains) if r in d.rois]
        scored = images or any(
            domains[k].pixels in ("bright", "active", "correlated") for k in users
        )
        if scored:
            imgs[r] = _score_images(
                movie.crop(y0, y0 + h, 0, w), (t0, t1), h, w, fs, settings
            )
        weights = np.zeros((h * w, len(users) + 1))
        outside = np.zeros((h * w, len(users)))
        weights[:, -1] = 1.0
        for j, k in enumerate(users):
            d = domains[k]
            if d.pixels == "all":
                idx = np.arange(h * w)
            elif d.pixels == "mask":
                idx = np.asarray(d.mask[r], dtype=np.int64)
                if idx.min() < 0 or idx.max() >= h * w:
                    raise ValueError(
                        f"domain {d.name}: mask pixels outside ROI {r}'s {h}x{w}"
                    )
            else:
                key = {"bright": "mean", "active": "sd", "correlated": "corr"}[d.pixels]
                idx = _top(imgs[r][key], d.fraction)
            masks[d.name][r] = idx
            weights[idx, j] = 1.0
            count[k] += idx.size
            if settings.background == "outside":
                # a domain covering its whole ROI has nothing outside it there
                rest = np.setdiff1d(np.arange(h * w), idx)
                if rest.size:
                    outside[rest, j] = 1.0 / rest.size
                    bg_weight[k] += idx.size
        ref_movie = (
            as_movie(arr, z=r, c=ref_channel) if ref_channel is not None else None
        )
        for b0 in range(t0, t1, batch_size):
            b1 = min(t1, b0 + batch_size)
            blk = movie.frames(b0, b1, rows, slice(0, w)).astype(np.float64)
            blk = blk.reshape(blk.shape[0], -1)
            sums = blk @ weights
            num[users, b0 - t0 : b1 - t0] += sums[:, :-1].T
            roi_mean.setdefault(r, np.zeros(n_frames))[b0 - t0 : b1 - t0] = sums[
                :, -1
            ] / (h * w)
            if settings.background == "dim":
                low = np.percentile(blk, settings.background_percentile, axis=1)
                for k in users:
                    bg_num[k, b0 - t0 : b1 - t0] += low * masks[domains[k].name][r].size
            elif settings.background == "outside":
                out = blk @ outside
                for j, k in enumerate(users):
                    bg_num[k, b0 - t0 : b1 - t0] += (
                        out[:, j] * masks[domains[k].name][r].size
                    )
            if ref_movie is not None:
                ref = ref_movie.frames(b0, b1, rows, slice(0, w)).astype(np.float64)
                ref_num[users, b0 - t0 : b1 - t0] += (
                    ref.reshape(ref.shape[0], -1) @ weights[:, :-1]
                ).T
        if progress is not None:
            progress(i, len(rois), time.time() - started)

    scale, offset = _conversion(arr, channel) if settings.convert else (1.0, 0.0)
    pixel = num / count[:, None] * scale + offset
    roi_mean = {r: v * scale + offset for r, v in roi_mean.items()}
    background = None
    if settings.background == "dim":
        background = bg_num / count[:, None] * scale + offset
    elif settings.background == "outside":
        background = np.zeros_like(bg_num)
        some = bg_weight > 0
        background[some] = bg_num[some] / bg_weight[some, None] * scale + offset
    elif settings.background == "domain":
        background = np.repeat(
            pixel[[domains.index(background_of)]], len(domains), axis=0
        )
    keep = [domains.index(d) for d in traced]
    pixel = pixel[keep]
    F = pixel.copy()
    if background is not None:
        background = background[keep]
        F = F - float(settings.background_coeff) * background
    reference = None
    if ref_num is not None:
        rs, ro = _conversion(arr, ref_channel) if settings.convert else (1.0, 0.0)
        reference = (ref_num / count[:, None] * rs + ro)[keep]
        if settings.reference == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                F = F / reference * reference.mean(axis=1, keepdims=True)
        else:
            rc = reference - reference.mean(axis=1, keepdims=True)
            fc = F - F.mean(axis=1, keepdims=True)
            beta = (rc * fc).sum(axis=1, keepdims=True) / np.maximum(
                (rc * rc).sum(axis=1, keepdims=True), 1e-12
            )
            F = F - beta * rc
    measured = F
    if settings.invert:
        F = invert(measured)
    info = {
        "channel": int(channel),
        "reference_channel": ref_channel,
        "conversion": [scale, offset],
        "pixels": {
            d.name: {int(r): int(v.size) for r, v in masks[d.name].items()}
            for d in domains
        },
        "seconds": round(time.time() - started, 3),
    }
    if background_of is not None:
        info["background_domain"] = background_of.name
    if settings.background == "outside":
        info["no_background"] = [
            d.name for k, d in enumerate(domains) if bg_weight[k] == 0
        ]
    return ExtractedTraces(
        names=[d.name for d in traced],
        fs=fs,
        frames=(t0, t1),
        F=F,
        measured=measured,
        pixel_mean=pixel,
        roi_mean=roi_mean,
        masks=masks,
        shapes=shapes,
        background=background,
        reference=reference,
        images=imgs,
        info=info,
    )
