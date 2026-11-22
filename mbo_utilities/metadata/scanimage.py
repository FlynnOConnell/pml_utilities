"""Acquisition facts read from a ScanImage header (``metadata["si"]``): stack type, channels, slices, rates and where each frame sits among the pages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

StackType = Literal["lbm", "piezo", "pollen", "single_plane"]

MAX_PMT_CHANNELS = 4  # a non-vDAQ ScanImage rig digitizes at most four PMT channels


def _si(metadata: dict, module: str) -> dict:
    return (metadata.get("si") or {}).get(module) or {}


def _number(value) -> float | None:
    """A header scalar, or None for the empty lists ScanImage writes for unset values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _channel_save(metadata: dict) -> list[int]:
    """``hChannels.channelSave`` as a flat list of 1-based channel ids."""
    saved = _si(metadata, "hChannels").get("channelSave")
    if saved is None:
        return []
    if isinstance(saved, (int, float)):
        return [int(saved)]
    flat = []
    for x in saved:
        if isinstance(x, list):
            if len(x) != 1:
                raise NotImplementedError(f"channelSave entry {x!r} has several elements")
            x = x[0]
        flat.append(int(x))
    return flat


def is_fast_z(metadata: dict) -> bool:
    """True when the volume was scanned with the fast-z actuator, one frame per slice."""
    return (
        _si(metadata, "hStackManager").get("stackMode") == "fast"
        or _si(metadata, "hFastZ").get("enable") is True
    )


def detect_stack_type(metadata: dict) -> StackType:
    """``lbm`` when ports carry several beamlets, ``piezo`` for a slice stack, ``pollen`` for both, else ``single_plane``.

    A stack is ``hStackManager.enable``, or ``hFastZ.enable`` with more than one
    slice (ScanImage before 2020 left the stack manager off for fast-z volumes).
    """
    if not metadata.get("si"):
        return "single_plane"
    stack = _si(metadata, "hStackManager")
    is_piezo = stack.get("enable") is True or (
        _si(metadata, "hFastZ").get("enable") is True
        and (_number(stack.get("numSlices")) or 1) > 1
    )
    is_lbm = get_num_zplanes_from_channels(metadata) > 1
    if is_lbm and is_piezo:
        return "pollen"
    if is_lbm:
        return "lbm"
    return "piezo" if is_piezo else "single_plane"


def is_lbm_stack(metadata: dict) -> bool:
    return detect_stack_type(metadata) in ("lbm", "pollen")


def is_piezo_stack(metadata: dict) -> bool:
    return detect_stack_type(metadata) in ("piezo", "pollen")


def get_saved_channel_ports(metadata: dict) -> dict[str, list[int]]:
    """AI port -> its saved channel ids, from ``hScan2D.virtualChannelSettings__N.source``.

    Dual-color single plane: ``{"AI0": [1], "AI1": [2]}``; dual-color 30-beam
    LBM: ``{"AI0": [1..15], "AI1": [16..30]}``. Empty without virtual channels.
    """
    saved = set(_channel_save(metadata)) or None
    sources: dict[str, list[int]] = {}
    for key, val in _si(metadata, "hScan2D").items():
        if not (key.startswith("virtualChannelSettings__") and isinstance(val, dict)):
            continue
        src = val.get("source")
        index = key.split("__")[1]
        if not src or not index.isdigit():
            continue
        if saved is None or int(index) in saved:
            sources.setdefault(src, []).append(int(index))
    return {port: sorted(ids) for port, ids in sources.items()}


def get_color_channel_ports(metadata: dict) -> list[str]:
    return sorted(get_saved_channel_ports(metadata))


def get_beamlets_per_port(metadata: dict) -> dict[str, int]:
    return {p: len(ch) for p, ch in get_saved_channel_ports(metadata).items()}


def get_num_zplanes_from_channels(metadata: dict) -> int:
    """Beamlets per color: the saved channels per AI port, or every saved channel past what a PMT rig can digitize.

    Without virtual channel settings a rig is not a vDAQ, so up to
    ``MAX_PMT_CHANNELS`` saved channels are PMT colors, never beamlets.
    """
    saved = _channel_save(metadata)
    per_port = get_beamlets_per_port(metadata)
    if per_port and sum(per_port.values()) >= len(saved):
        return max(per_port.values())
    if len(saved) > MAX_PMT_CHANNELS:
        return len(saved) // get_num_color_channels(metadata)
    return 1


def get_num_color_channels(metadata: dict) -> int:
    """Color channels: AI ports among the saved channels, else (no virtual channels, PMT rig) the saved channel count."""
    ports = get_color_channel_ports(metadata)
    if ports:
        return len(ports)
    saved = _channel_save(metadata)
    return len(saved) if 0 < len(saved) <= MAX_PMT_CHANNELS else 1


def get_num_zplanes(metadata: dict) -> int:
    """Z-planes: beamlets per color for LBM and pollen, slices for a piezo stack, else 1."""
    stack_type = detect_stack_type(metadata)
    if stack_type in ("lbm", "pollen"):
        return get_num_zplanes_from_channels(metadata)
    if stack_type == "piezo":
        return get_num_slices(metadata)
    return 1


def get_frames_per_slice(metadata: dict) -> int:
    """Frames acquired at each slice: ``hStackManager.framesPerSlice``, 1 for fast-z (which records ``Inf``).

    ``hScan2D.logFramesPerSlice`` is often None and is not read.
    """
    if is_fast_z(metadata):
        return 1
    frames = _number(_si(metadata, "hStackManager").get("framesPerSlice"))
    return int(frames) if frames and math.isfinite(frames) else 1


def get_log_average_factor(metadata: dict) -> int:
    """``hScan2D.logAverageFactor``: above 1, that many frames were averaged into each saved page."""
    return int(_number(_si(metadata, "hScan2D").get("logAverageFactor")) or 1)


def get_z_step_size(metadata: dict) -> float | None:
    """The stack's z-step in µm (``actualStackZStepSize``, else ``stackZStepSize``), None when no stack ran.

    LBM beamlet spacing is not in the header; the user supplies it.
    """
    if not is_piezo_stack(metadata):
        return None
    stack = _si(metadata, "hStackManager")
    dz = _number(stack.get("actualStackZStepSize"))
    if dz is None:
        dz = _number(stack.get("stackZStepSize"))
    return float(dz) if dz else None


def get_num_volumes(metadata: dict) -> int | None:
    """The volumes requested (``actualNumVolumes``, else ``numVolumes``); an aborted run saved fewer, ``ScanLayout`` counts those."""
    if not is_piezo_stack(metadata):
        return None
    stack = _si(metadata, "hStackManager")
    n = _number(stack.get("actualNumVolumes")) or _number(stack.get("numVolumes"))
    return int(n) if n else None


def get_num_slices(metadata: dict) -> int:
    """Slices per volume of a stack (``actualNumSlices``, else ``numSlices``); 1 without one."""
    if not is_piezo_stack(metadata):
        return 1
    stack = _si(metadata, "hStackManager")
    n = _number(stack.get("actualNumSlices")) or _number(stack.get("numSlices"))
    return int(n) if n else 1


def get_frames_per_volume(metadata: dict) -> int | None:
    """``hStackManager.numFramesPerVolume`` of a stack, None without one."""
    if not is_piezo_stack(metadata):
        return None
    n = _number(_si(metadata, "hStackManager").get("numFramesPerVolume"))
    return int(n) if n else None


def get_flyback_frames(metadata: dict) -> int:
    """Frames saved at the end of each fast-z volume while the actuator flies back.

    ``numFramesPerVolumeWithFlyback - numFramesPerVolume``; 0 when
    ``hFastZ.discardFlybackFrames`` kept them out of the file.
    """
    if not is_fast_z(metadata) or _si(metadata, "hFastZ").get("discardFlybackFrames") is True:
        return 0
    stack = _si(metadata, "hStackManager")
    with_flyback = _number(stack.get("numFramesPerVolumeWithFlyback"))
    without = _number(stack.get("numFramesPerVolume"))
    if with_flyback is None or without is None:
        return 0
    return max(int(with_flyback) - int(without), 0)


def rois_form_one_row(roi_groups: list[dict]) -> bool:
    """
    True when the mROI scanfields sit side by side left to right, each touching the next, as the strips of one field.

    Parameters
    ----------
    roi_groups : list[dict]
        ScanImage ``RoiGroups.imagingRoiGroup.rois``, in page order.
    """
    fields = []
    for roi in roi_groups:
        scanfields = roi["scanfields"]
        fields.append(scanfields[0] if isinstance(scanfields, list) else scanfields)
    centers = np.array([f["centerXY"] for f in fields], dtype=float)  ## (n_rois, 2) in scan degrees
    sizes = np.array([f["sizeXY"] for f in fields], dtype=float)
    tolerance = 0.01 * sizes[:, 0].min()
    same_row = np.all(np.abs(centers[:, 1] - centers[0, 1]) <= tolerance)
    gaps = np.diff(centers[:, 0]) - (sizes[:-1, 0] + sizes[1:, 0]) / 2
    return bool(same_row and np.all(np.abs(gaps) <= tolerance))


def get_roi_info(metadata: dict) -> dict:
    """``num_mrois``, plus ``roi`` (one strip's width, height) and ``fov`` (every strip side by side) in px when the header has them."""
    roi_mgr = _si(metadata, "hRoiManager")
    num_mrois = metadata.get("num_rois") or metadata.get("num_mrois")
    if num_mrois is None and isinstance(metadata.get("roi_groups"), list):
        num_mrois = len(metadata["roi_groups"])
    if num_mrois is None:
        rois = (roi_mgr.get("roiGroup") or {}).get("rois") if roi_mgr.get("mroiEnable") else None
        num_mrois = len(rois) if isinstance(rois, list) else 1

    result = {"num_mrois": num_mrois}
    width, height = roi_mgr.get("pixelsPerLine"), roi_mgr.get("linesPerFrame")
    if width is not None and height is not None:
        result["roi"] = (width, height)
        result["fov"] = (num_mrois * width, height)
    return result


def get_frame_rate(metadata: dict) -> float | None:
    """The scanner's frame rate in Hz (``scanFrameRate``, else ``1 / scanFramePeriod``), unrounded."""
    roi_mgr = _si(metadata, "hRoiManager")
    fs = _number(roi_mgr.get("scanFrameRate"))
    if fs:
        return float(fs)
    period = _number(roi_mgr.get("scanFramePeriod"))
    return 1.0 / float(period) if period else None


def get_volume_rate(metadata: dict) -> float | None:
    """Volumes per second of a stack: ``scanVolumeRate``, else the frame rate over the frames of one volume."""
    rate = _number(_si(metadata, "hRoiManager").get("scanVolumeRate"))
    if rate:
        return float(rate)
    fs = get_frame_rate(metadata)
    frames = get_num_slices(metadata) * get_frames_per_slice(metadata) + get_flyback_frames(metadata)
    return fs / frames if fs else None


@dataclass(frozen=True)
class ScanLayout:
    """Where every (t, c, z) of a raw ScanImage acquisition sits among its pages.

    Pages run frame by frame, ``colors * beamlets`` pages per frame in
    ``channelSave`` order (beamlets within a color). Frames run volume by
    volume: ``slices`` slices of ``reps`` saved frames each, then ``flyback``
    frames nobody reads. Z holds the beamlets when there are several (LBM,
    pollen) and the slices otherwise; slices then move onto T, volume-major.
    Repeats go on T after their slice, or are averaged away.
    """

    stack_type: StackType
    colors: int
    beamlets: int
    slices: int
    reps: int
    flyback: int
    volumes: int

    @property
    def pages_per_frame(self) -> int:
        return self.colors * self.beamlets

    @property
    def frames_per_volume(self) -> int:
        return self.slices * self.reps + self.flyback

    @property
    def slices_on_z(self) -> bool:
        return self.beamlets == 1

    def shape(self, average: bool) -> tuple[int, int, int]:
        """(T, C, Z), with the repeats of each slice averaged or kept on T."""
        per_volume = (1 if self.slices_on_z else self.slices) * (1 if average else self.reps)
        z = self.slices if self.slices_on_z else self.beamlets
        return self.volumes * per_volume, self.colors, z

    def pages(self, t, c, z, average: bool) -> np.ndarray:
        """Raw page of every requested (t, c, z); shape (T, R, C, Z) with R the averaged repeats (1 when not averaging)."""
        t = np.asarray(t, dtype=np.int64)[:, None, None, None]
        c = np.asarray(c, dtype=np.int64)[None, None, :, None]
        z = np.asarray(z, dtype=np.int64)[None, None, None, :]
        if average:
            rep = np.arange(self.reps)[None, :, None, None]
        else:
            t, rep = np.divmod(t, self.reps)
        if self.slices_on_z:
            volume, slice_, beam = t, z, 0
        else:
            (volume, slice_), beam = np.divmod(t, self.slices), z
        frame = volume * self.frames_per_volume + slice_ * self.reps + rep
        return frame * self.pages_per_frame + c * self.beamlets + beam


def scan_layout(metadata: dict, num_pages: int) -> ScanLayout:
    """The layout of ``num_pages`` saved pages of the acquisition ``metadata`` describes.

    Volumes are counted from the pages, never from ``numVolumes``, which is what
    was requested; a trailing partial volume is left out.
    """
    stack_type = detect_stack_type(metadata)
    colors = get_num_color_channels(metadata)
    beamlets = get_num_zplanes_from_channels(metadata)
    slices = get_num_slices(metadata)
    reps = max(get_frames_per_slice(metadata) // get_log_average_factor(metadata), 1)
    flyback = get_flyback_frames(metadata)
    frames = num_pages // (colors * beamlets)
    return ScanLayout(
        stack_type=stack_type,
        colors=colors,
        beamlets=beamlets,
        slices=slices,
        reps=reps,
        flyback=flyback,
        volumes=frames // (slices * reps + flyback),
    )


def layout_metadata(metadata: dict, layout: ScanLayout, average: bool) -> dict:
    """The canonical counts, ``fs`` and ``dz`` of the (T, C, Z) ``layout`` presents; a key the axes give no value is absent.

    T samples frames (single plane, LBM), volumes (piezo; repeats kept on T
    make it uneven unless there is one volume) or slices (pollen, not time).
    ``dz`` is the stack's step when Z holds slices; beamlet spacing is unknown.
    """
    t, c, z = layout.shape(average)
    md = {"num_timepoints": t, "num_color_channels": c, "num_zplanes": z}
    if layout.stack_type in ("single_plane", "lbm"):
        md["fs"] = get_frame_rate(metadata)
    elif layout.stack_type == "piezo":
        md["dz"] = get_z_step_size(metadata)
        if average or layout.reps == 1:
            md["fs"] = get_volume_rate(metadata)
        elif layout.volumes == 1:
            md["fs"] = get_frame_rate(metadata)
    return {k: v for k, v in md.items() if v is not None}


def extract_roi_slices(metadata: dict) -> list[dict]:
    """Each mROI strip's rows in a page: ``y_start``, ``y_end``, ``height``, ``width``, ``x`` (0) and ``slice``.

    Strips are stacked down the page with fly-to lines between them. When the
    strips' ``pixelResolutionXY`` heights and a whole number of fly-to lines fill
    the page exactly, those are the boundaries; otherwise the heights are scaled
    to fit around ``num_fly_to_lines`` (which comes from a rounded timing).
    """
    roi_groups = metadata.get("roi_groups") or []
    if isinstance(roi_groups, dict):
        roi_groups = [roi_groups]
    page_width = metadata.get("page_width")
    page_height = metadata.get("page_height")
    if not roi_groups or page_width is None or page_height is None:
        return []

    heights = []
    for roi in roi_groups:
        scanfields = roi.get("scanfields")
        if isinstance(scanfields, list):
            scanfields = scanfields[0]
        pixel_res = (scanfields or {}).get("pixelResolutionXY")
        if pixel_res and len(pixel_res) >= 2:
            heights.append(int(pixel_res[1]))
    if not heights:
        return []

    gaps = len(heights) - 1
    spare = page_height - sum(heights)
    if gaps and spare >= 0 and spare % gaps == 0:
        fly_to = spare // gaps
    elif not gaps and spare == 0:
        fly_to = 0
    else:
        fly_to = int(metadata.get("num_fly_to_lines", 0))
        available = page_height - gaps * fly_to
        scaled = [round(h * available / sum(heights)) for h in heights[:-1]]
        heights = [*scaled, available - sum(scaled)]

    rois, y = [], 0
    for height in heights:
        rois.append(
            {
                "y_start": y,
                "y_end": y + height,
                "width": page_width,
                "height": height,
                "x": 0,
                "slice": slice(y, y + height),
            }
        )
        y += height + fly_to
    return rois
