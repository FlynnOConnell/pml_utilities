"""BehaviorMate's ``.tdml`` log as a ``Behavior``.

One JSON object per line. ``time`` is seconds since the session started on
BehaviorMate's clock. ``position_controller`` lines carry the treadmill
position ``y`` in mm along the track, which wraps at ``track_length`` every
lap. ``behavior_controller`` lines carry licks (``lick`` start and stop on a
sensor), valves (``valve`` open and close on a pin: the reward valves, and
the ``sync_pin`` whose pulse triggers the microscope) and contexts
(``context`` start and stop by id, a reward zone). ``lap`` lines count laps.
The ``settings`` line names the sync pin and the track length; the line
with ``mouse`` and ``start`` names the subject and the wall-clock start.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# faster than any mouse on a treadmill (5 m/s): a position jump, not a run
MAX_SPEED_MM_S = 5000.0
# the speed is averaged over this long
SPEED_WINDOW_S = 0.1


def read_tdml(path):
    """The session in ``path`` as a ``Behavior``: ``position`` (mm) and
    ``speed`` (mm/s over ``SPEED_WINDOW_S``; a wrap at the track's end is
    unwrapped, a jump faster than ``MAX_SPEED_MM_S`` such as the reset to
    zero after a lap is NaN) signals; ``lick``, ``reward`` (any valve but the sync
    pin opening) and ``lap`` events; one epoch per context id; times shifted
    so the first sync pulse is 0.
    """
    from mbo_utilities.behavior import Behavior, BehaviorSignal

    path = Path(path)
    settings, session, version = {}, {}, ""
    pos_t, pos_y, licks, laps = [], [], [], []
    valve_open = defaultdict(list)
    contexts = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("time")
            if "settings" in rec:
                settings = rec["settings"]
            if "version" in rec:
                version = str(rec["version"])
            if "mouse" in rec:
                session = rec
            if t is None:
                continue
            if "y" in rec:
                pos_t.append(float(t))
                pos_y.append(float(rec["y"]))
            if "lap" in rec:
                laps.append(float(t))
            bc = rec.get("behavior_controller")
            if not isinstance(bc, dict):
                continue
            lick = bc.get("lick")
            if isinstance(lick, dict) and lick.get("action") == "start":
                licks.append(float(t))
            valve = bc.get("valve")
            if isinstance(valve, dict) and valve.get("action") == "open":
                valve_open[valve.get("pin")].append(float(t))
            context = bc.get("context")
            if isinstance(context, dict):
                contexts[str(context.get("id"))].append(
                    (str(context.get("action")), float(t))
                )
    sync_pin = settings.get("sync_pin")
    sync = np.asarray(valve_open.pop(sync_pin, []), dtype=np.float64)
    offset = float(sync[0]) if len(sync) else 0.0
    signals = {}
    if pos_t:
        t = np.asarray(pos_t, dtype=np.float64) - offset
        y = np.asarray(pos_y, dtype=np.float64)
        signals["position"] = BehaviorSignal(t, y, "mm")
        if len(t) > 1:
            dy = np.diff(y)
            track = float(settings.get("track_length") or 0)
            if track > 0:
                # the position wraps at the track's end while the animal keeps running
                dy[dy < -track / 2] += track
            dt = np.diff(t)
            speed = np.divide(dy, dt, out=np.zeros_like(dy), where=dt > 0)
            # a jump no animal makes: the position zeroed after a lap's
            # inter-trial interval, a gap in the speed rather than a run backwards
            gap = np.abs(speed) > MAX_SPEED_MM_S
            # one encoder step per sample is noise at 9 ms; average over
            # SPEED_WINDOW_S, the gaps left out of the average and kept as gaps
            width = max(int(round(SPEED_WINDOW_S / float(np.median(dt)))), 1)
            kernel = np.ones(width)
            counted = np.convolve(np.where(gap, 0.0, 1.0), kernel, "same")
            summed = np.convolve(np.where(gap, 0.0, speed), kernel, "same")
            speed = np.divide(summed, counted, out=np.full_like(speed, np.nan), where=counted > 0)
            speed[gap] = np.nan
            signals["speed"] = BehaviorSignal(t[1:], speed, "mm/s")
    events = {}
    if licks:
        events["lick"] = np.asarray(licks, dtype=np.float64) - offset
    rewards = [t for times in valve_open.values() for t in times]
    if rewards:
        events["reward"] = np.sort(np.asarray(rewards, dtype=np.float64)) - offset
    if laps:
        events["lap"] = np.asarray(laps, dtype=np.float64) - offset
    epochs = {}
    for name, actions in contexts.items():
        spans, open_at = [], None
        for action, t in actions:
            if action == "start":
                open_at = t
            elif action == "stop" and open_at is not None:
                spans.append((open_at - offset, t - offset))
                open_at = None
        if spans:
            epochs[name] = np.asarray(spans, dtype=np.float64)
    start = session.get("start")
    return Behavior(
        source=f"BehaviorMate {version}".strip(),
        signals=signals,
        events=events,
        epochs=epochs,
        sync=sync,
        offset_s=offset,
        start=datetime.strptime(start, "%Y-%m-%d %H:%M:%S") if start else None,
        subject=str(session.get("mouse") or ""),
        path=path,
        info=settings,
    )
