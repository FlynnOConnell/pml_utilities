"""Orientation op lists for the BigStitcher export: ``["rot", axis, deg]`` then ``["flip", axis]``, rotations first."""

from __future__ import annotations


def orientation_ops(rotations: list, flips: list) -> list:
    """Op list from live UI state: rotations first, then flips."""
    ops: list = []
    for rot in rotations:
        deg = int(rot["deg"])
        if rot["sign"] == "-":
            deg = -deg
        ops.append(["rot", rot["axis"], deg])
    for axis in flips:
        ops.append(["flip", axis])
    return ops
