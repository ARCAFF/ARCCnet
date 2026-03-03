"""Shared helpers for parsing serialized timeseries path grids."""

from __future__ import annotations

import ast
import json

import numpy as np


def parse_paths_grid(raw_paths):
    """
    Parse serialized path grids into ``list[list]``.

    Supports:
    - JSON strings
    - Python-literal strings
    - numpy arrays
    - already-materialized lists/tuples
    """
    if isinstance(raw_paths, str):
        try:
            parsed = json.loads(raw_paths)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(raw_paths)
    else:
        parsed = raw_paths

    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()

    normalized = []
    for timestep_paths in parsed:
        if isinstance(timestep_paths, np.ndarray):
            timestep_paths = timestep_paths.tolist()
        normalized.append(list(timestep_paths))
    return normalized
