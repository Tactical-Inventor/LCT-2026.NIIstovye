"""Group flagged returns into objects, with a reach that follows the sampling.

A lidar spaces its returns by angle, so the same object is sampled finely near
the sensor and coarsely far away: on doubleT_obstacle the returns of one ring
lie 0.10 m apart across the track at 56 m and a few millimetres apart at 3 m,
and successive rings are 0.14 m apart in height there. A single metric radius
therefore either splits a distant object into single returns or merges
everything nearby into one blob, so the reach grows with range instead.

Along the route the reach stays fixed: stations of one object are its own
length, not a sampling artefact, and a distant ring landing on flat ground is
metres away from its neighbour, which is exactly the gap that must not close.
"""
from __future__ import annotations

import numpy as np
from numba import njit

STATION_REACH_M = 1.2      # along the route; an object's own length, not sampling
LATERAL_REACH_M = 0.25     # across the route, at short range
HEIGHT_REACH_M = 0.20      # in height, at short range
ANGULAR_REACH = 0.006      # radians; both of the above grow by this times range


@njit(cache=True)
def _components(station, offset, height, lateral, vertical):
    """Union-find over pairs within reach, scanned in station order."""
    n = len(station)
    parent = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            if station[j] - station[i] > STATION_REACH_M:
                break
            gate = lateral[i] if lateral[i] > lateral[j] else lateral[j]
            if abs(offset[j] - offset[i]) > gate:
                continue
            gate = vertical[i] if vertical[i] > vertical[j] else vertical[j]
            if abs(height[j] - height[i]) > gate:
                continue
            a = i
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            b = j
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                parent[b] = a
    labels = np.empty(n, np.int64)
    numbers = np.full(n, -1, np.int64)
    count = 0
    for i in range(n):
        a = i
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        if numbers[a] < 0:
            numbers[a] = count
            count += 1
        labels[i] = numbers[a]
    return count, labels


def group(station, offset, height, sensor_range):
    """Return one array of row indices per connected group, station order kept."""
    if not len(station):
        return []
    order = np.argsort(station, kind='stable')
    spread = ANGULAR_REACH * np.asarray(sensor_range, float)
    ready = lambda a: np.ascontiguousarray(np.asarray(a, float)[order])
    count, labels = _components(ready(station), ready(offset), ready(height),
                                ready(np.maximum(LATERAL_REACH_M, spread)),
                                ready(np.maximum(HEIGHT_REACH_M, spread)))
    return [order[labels == k] for k in range(count)]


def warmup():
    zero = np.zeros(2)
    _components(zero, zero, zero, np.ones(2), np.ones(2))
