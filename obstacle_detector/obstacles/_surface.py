"""The track surface as the frame itself shows it, not as an absolute height.

The route's lower profile z0(s) is a single quadratic anchored to the near rail
heads, so it drifts away from the observed surface with range: measured on
doubleT_obstacle the surface sits 0.22 m below it at 5 m and 0.50 m below it at
85 m. A threshold on the height h above that profile therefore means something
different at every range and cannot find a low object far away.

What does stay put is the shape of the cross-section. Between the rails the
track has structure of its own - rails 0.25 m above the walkway, a drainage
channel 0.35 m below it - and that shape repeats along the route while the
whole of it drifts down together. So the surface is modelled as

    surface(s, d) = shape(d) + drift(s)

with shape(d) learned where returns are dense and drift(s) a single number per
station, which a handful of returns can still support far away. What a return
stands above is then its own part of the cross-section at its own station, and
the drift cancels out of the difference.

shape(d) is the top of the track structure at that offset, not its middle: a
drainage channel that sleepers cross every half metre is two surfaces in one
place, and measuring against the lower one would report every crossing. It is
taken over stations rather than over returns, so a level counts only if the
track holds it along the route. A thing standing in one place, however many
returns it leaves, cannot raise the surface it is then measured against.

Only the sum of the two is determined: adding a constant to shape and taking
it back out of drift leaves every surface height the same. Which split this
lands on depends on where the first pass put the near field, so read shape as
the steps of the cross-section and drift as how those steps travel with range,
not as absolute heights.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit

CELL_M = 0.05         # lateral resolution of the cross-section
SPREAD_M = 0.10       # lateral neighbourhood used for the maximum surface height
SHOULDER_M = 0.35     # lateral neighbourhood the surrounding track level spans
SHAPE_QUANTILE = 95   # over stations: the level the track keeps returning to
STATION_BIN_M = 0.5   # station resolution of both the drift and the shape
SMOOTH_M = 5.0        # stations the running median of the drift spans each way
MIN_STATIONS = 4      # stations a cross-section cell needs to be trusted
MIN_BIN = 3           # returns a station bin needs to carry its own drift
ITERATIONS = 2
SHIFT_MAX_M = 0.6     # furthest the track may lie beside the route centre
SHIFT_STEP_M = 0.025  # resolution the lateral shift is searched at
SHIFT_BIN_M = 2.0     # stations a window advances by; a window spans two
SHIFT_MIN_CELLS = 20  # cross-section cells a window must fill to be read
SHIFT_GAIN_M = 0.01   # how much better than no shift a window must fit
SHIFT_ROBUST_M = 0.1  # windows further than this off the fitted curve count less
MISFIT_M = 0.3        # a cell further off than this counts only this much
STEADY_SPREAD_M = 0.12  # a cell whose top varies less than this between stations is steady
STEADY_SHARE = 40     # ... and at least this percentage of the cells counts as steady


class Surface(NamedTuple):
    """The track under, beside and across each return, at its own station.

    base is the surface the return stands on; shoulder is the track a third of
    a metre to either side of it; crest is the highest the cross-section gets
    anywhere at that station, which on a railway is the rail heads. shape and
    drift are the two halves the first three are built from, and support is how
    many returns the station had to offer.
    """
    base: np.ndarray
    shoulder: np.ndarray
    crest: np.ndarray
    shape: np.ndarray
    drift: np.ndarray
    support: np.ndarray
    shift: np.ndarray


def _cells(offset, half_width):
    count = max(1, int(round(2 * half_width / CELL_M)))
    index = np.floor((offset + half_width) / CELL_M).astype(np.int64)
    return np.clip(index, 0, count - 1), count


def _rank_by_group(values, groups, count, quantile, minimum):
    """Nearest-rank quantile of values within each group, NaN where unsupported.

    The values are dealt to their groups and each group's run is sorted on
    its own; the highest value needs no sort at all.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    groups = np.ascontiguousarray(groups, dtype=np.int64)
    return _ranked(values, groups, int(count), float(quantile), int(minimum))


@njit(cache=True)
def _ranked(values, groups, count, quantile, minimum):
    out = np.full(count, np.nan)
    n = len(values)
    if n == 0:
        return out
    sizes = np.zeros(count, np.int64)
    for i in range(n):
        sizes[groups[i]] += 1
    if quantile >= 100.0:
        # The highest value needs no sort; a NaN sorts last, so it wins.
        top = np.full(count, -np.inf)
        for i in range(n):
            g = groups[i]
            if values[i] > top[g] or values[i] != values[i]:
                if top[g] == top[g]:
                    top[g] = values[i]
        for g in range(count):
            if sizes[g] and sizes[g] >= minimum:
                out[g] = top[g]
        return out
    starts = np.empty(count, np.int64)
    total = 0
    for g in range(count):
        starts[g] = total
        total += sizes[g]
    fill = starts.copy()
    dealt = np.empty(n)
    for i in range(n):
        g = groups[i]
        dealt[fill[g]] = values[i]
        fill[g] += 1
    for g in range(count):
        size = sizes[g]
        if size == 0 or size < minimum:
            continue
        run = dealt[starts[g]:starts[g] + size]
        run.sort()
        out[g] = run[min(size - 1, int(size * quantile / 100.0))]
    return out


def _fill(values):
    """Carry known values across the gaps; all-unknown becomes flat zero."""
    known = np.flatnonzero(np.isfinite(values))
    if not len(known):
        return np.zeros(len(values))
    return np.interp(np.arange(len(values)), known, values[known])


def _smooth(binned, reach):
    """Running median of the station bins, one pass over every window at once.

    Wide enough that a thing standing on the track cannot carry the drift up
    with it: over ten metres an object would have to fill most of the stretch
    to move the median. Support in the far field comes from carrying known
    bins across the empty ones, not from widening this window, because out
    there whole stretches have no returns at all.
    """
    padded = np.pad(binned, reach, mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1), axis=1)


def _shape(residual, cells, cell_count, bins, bin_count, quantile):
    """Top of the track structure per cross-section cell, measured over stations.

    Each station bin contributes the highest return it has in the cell, and the
    cell then takes a quantile over those bins. A level reached at one station
    only - an object, a stray return - never becomes part of the shape, while
    one the track reaches again and again does.

    Also returns which cells are steady: those whose top keeps its level from
    station to station (see _steady).
    """
    key = cells * bin_count + bins
    highest = _rank_by_group(residual, key, cell_count * bin_count, 100, 1)
    occupied = np.flatnonzero(np.isfinite(highest))
    if not len(occupied):
        return np.zeros(cell_count), np.zeros(cell_count, bool)
    tops, groups = highest[occupied], occupied // bin_count
    return (_fill(_rank_by_group(tops, groups, cell_count, quantile, MIN_STATIONS)),
            _steady(tops, groups, cell_count))


def _steady(tops, groups, cell_count):
    """Cells whose top keeps one level along the near stations.

    The drift is one number per station, read off the returns at that station,
    and it assumes the whole cross-section moves together. Where a part of the
    section changes along the route instead - a drainage channel open near the
    sensor and covered further on, a platform, the plates of a switch - its
    returns pull that number with them, in proportion to how many they are.
    Measured on doubleT_platform: an open channel over the first 9 m lowered
    the drift there by 0.1 m, and the right rail stood 0.15 m above the surface
    for 6-9 m in 13 frames running. Rails and walkway vary by 0.01-0.08 m
    between station bins, a channel crossed by sleepers or a platform by
    0.13-0.30 m, so 0.12 m separates them. Where the whole section varies, the
    steadiest share of the cells still carries the drift.
    """
    spread = (_rank_by_group(tops, groups, cell_count, 90, MIN_STATIONS)
              - _rank_by_group(tops, groups, cell_count, 10, MIN_STATIONS))
    known = np.isfinite(spread)
    if not known.any():
        return known
    limit = max(STEADY_SPREAD_M, float(np.percentile(spread[known], STEADY_SHARE)))
    return known & (spread <= limit)


def _track_shift(station, offset, residual, shape, span):
    """How far the track lies beside the route centre, at every station.

    The route centre is a fit, and on a curve it can wander off the track as
    the range grows: measured on roundT_pressureGate_roundT the rails are 0.2 m
    to one side at 40 m and 0.4 m at 50 m, while on the standing, straight
    doubleT_obstacle they stay within 0.07 m. The cross-section itself does
    not change, so each window of stations is slid across until its own
    cross-section fits the near one, and the slide is the shift.

    A window only moves off zero when the slide fits clearly better. The
    windows then decide a smooth curve rather than each its own answer: the
    route is anchored at the sensor and goes wrong by its heading and its
    bend, so the shift is a*s + b*s^2, fitted robustly and weighted by how
    clearly each window chose. Measured on doubleT_platform standing at the
    platform, single windows at 52-58 m and 62-68 m disagree by half a metre;
    followed one by one they pulled the platform edge 0.4 m into the gauge.
    Beyond the last window that says anything the curve is not extended.
    """
    fine = int(round(2 * span / SHIFT_STEP_M))
    reach = int(round(SHIFT_MAX_M / SHIFT_STEP_M))
    if not len(station):
        return np.zeros(0)
    bins = (station / SHIFT_BIN_M).astype(np.int64)
    count = int(bins.max()) + 2
    column = np.clip(np.floor((offset + span) / SHIFT_STEP_M).astype(np.int64), 0, fine - 1)
    top = _rank_by_group(residual, bins * fine + column, count * fine, 100, 1)
    top = top.reshape(count, fine)
    windows = np.fmax(top[:-1], top[1:])                 # two bins each
    padded = np.pad(windows, ((0, 0), (reach, reach + 1)), constant_values=np.nan)
    cells = len(shape)
    scores = np.full((len(windows), 2 * reach + 1), np.inf)
    filled_at_zero = np.zeros(len(windows))
    for k in range(-reach, reach + 1):
        # A return at d stands at d - e in the near cross-section, so a cell
        # of that section is filled by the returns 2j + k, 2j + 1 + k.
        start = reach + k
        seen = np.fmax(padded[:, start:start + 2 * cells:2],
                       padded[:, start + 1:start + 1 + 2 * cells:2])
        filled = np.isfinite(seen)
        misfit = np.where(filled, np.minimum(np.abs(seen - shape), MISFIT_M), 0.0)
        many = filled.sum(axis=1)
        good = many >= SHIFT_MIN_CELLS
        scores[good, k + reach] = misfit[good].sum(axis=1) / many[good]
        if k == 0:
            filled_at_zero = many
    known = np.isfinite(scores[:, reach])
    if not known.any():
        return np.zeros(len(station))
    best = np.argmin(scores[known], axis=1)
    gain = scores[known, reach] - scores[known][np.arange(len(best)), best]
    found = np.where(gain >= SHIFT_GAIN_M, (best - reach) * SHIFT_STEP_M, 0.0)
    centres = (np.flatnonzero(known) + 1.0) * SHIFT_BIN_M
    # A window that fits no better shifted still says something: that the
    # track is where the route puts it. It weighs as much as a clear choice.
    # A window holding only part of the section can be slid onto the wrong
    # feature - a platform edge onto a rail - so it weighs by how much of the
    # section it holds. Measured on doubleT_platform: windows filling 12-14 of
    # 46 cells at 60-68 m chose -0.5 m against +0.1 m from the dense ones.
    weight = np.maximum(gain, SHIFT_GAIN_M) * filled_at_zero[known] / (2 * cells)
    design = np.column_stack([centres, centres ** 2])
    fitted = np.zeros(len(centres))
    for _ in range(4):
        # Huber weights: a window far off the curve still counts, only less.
        off = np.abs(found - fitted)
        robust = weight * np.minimum(1.0, SHIFT_ROBUST_M / np.maximum(off, 1e-9))
        root = np.sqrt(robust)
        coef = np.linalg.lstsq(design * root[:, None], found * root, rcond=None)[0]
        fitted = design @ coef
    along = np.minimum(station, centres.max())
    return np.clip(coef[0] * along + coef[1] * along ** 2, -SHIFT_MAX_M, SHIFT_MAX_M)


def estimate(station, offset, height, half_width, shape_range_m=40.0,
             shape_quantile=SHAPE_QUANTILE, rail_reach_m=0.95, follow_rails=True):
    """Surface height under every return, plus the shape and drift behind it.

    shape_range_m bounds the stations the cross-section is learned from: near
    returns are dense enough to resolve rails, walkway and channel, and the far
    field then only has to supply one number per station. A cross-section that
    changes further along the route - a switch, a platform - is therefore
    measured against the near one, and shows up as excess.

    follow_rails measures where the track lies beside the route centre at each
    station and reads every return against its own place in the cross-section
    rather than the one the route centre gives it. The cross-section is then
    kept wider than the corridor by the largest shift allowed, so a track that
    lies to one side still has a section to be read against.
    """
    station = np.asarray(station, float)
    offset = np.asarray(offset, float)
    height = np.asarray(height, float)
    span = half_width + (SHIFT_MAX_M if follow_rails else 0.0)
    cells, cell_count = _cells(offset, span)
    extent = float(station.max()) if len(station) else 0.0
    bin_count = max(1, int(np.ceil((extent + STATION_BIN_M) / STATION_BIN_M)))
    bins = np.clip((station / STATION_BIN_M).astype(np.int64), 0, bin_count - 1)
    centres = (np.arange(bin_count) + 0.5) * STATION_BIN_M
    reach = max(1, int(round(SMOOTH_M / STATION_BIN_M)))
    near = station <= shape_range_m
    near_count = max(1, int(bins[near].max()) + 1) if near.any() else 1

    drift = np.zeros(len(station))
    shape = np.zeros(cell_count)
    shift = np.zeros(len(station))
    for round_ in range(ITERATIONS + (1 if follow_rails else 0)):
        if follow_rails and round_ == ITERATIONS:
            # Only once shape and drift are settled is the shift worth finding;
            # then both are learned once more with every return in its place.
            shift = _track_shift(station, offset, height - drift, shape, span)
            cells, _ = _cells(offset - shift, span)
        shape, steady = _shape((height - drift)[near], cells[near], cell_count,
                               bins[near], near_count, shape_quantile)
        residual = height - shape[cells]
        use = steady[cells]
        # The drift comes off the steady part of the section; a bin that has
        # too few returns there falls back on all of its returns.
        binned = _rank_by_group(residual[use], bins[use], bin_count, 50, MIN_BIN)
        everywhere = _rank_by_group(residual, bins, bin_count, 50, MIN_BIN)
        binned = _fill(np.where(np.isfinite(binned), binned, everywhere))
        drift = np.interp(station, centres, _smooth(binned, reach))

    # The route centre is a fit, so the rails can sit a few centimetres off
    # where the near cross-section puts them. Taking the highest cell in a
    # small lateral neighbourhood keeps that shift from reading as an object.
    ceiling = _ceiling(shape, SPREAD_M)
    # The same reading over a wider neighbourhood is the level of the track
    # around a return rather than under it: something sunk in the drainage
    # channel can stand well above the channel floor and still be below the
    # walkway on either side of it, which is a different thing from an object
    # standing on the track. Reported, not subtracted.
    shoulder = _ceiling(shape, SHOULDER_M)
    support = np.bincount(bins, minlength=bin_count)[bins]
    # The crest of the cross-section is the rail heads: over the track itself
    # nothing stands higher at the same station. It is taken over the rail band
    # alone, not the whole width, because a platform edge at the far side of
    # the corridor would otherwise raise it and hide anything between the
    # rails underneath.
    centres = -span + (np.arange(cell_count) + 0.5) * CELL_M
    band = np.abs(centres) <= rail_reach_m
    crest = float(shape[band].max()) if band.any() else float(shape.max())
    return Surface(ceiling[cells] + drift, shoulder[cells] + drift,
                   crest + drift, shape, drift, support, shift)


def _ceiling(shape, reach_m):
    """Highest cross-section cell within reach_m of each cell."""
    span = max(1, int(round(reach_m / CELL_M)))
    padded = np.pad(shape, span, mode='edge')
    return np.max(np.lib.stride_tricks.sliding_window_view(padded, 2 * span + 1), axis=1)
