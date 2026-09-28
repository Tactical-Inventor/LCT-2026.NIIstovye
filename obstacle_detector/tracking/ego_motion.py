"""How far the vehicle travelled along its track between two frames.

A rail vehicle cannot leave its rails, so its motion between two frames is one
number: the distance it travelled along the route. Where it then stands and
which way it points follow from the route the previous frame described. The
distance is read off the two clouds themselves: previous returns are moved to
where the sensor would see them after travelling d, and d is chosen where they
land in voxels the current cloud occupies.

A single pair of frames is not enough. Inside a tunnel the scene repeats
itself about every metre, and walls and floor run parallel to the motion, so
their returns fall on the same voxels whatever the travel - the ring pattern
is fixed to the sensor. Measured on cloud_with_fake_obj, the best of all
candidates is a wrong one on some frames: 0.65 m against a true 1.70 m, or no
motion at all. A train cannot change its speed by that much in a tenth of a
second, though, so once locked the estimate only searches a narrow window
around the travel of the previous frame, where neither of the false optima
lies. Locked, it matches the range of a standing obstacle: 0.30-0.40 m against
0.27-0.37 m at 3 m/s, 1.70 m against 1.68-1.70 m at 17 m/s.

The lock is acquired from a full search only when the vehicle is slow (the
false optima are then out of range) or when told the travel from outside:
correct() takes it from a standing obstacle the tracker follows. Frame
timestamps of the recordings jitter between 53 and 140 ms while the clouds
advance evenly, so the prediction is per frame, not per second.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np
from numba import njit, prange
from scipy.interpolate import BSpline

VOXEL_M = 0.1          # voxel size the two clouds are compared at
NEAR_M = 3.0           # the mounting and its halo end here
FAR_M = 40.0           # beyond this returns thin out and the route adds error
LATERAL_M = 6.0
HEIGHT_M = (-4.0, 5.0)
SAMPLE = 4000          # previous returns scored per candidate
STEP_M = 0.05          # candidate resolution before parabolic refinement
WINDOW_M = 0.45        # tracking window around the predicted travel
MAX_TRAVEL_M = 4.5     # 160 km/h at 10 Hz
MIN_TRAVEL_M = -0.3    # standing still, with noise
SLOW_M = 0.6           # a full search below this travel cannot hit a false optimum
SMOOTH = 0.5           # weight of a new measurement in the travel estimate
MAX_STEP_M = 0.15      # largest change of travel accepted from one frame to the next
COAST_FRAMES = 30      # frames the travel is carried over without a measurement
CHORD_M = 10.5         # middle of the 3-18 m near rails the route frame is aligned to
MOVING_RESIDUAL = 300  # previous returns off the sensor-fixed pattern needed to read the travel from them
RELOCK_FRAMES = 3      # frames in a row that must agree before a lock on standing still is abandoned
RELOCK_SPREAD_M = 0.2  # ... within this of their median

FORWARD = np.array([0.0, -1.0, 0.0])   # sensor convention: forward -Y, up +Z
UP = np.array([0.0, 0.0, 1.0])
_SHAPE = (int(round((FAR_M + 1.0) / VOXEL_M)), int(round(2 * LATERAL_M / VOXEL_M)),
          int(round((HEIGHT_M[1] - HEIGHT_M[0]) / VOXEL_M)))


@dataclass(frozen=True)
class Motion:
    """Rigid motion of the sensor from the previous frame to this one.

    A point p_prev given in the previous sensor frame is at
    (p_prev - translation) @ rotation in the current one. travel_m is the
    distance along the route. measured says whether this frame's clouds
    were compared at all; locked whether the estimate follows a lock.
    """
    travel_m: float
    yaw_rad: float
    translation: np.ndarray = field(repr=False)
    rotation: np.ndarray = field(repr=False)
    measured: bool
    locked: bool
    raw_travel_m: float | None = None
    elapsed_ms: float = 0.0

    def to_current(self, points):
        """Previous sensor-frame points expressed in the current sensor frame."""
        return (np.asarray(points, float) - self.translation) @ self.rotation


STANDING = Motion(0.0, 0.0, np.zeros(3), np.eye(3), False, False)


def motion_along(geometry, travel, chord_m=CHORD_M):
    """Translation, rotation and yaw of the sensor after `travel` metres along the track.

    The translation is in the previous sensor frame. Without a geometry the
    vehicle is taken to move straight ahead.

    The turn comes from the curvature of the measured rails, not from the
    route spline: the spline is anchored straight to the near rails and only
    bends further on (on cloud_with_fake_obj its slope changes by 1e-4 over
    the first 1.6 m of a curve of radius 400 m, which turns by 4e-3). The
    route frame points along the chord of the near rails the rail pose was
    fitted to, so the track at the sensor itself heads back by the curvature
    times the middle of that chord.
    """
    translations, rotations, yaws = motions_along(geometry, np.array([float(travel)]), chord_m)
    return translations[0], rotations[0], float(yaws[0])


def motions_along(geometry, travels, chord_m=CHORD_M):
    """motion_along for an array of travels at once: (k,3) translations, (k,3,3) rotations, (k,) yaws."""
    travels = np.asarray(travels, float)
    if geometry is None:
        return (travels[:, None] * FORWARD, np.repeat(np.eye(3)[None], len(travels), axis=0),
                np.zeros(len(travels)))
    basis = np.asarray(geometry.basis, float)
    origin = np.asarray(geometry.origin, float)
    height = BSpline(geometry.knots, geometry.height_coefficients, geometry.degree)
    start = float((-origin @ basis)[0])
    curvature = float(getattr(geometry, 'rails_end_curvature_per_m', 0.0) or 0.0)
    heading = -curvature * chord_m
    steps = np.column_stack([travels, heading * travels + 0.5 * curvature * travels ** 2,
                             height(start + travels) - height(start)])
    yaws = curvature * travels
    # Rotation about the route's up axis (Rodrigues), one per travel.
    axis = basis[:, 2] / np.linalg.norm(basis[:, 2])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotations = (np.eye(3)[None] + np.sin(yaws)[:, None, None] * k
                 + (1 - np.cos(yaws))[:, None, None] * (k @ k))
    return steps @ basis.T, rotations, yaws


@njit(cache=True)
def _mark(xyz, stamps, stamp, sample, stride):
    """One pass over the frame: mark the occupied voxels and keep every stride-th near return.

    Returns how many near returns there were and how many were kept.
    """
    n1, n2 = _SHAPE[1], _SHAPE[2]
    near = 0
    kept = 0
    for i in range(xyz.shape[0]):
        x = xyz[i, 0]
        y = xyz[i, 1]
        z = xyz[i, 2]
        if not (y <= -NEAR_M and y >= -FAR_M and abs(x) <= LATERAL_M
                and z > HEIGHT_M[0] and z < HEIGHT_M[1]):
            continue
        a = int(np.floor(-y / VOXEL_M))
        b = int(np.floor((x + LATERAL_M) / VOXEL_M))
        c = int(np.floor((z - HEIGHT_M[0]) / VOXEL_M))
        if 0 <= a < _SHAPE[0] and 0 <= b < n1 and 0 <= c < n2:
            stamps[(a * n1 + b) * n2 + c] = stamp
        if near % stride == 0 and kept < sample.shape[0]:
            sample[kept, 0] = x
            sample[kept, 1] = y
            sample[kept, 2] = z
            kept += 1
        near += 1
    return near, kept


@njit(cache=True)
def _landed(sample, stamps, stamp):
    """Which sampled previous returns land on marked voxels without any motion."""
    n1, n2 = _SHAPE[1], _SHAPE[2]
    out = np.zeros(sample.shape[0], np.bool_)
    for i in range(sample.shape[0]):
        a = int(np.floor(-sample[i, 1] / VOXEL_M))
        b = int(np.floor((sample[i, 0] + LATERAL_M) / VOXEL_M))
        c = int(np.floor((sample[i, 2] - HEIGHT_M[0]) / VOXEL_M))
        if 0 <= a < _SHAPE[0] and 0 <= b < n1 and 0 <= c < n2:
            out[i] = stamps[(a * n1 + b) * n2 + c] == stamp
    return out


@njit(cache=True, parallel=True)
def _hits(sample, translations, rotations, stamps, stamp):
    """Per candidate motion, the share of sampled previous returns landing on marked voxels.

    Candidates are independent and each is counted exactly, so they run in parallel.
    """
    n1, n2 = _SHAPE[1], _SHAPE[2]
    out = np.zeros(translations.shape[0])
    for k in prange(translations.shape[0]):
        t = translations[k]
        r = rotations[k]
        count = 0
        for i in range(sample.shape[0]):
            px = sample[i, 0] - t[0]
            py = sample[i, 1] - t[1]
            pz = sample[i, 2] - t[2]
            x = px * r[0, 0] + py * r[1, 0] + pz * r[2, 0]
            y = px * r[0, 1] + py * r[1, 1] + pz * r[2, 1]
            z = px * r[0, 2] + py * r[1, 2] + pz * r[2, 2]
            a = int(np.floor(-y / VOXEL_M))
            b = int(np.floor((x + LATERAL_M) / VOXEL_M))
            c = int(np.floor((z - HEIGHT_M[0]) / VOXEL_M))
            if 0 <= a < _SHAPE[0] and 0 <= b < n1 and 0 <= c < n2:
                if stamps[(a * n1 + b) * n2 + c] == stamp:
                    count += 1
        out[k] = count / max(sample.shape[0], 1)
    return out


class EgoMotion:
    """Frame-to-frame travel along the route from the clouds alone."""

    def __init__(self, window_m=WINDOW_M, sample=SAMPLE):
        self.window_m = window_m
        self.sample = sample
        # Voxels hold the number of the frame that last occupied them, so the
        # grid never has to be cleared.
        self.stamps = np.zeros(int(np.prod(_SHAPE)), np.int64)
        self.stamp = 0
        self.stride = 1
        self.reset()

    def reset(self):
        self.previous = None        # sampled near returns of the last frame
        self.travel = None          # current estimate of travel per frame, m
        self.locked = False
        self.coasted = 0
        self.moving = deque(maxlen=RELOCK_FRAMES)   # travels read off the moving part of the scene

    def correct(self, travel_m):
        """Adopt a travel measured elsewhere, e.g. from a standing obstacle, and lock on it."""
        self.travel = float(travel_m)
        self.locked = True
        self.coasted = 0

    def _moving_travel(self, previous, geometry):
        """Travel read only from the returns that do not repeat at zero travel.

        The ring pattern on walls and floor is fixed to the sensor, so at any
        speed most returns land where the previous ones did, and the zero-travel
        optimum wins the full search as well as a window around a lock on it.
        Measured on roundT_pressureGate_roundT, roundT_doubleT and the start of
        doubleT_platform: the train moves 1.5-1.9 m per frame from the first
        frame and the estimate stayed locked at zero throughout. The returns
        that do not land on the current cloud without motion - posts, joints,
        anything that moved in view - show the travel on their own: 1.5 m on
        every one of those frames, and about zero with barely any of them
        where the vehicle stands (doubleT_obstacle: 90-120 of 4000).
        None when too few returns are off the pattern.
        """
        rest = previous[~_landed(previous, self.stamps, self.stamp)]
        if len(rest) < MOVING_RESIDUAL:
            return None
        candidates = np.arange(MIN_TRAVEL_M, MAX_TRAVEL_M + 1e-9, STEP_M)
        scores = self._score(rest, geometry, candidates)
        return float(candidates[int(np.argmax(scores))])

    def _relock(self, travel):
        """Leave a lock on standing still once the moving part of the scene agrees it moves."""
        if travel is None or travel < SLOW_M:
            self.moving.clear()
            return False
        self.moving.append(travel)
        if len(self.moving) < RELOCK_FRAMES:
            return False
        middle = float(np.median(self.moving))
        if max(abs(v - middle) for v in self.moving) > RELOCK_SPREAD_M:
            return False
        if self.travel is not None and self.travel >= SLOW_M:
            return False
        self.travel = middle
        self.locked = True
        self.moving.clear()
        return True

    def _score(self, sample, geometry, candidates):
        """Share of the sampled previous returns that land on occupied voxels, per candidate."""
        translations, rotations, _ = motions_along(geometry, candidates)
        return _hits(sample, np.ascontiguousarray(translations), np.ascontiguousarray(rotations),
                     self.stamps, self.stamp)

    def _coast(self, geometry, tick):
        self.coasted += 1
        if self.coasted > COAST_FRAMES:
            self.locked = False
        travel = self.travel if self.travel is not None else 0.0
        translation, rotation, yaw = motion_along(geometry, travel)
        return Motion(travel, yaw, translation, rotation, False, self.locked, None,
                      1000 * (time.perf_counter() - tick))

    def update(self, xyz, geometry=None):
        """Motion from the previous call's frame to this one.

        geometry is the route the sensor followed since the previous frame,
        in the previous sensor frame; None means straight ahead.
        """
        tick = time.perf_counter()
        xyz = np.ascontiguousarray(xyz)
        self.stamp += 1
        if self.previous is None:
            # About half of a frame lies in the near field.
            self.stride = max(1, len(xyz) // (2 * self.sample)) | 1
        sample = np.empty((self.sample, 3))
        near, kept = _mark(xyz, self.stamps, self.stamp, sample, self.stride)
        # About SAMPLE returns spread over the whole near field next time; an
        # odd stride does not fall in step with the 128 lasers of a column.
        self.stride = max(1, near // self.sample) | 1
        sample = sample[:kept]
        previous, self.previous = self.previous, (sample if kept >= 200 else None)
        if previous is None or near < 200:
            return self._coast(geometry, tick) if self.travel is not None else STANDING
        if self._relock(self._moving_travel(previous, geometry)):
            self.coasted = 0
            translation, rotation, yaw = motion_along(geometry, self.travel)
            return Motion(self.travel, yaw, translation, rotation, True, True, self.travel,
                          1000 * (time.perf_counter() - tick))
        if self.locked and self.travel is not None:
            centre = self.travel
            candidates = np.arange(centre - self.window_m, centre + self.window_m + 1e-9, STEP_M)
        else:
            candidates = np.arange(MIN_TRAVEL_M, MAX_TRAVEL_M + 1e-9, STEP_M)
        candidates = candidates[(candidates >= MIN_TRAVEL_M) & (candidates <= MAX_TRAVEL_M)]
        scores = self._score(previous, geometry, candidates)
        tracking = self.locked and self.travel is not None
        if tracking:
            # The pattern overlap falls off slowly from zero travel and would
            # drag a plain maximum to the edge of the window nearer to it; the
            # travel shows as a sharp peak on that slope. Take the slope out
            # and accept only a peak inside the window.
            trend = np.polyval(np.polyfit(candidates, scores, 1), candidates)
            scores = scores - trend
        best = int(np.argmax(scores))
        raw = float(candidates[best])
        interior = 0 < best < len(scores) - 1
        if interior:
            low, mid, high = scores[best - 1], scores[best], scores[best + 1]
            curve = low - 2 * mid + high
            if curve < 0:
                raw += 0.5 * STEP_M * (low - high) / curve
        if tracking:
            if not interior:
                return self._coast(geometry, tick)
            change = float(np.clip(raw - self.travel, -MAX_STEP_M, MAX_STEP_M))
            self.travel += SMOOTH * change
        else:
            # A full search is trusted only where no false optimum can compete.
            self.travel = raw
            self.locked = raw < SLOW_M
        self.coasted = 0
        translation, rotation, yaw = motion_along(geometry, self.travel)
        return Motion(self.travel, yaw, translation, rotation, True, self.locked, raw,
                      1000 * (time.perf_counter() - tick))
