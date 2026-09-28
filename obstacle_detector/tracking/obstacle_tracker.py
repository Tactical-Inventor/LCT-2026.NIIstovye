"""Objects on the track as things that persist, not as a verdict per frame.

The single-frame detector sorts every raised group by its shape, and the same
object is sorted differently from frame to frame. Measured on
cloud_with_fake_obj, of the 119 labelled frames it missed, 78 held a group
exactly where the object was, only sorted as something else: a 2x2 m box far
ahead reads as cut off by the gauge ceiling once the far profile drifts, a
thin post as possibly beside the track once the forecast widens, a low bar as
level with the rail heads once it spoils the crest estimate itself. A further
13 frames had no route at all, the rails hidden by the object. The object did
not come and go; the reading of it did.

So each group is followed from frame to frame. Where it was seen last is
carried by the vehicle's motion, the group found there is the same thing, and
the evidence about it accumulates:

* seen as an object: strong evidence for it;
* seen as possibly beside the track or just appeared: some evidence;
* seen as structure or track furniture: no evidence either way once the
  object is established - a different reading of the same returns is not
  proof that the thing left - except where it now lies outside the gauge;
* not seen where it should be visible: evidence against it, weighed by how
  well that range is sampled; not seen where it cannot be - behind a nearer
  object, past the horizon, under the sensor's own blind zone - nothing.

An object is reported once its evidence passes a threshold and until it falls
below a lower one, or until the vehicle reaches it. It may move: each track
keeps its own speed along the track, zero for a thing standing still, so a
moving obstacle is followed rather than lost; its speed is its own estimate,
not a verified one.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import numpy as np

OBJECT_GAIN = 1.0         # evidence for a frame that reads the group as an object
WEAK_CONFIDENCE = 0.2     # ... scaled by its confidence from here
CONFIDENCE_SPAN = 0.5     # ... to full weight this much higher
MIN_OBJECT_GAIN = 0.2
CANDIDATE_GAIN = 0.35     # ... as possibly beside the track, or newly appeared
OUTSIDE_LOSS = 0.6        # ... as lying outside the gauge of the rails
UNSEEN_DECAY = 0.25       # an unconfirmed track loses this per frame without object evidence
CONFIRM = 1.8             # evidence at which a track is reported
CONFIRM_OBJECT_READINGS = 2   # ... of which at least this many frames read it as an object
RELEASE = 0.6             # ... and below which it no longer is
CEILING = 4.0             # evidence saturates here, so an old track can still be released
DELETE = -1.0
MISS_NEAR = 0.6           # evidence against, per frame not seen where visible, within 40 m
MISS_MID = 0.35           # 40-80 m
MISS_FAR = 0.15           # beyond
MISS_BLIND = 0.1          # within BLIND_M, where the object may be under the lower field of view
BLIND_M = 5.0
SELF_RETURN_M = 2.5       # the detector drops returns this close
CLIPPED_M = 1.0           # within this of that zone a group may be cut by it
PASSED_M = 1.0            # a track this close has been reached by the vehicle
UNCONFIRMED_LOST = 3      # frames an unconfirmed track survives unseen
GATE_RANGE_M = 1.0        # association tolerance along the line of sight, near
GATE_RANGE_PER_M = 0.03   # ... growing with range
GATE_APPROACH_M = 2.5     # an unconfirmed track may come this much closer than predicted
GATE_SIDE_M = 0.7         # association tolerance across the line of sight
GATE_SIDE_PER_M = 0.01
SPEED_GAIN = 0.15         # weight of one frame's range innovation in a track's own speed
SPEED_FORGET = 0.05       # ... which returns to standing still at this rate per frame
INNOVATION_M = 1.0        # largest innovation one frame may contribute
MAX_SPEED_M = 3.0         # own speed per frame a track may take, either way
MIN_RETURNS = 3


@dataclass
class Track:
    """One followed thing, in the current sensor frame."""
    id: int
    centre: np.ndarray                  # centroid of its returns, sensor frame
    near_offset: float                  # centroid range minus nearest-return range
    low: np.ndarray                     # route-local box of the last observation
    high: np.ndarray
    kind: str
    evidence: float = 0.0
    confirmed: bool = False
    speed: float = 0.0                  # own motion along the track, m per frame
    hits: int = 0
    object_hits: int = 0
    unseen: int = 0
    age: int = 0
    confidence: float = 0.0
    seen: bool = True

    @property
    def range_m(self):
        return max(0.0, float(np.linalg.norm(self.centre)) - self.near_offset)


@dataclass(frozen=True)
class Observation:
    """One detector group, reduced to what the tracker needs."""
    kind: str
    confidence: float
    reasons: tuple
    range_m: float
    unique_returns: int
    centre: np.ndarray          # centroid of its returns, sensor frame
    low: np.ndarray             # route-local box
    high: np.ndarray


def observations(detection, region):
    """Observations of every group the detector reported on this frame."""
    out = []
    if detection is None or region is None:
        return out
    for obstacle in detection.obstacles:
        if obstacle.unique_returns < MIN_RETURNS:
            continue
        local = np.asarray(region.local[obstacle.indices], float)
        out.append(Observation(obstacle.kind, float(obstacle.confidence), tuple(obstacle.reasons),
                               float(obstacle.range_m), int(obstacle.unique_returns),
                               np.asarray(region.points[obstacle.indices], float).mean(axis=0),
                               local.min(axis=0), local.max(axis=0)))
    return out


@dataclass(frozen=True)
class TrackReport:
    """A track as the stream reports it for one frame."""
    id: int
    confirmed: bool
    seen: bool
    kind: str
    range_m: float
    station_m: float
    offset_m: float
    evidence: float
    object_hits: int
    speed_m_per_frame: float
    low: tuple
    high: tuple
    centre: tuple


def _local(geometry, points):
    return (np.asarray(points, float) - np.asarray(geometry.origin, float)) @ np.asarray(geometry.basis, float)


class ObstacleTracker:
    """Follow detector groups over frames and decide which of them block the track."""

    def __init__(self):
        self.ids = count(1)
        self.reset()

    def reset(self):
        self.tracks: list[Track] = []

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _gain(obstacle):
        reasons = set(obstacle.reasons)
        if obstacle.kind in ('object', 'transient'):
            # A weak reading - three or four returns, a low evidence score -
            # is what the far false objects on the clean recordings look like.
            return OBJECT_GAIN * float(np.clip((obstacle.confidence - WEAK_CONFIDENCE) / CONFIDENCE_SPAN,
                                               MIN_OBJECT_GAIN, 1.0))
        if obstacle.kind == 'unplaced':
            return CANDIDATE_GAIN
        if 'outside_the_gauge_of_the_rails' in reasons:
            return -OUTSIDE_LOSS
        return 0.0

    def _predict(self, track, motion, geometry):
        centre = track.centre if motion is None else motion.to_current(track.centre[None])[0]
        if geometry is not None and track.speed:
            centre = centre + track.speed * np.asarray(geometry.basis, float)[:, 0]
        return centre

    @staticmethod
    def _gates(track, predicted):
        distance = float(np.linalg.norm(predicted))
        along = GATE_RANGE_M + GATE_RANGE_PER_M * distance
        # A new track does not know its own speed yet and may be approaching.
        approach = along + (0.0 if track.hits > 2 else GATE_APPROACH_M)
        side = GATE_SIDE_M + GATE_SIDE_PER_M * distance
        return along, approach, side

    # ----------------------------------------------------------------- update
    def update(self, found, geometry, motion=None):
        """Advance all tracks by one frame with this frame's observations.

        found is a list of Observation (see observations()); geometry the
        route the region was selected with; motion the ego Motion since the
        previous frame. Returns the list of TrackReport.
        """
        predicted = [self._predict(t, motion, geometry) for t in self.tracks]
        # Greedy association, nearest pairs first. Along the line of sight the
        # nearest return is compared, not the centroid: the centroid of a large
        # object moves as more of it comes into view.
        pairs = []
        for ti, (track, centre) in enumerate(zip(self.tracks, predicted)):
            along, approach, side = self._gates(track, centre)
            expected = float(np.linalg.norm(centre)) - track.near_offset
            direction = centre / max(np.linalg.norm(centre), 1e-6)
            for gi, found_one in enumerate(found):
                radial = found_one.range_m - expected
                delta = found_one.centre - centre
                lateral = float(np.linalg.norm(delta - float(delta @ direction) * direction))
                if -approach <= radial <= along and lateral <= side:
                    pairs.append((abs(radial) / along + lateral / side, ti, gi))
        pairs.sort()
        used_groups, matched = set(), {}
        for _, ti, gi in pairs:
            if ti in matched or gi in used_groups:
                continue
            used_groups.add(gi); matched[ti] = gi

        extent = 0.0 if geometry is None else float(geometry.extent_m)
        half_width = 1.15 if geometry is None else float(geometry.half_width_m)
        survivors = []
        for ti, track in enumerate(self.tracks):
            track.age += 1
            centre = predicted[ti]
            if ti in matched:
                seen = found[matched[ti]]
                expected = float(np.linalg.norm(centre)) - track.near_offset
                if seen.range_m < SELF_RETURN_M + CLIPPED_M:
                    # Its near part is already inside the zone the detector
                    # drops, so what is left of it stays at the same range
                    # while the thing itself comes closer: evidence that it
                    # is there, not a measurement of where.
                    track.centre = centre
                else:
                    # A thing standing still is where the motion carried it;
                    # its own speed is learnt slowly and forgotten slowly, so
                    # one late or early reading - an object injected on its own
                    # clock stalls for a frame and then jumps - does not set it
                    # moving.
                    innovation = float(np.clip(seen.range_m - expected, -INNOVATION_M, INNOVATION_M))
                    track.speed = float(np.clip((1 - SPEED_FORGET) * track.speed
                                                + SPEED_GAIN * innovation, -MAX_SPEED_M, MAX_SPEED_M))
                    track.centre = seen.centre
                    track.near_offset = float(np.linalg.norm(seen.centre)) - seen.range_m
                track.low, track.high = seen.low, seen.high
                track.kind = seen.kind
                track.confidence = seen.confidence
                gain = self._gain(seen)
                track.evidence += gain
                track.hits += 1
                track.object_hits += seen.kind in ('object', 'transient')
                track.unseen = 0
                track.seen = True
                if gain <= 0 and not track.confirmed:
                    track.evidence -= UNSEEN_DECAY
            else:
                track.centre = centre
                track.seen = False
                track.unseen += 1
                track.evidence -= self._miss(track, geometry, extent, half_width, predicted, ti)
                if not track.confirmed:
                    track.evidence -= UNSEEN_DECAY
            track.evidence = min(track.evidence, CEILING)
            if (not track.confirmed and track.evidence >= CONFIRM
                    and track.object_hits >= CONFIRM_OBJECT_READINGS):
                track.confirmed = True
            elif track.confirmed and track.evidence < RELEASE:
                track.confirmed = False
            if self._ahead(track, geometry) < PASSED_M or track.evidence < DELETE:
                continue          # reached by the vehicle, or disproved
            if not track.confirmed and track.unseen > UNCONFIRMED_LOST:
                continue
            survivors.append(track)

        for gi, seen in enumerate(found):
            if gi in used_groups:
                continue
            gain = self._gain(seen)
            if gain <= 0:
                continue          # structure and track furniture do not open a track
            survivors.append(Track(next(self.ids), seen.centre,
                                   float(np.linalg.norm(seen.centre)) - seen.range_m, seen.low, seen.high,
                                   seen.kind, evidence=gain, hits=1,
                                   object_hits=int(seen.kind in ('object', 'transient')),
                                   confidence=seen.confidence))
        self.tracks = self._merged(survivors)
        return [self._report(t, geometry) for t in self.tracks]

    @staticmethod
    def _ahead(track, geometry):
        """How far ahead along the track its near side is; negative once passed."""
        if geometry is None:
            along = -float(track.centre[1])          # sensor forward is -Y
        else:
            along = float(_local(geometry, track.centre[None])[0][0])
        return along - track.near_offset

    def _merged(self, tracks):
        """One track per thing: a later track found on top of an older one joins it."""
        tracks = sorted(tracks, key=lambda t: t.id)
        kept = []
        for track in tracks:
            host = None
            for other in kept:
                along, _, side = self._gates(other, other.centre)
                delta = track.centre - other.centre
                direction = other.centre / max(np.linalg.norm(other.centre), 1e-6)
                lateral = float(np.linalg.norm(delta - float(delta @ direction) * direction))
                if abs(track.range_m - other.range_m) <= along and lateral <= side:
                    host = other
                    break
            if host is None:
                kept.append(track)
                continue
            # The newer one carries the current observation.
            if track.seen and not host.seen:
                host.centre, host.near_offset = track.centre, track.near_offset
                host.low, host.high, host.kind = track.low, track.high, track.kind
                host.seen, host.unseen = True, 0
            host.evidence = min(CEILING, max(host.evidence, track.evidence))
            host.object_hits = max(host.object_hits, track.object_hits)
            host.confirmed = host.confirmed or track.confirmed
        return kept

    def _miss(self, track, geometry, extent, half_width, predicted, index):
        """Evidence against a track not seen this frame, zero where it could not be seen."""
        if geometry is None:
            return 0.0
        local = _local(geometry, track.centre[None])[0]
        distance = track.range_m
        if local[0] > extent - 2.0 or abs(local[1]) > half_width + 0.5:
            return 0.0
        # Hidden behind a nearer confirmed track in the same lateral band.
        for other, centre in zip(self.tracks, predicted):
            if other is track or not other.confirmed or other.range_m >= distance - 1.0:
                continue
            if abs(float(_local(geometry, centre[None])[0][1]) - local[1]) < 1.5:
                return 0.0
        if distance < SELF_RETURN_M + 0.5:
            return 0.0            # the detector drops these returns as the mounting
        if distance < BLIND_M:
            return MISS_BLIND
        if distance < 40.0:
            return MISS_NEAR
        if distance < 80.0:
            return MISS_MID
        return MISS_FAR

    @staticmethod
    def _report(track, geometry):
        station = offset = float('nan')
        if geometry is not None:
            local = _local(geometry, track.centre[None])[0]
            station, offset = float(local[0]), float(local[1])
        return TrackReport(track.id, track.confirmed, track.seen, track.kind, track.range_m,
                           station, offset, track.evidence, track.object_hits, track.speed,
                           tuple(map(float, track.low)),
                           tuple(map(float, track.high)), tuple(map(float, track.centre)))

    def blocking(self, reports):
        """Confirmed tracks ahead of the vehicle, nearest first."""
        return sorted((r for r in reports if r.confirmed), key=lambda r: r.range_m)
