"""Temporal evidence and geometric overlap of nominal objects."""
from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class MarginConfig:
    mode: str = 'ramp'
    maximum_m: float = .30
    start_m: float = 30.
    full_m: float = 50.
    continuity: bool = True
    audit_extra_m: float = .15
    outside_frames: int = 5
    missing_frames: int = 5
    min_unique: int = 3
    rise_m: float = .10
    self_return_m: float = 2.5

    def __post_init__(self):
        if self.mode not in {'none', 'constant', 'hard', 'ramp'}:
            raise ValueError('Unknown margin mode')
        if not all(np.isfinite(x) for x in (self.maximum_m, self.start_m, self.full_m,
                                           self.audit_extra_m, self.rise_m, self.self_return_m)):
            raise ValueError('Nonfinite margin configuration')
        if self.maximum_m < 0 or self.start_m < 0 or self.full_m <= self.start_m:
            raise ValueError('Invalid distance ramp')
        if self.audit_extra_m < 0 or self.rise_m <= 0 or self.self_return_m < 0:
            raise ValueError('Invalid spatial thresholds')
        if self.outside_frames < 1 or self.missing_frames < 1 or self.min_unique < 3:
            raise ValueError('Invalid evidence thresholds')

    def width(self, distance):
        distance = np.asarray(distance, float)
        if self.mode == 'none': return np.zeros_like(distance)
        if self.mode == 'constant': return np.full_like(distance, self.maximum_m)
        if self.mode == 'hard': return self.maximum_m*(distance > self.start_m)
        return self.maximum_m*np.clip((distance-self.start_m)/(self.full_m-self.start_m), 0., 1.)


@dataclass(frozen=True)
class InnerConfig(MarginConfig):
    inner_maximum_m: float = .20
    promote_hits: int = 2
    promote_window: int = 3
    demote_frames: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.mode != 'ramp' or self.maximum_m <= 0:
            raise ValueError('Inner gauge requires a positive ramped outer margin')
        if not np.isfinite(self.inner_maximum_m) or not 0 <= self.inner_maximum_m < 1.15:
            raise ValueError('Invalid inner margin')
        if not 1 <= self.promote_hits <= self.promote_window or self.demote_frames < 1:
            raise ValueError('Invalid temporal evidence thresholds')

    def inner_width(self, distance):
        return self.width(distance)*self.inner_maximum_m/self.maximum_m


def classify(local, half, height, inner, outer, minimum=3):
    """At least minimum distinct returns must penetrate the inner core.

    One or two remaining edge returns prevent claiming observed outside.
    The bottom of the gauge is unchanged; only sides and ceiling are offset.
    supported_gap is the minimum-th smallest gap, not a centroid/percentile.
    """
    unique = np.unique(local, axis=0)
    gap = np.maximum(abs(unique[:, 1])-half, unique[:, 2]-height)
    eligible = unique[:, 2] >= 0.
    deep = int(np.count_nonzero(eligible & (gap < -inner-1e-8)))
    possible = eligible & (gap <= outer+1e-8)
    state = 'inside' if deep >= minimum else ('borderline' if np.any(possible) else 'outside')
    supported = float(np.sort(gap)[minimum-1]) if len(gap) >= minimum else None
    return state, deep, supported


@dataclass
class InnerState:
    recent: deque = field(default_factory=lambda: deque(maxlen=3))
    deep_recent: deque = field(default_factory=deque)
    exists: bool = False
    held_blocked: bool = False
    outside_streak: int = 0
    edge_streak: int = 0
    outside_confirmed: bool = False
    retained_margin_m: float = 0.
    last_relation: str = 'unknown'
    source_ids: set = field(default_factory=set)

    def observe(self, candidate, config, missing=0):
        self.recent.append(candidate is not None)
        # Entry needs readings of a thing standing on the track; other readings
        # of an established object still hold it (see GaugeConfig.promote_kinds).
        self.deep_recent.append(candidate is not None and candidate.relation == 'inside'
                                and getattr(candidate, 'promotable', True))
        while len(self.deep_recent) > config.promote_window:
            self.deep_recent.popleft()
        if candidate is None:
            # Neither occlusion nor a missing scan supplies release evidence.
            self.edge_streak = self.outside_streak = 0
            return
        self.exists |= sum(self.recent) >= 2 or candidate.source_confirmed
        self.source_ids.update(candidate.source_ids)
        self.retained_margin_m = max(self.retained_margin_m, candidate.margin_m)
        self.last_relation = candidate.relation
        self.edge_streak = self.edge_streak+1 if candidate.relation != 'inside' else 0
        self.outside_streak = self.outside_streak+1 if candidate.relation == 'outside' else 0
        if candidate.relation != 'outside':
            self.outside_confirmed = False
        if self.exists and candidate.relation == 'inside' and sum(self.deep_recent) >= config.promote_hits:
            self.held_blocked = True
        if self.edge_streak >= config.demote_frames:
            self.held_blocked = False
        if self.outside_streak >= config.outside_frames:
            self.held_blocked = False
            self.outside_confirmed = True

    @property
    def decision(self):
        if self.exists and self.held_blocked:
            return 'BLOCKED'
        return 'CLEAR' if self.outside_confirmed else 'CAUTION'
