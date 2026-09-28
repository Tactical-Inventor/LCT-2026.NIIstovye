"""Persistent observed-rail validation and local route reacquisition.

Recovery keeps the ten transported raw routes, maximum reach and its rate
limit, and the one-sided C2 spline; only the stale shape
reference stops vetoing the fit. No frame number, obstacle event or future
measurement participates in the decision.
"""
from dataclasses import dataclass

import numpy as np

from .one_sided_filter import OneSidedRouteFilter
from .route_filter import FilterConfig, Measurement, STANDING, median


@dataclass(frozen=True)
class RecoveryConfig:
    minimum_support: int = 7
    minimum_stations: int = 4
    minimum_span_m: float = 15.
    max_mad_m: float = .15
    max_mad_per_m: float = .003
    displacement_threshold_m: float = .35
    noise_multiplier: float = 3.
    contradiction_frames: int = 5
    direction_threshold_per_m: float = .0005
    straight_threshold_per_m: float = .0002
    exit_residual_m: float = .12
    exit_budget_fraction: float = .8
    exit_confirmation_frames: int = 10


class RecoveryRouteFilter(OneSidedRouteFilter):
    def __init__(self, config=FilterConfig(reach_quantile=1.), recovery=RecoveryConfig()):
        self.recovery_config = recovery
        super().__init__(config)

    def reset(self):
        super().reset()
        self.mode = 'normal'
        self.position_streak = self.direction_streak = self.exit_streak = 0
        self.recovery_entries = 0
        self._monitor = None
        self._entered = self._exited = False
        self._carried_previous = None

    def _observed_reference(self, raw):
        cfg = self.recovery_config
        if raw is None or not self.history or self.history[-1].age != 0:
            return None
        stations = np.arange(5., 40.1, 5.)
        observed = np.array([np.where(stations <= m.reach, m.values(stations)[0], np.nan)
                             for m in self.history])
        center = median(observed)
        mad = median(np.abs(observed-center))
        support = np.sum(np.isfinite(observed), axis=0)
        current = observed[-1]
        keep = ((support >= cfg.minimum_support) & np.isfinite(current) &
                (mad <= cfg.max_mad_m + cfg.max_mad_per_m*stations) &
                (np.abs(current-center) <= cfg.displacement_threshold_m + cfg.noise_multiplier*mad))
        if keep.sum() < cfg.minimum_stations or np.ptp(stations[keep]) < cfg.minimum_span_m:
            return None
        return stations[keep], center[keep], mad[keep], support[keep]

    def _residual(self, measurement):
        if self._monitor is None or measurement is None:
            return None
        s, center, mad, support = self._monitor
        values = measurement.values(s)[0]
        if not np.all(np.isfinite(values)):
            return None
        return float(np.quantile(np.maximum(0., np.abs(values-center) -
                     self.recovery_config.noise_multiplier*mad), .75))

    def _prepare_estimate(self, raw):
        cfg = self.recovery_config
        self._entered = self._exited = False
        self._carried_previous = self.previous
        self._monitor = self._observed_reference(raw)
        self._prior_residual = self._residual(self.previous)
        curvature = np.asarray(self.direction_history)
        finite = curvature[np.isfinite(curvature)]
        self._local_curvature = float(np.median(finite)) if len(finite) else None
        reliable = self._monitor is not None and len(finite) >= cfg.minimum_support
        opposite = (reliable and self.turn_sign != 0 and
                    self.turn_sign*self._local_curvature < -cfg.direction_threshold_per_m)
        displaced = (reliable and self._prior_residual is not None and
                     self._prior_residual > cfg.displacement_threshold_m)
        self.position_streak = self.position_streak + 1 if displaced else 0
        self.direction_streak = self.direction_streak + 1 if opposite else 0
        self._reason = None
        if self.mode == 'normal' and max(self.position_streak, self.direction_streak) >= cfg.contradiction_frames:
            self._reason = 'direction' if self.direction_streak >= cfg.contradiction_frames else 'position'
            self.mode = 'recovery'
            self.recovery_entries += 1
            self.exit_streak = 0
            self._entered = True
        if self.mode == 'recovery' and reliable:
            self.turn_sign = (int(np.sign(self._local_curvature))
                              if abs(self._local_curvature) >= cfg.straight_threshold_per_m else 0)

    def _previous_shape_reference(self):
        # Without reliable current rails, coast using D and its ordinary caps.
        return None if self.mode == 'recovery' and self._monitor is not None else self.previous

    def _can_exit_recovery(self):
        return True

    def update(self, raw, motion=STANDING):
        geometry = super().update(raw, motion)
        if geometry is None:
            self._monitor = None
            self.position_streak = self.direction_streak = self.exit_streak = 0
            self.diagnostics.update(recovery_mode=self.mode, recovery_entered=False,
                                    recovery_exited=False, recovery_entries=self.recovery_entries)
            return None
        residual = self._residual(self.previous)
        innovation_ratio = None
        if self._monitor is not None and self._carried_previous is not None:
            stations = self._monitor[0]
            old = self._carried_previous.values(stations)[0]
            current = self.previous.values(stations)[0]
            if np.all(np.isfinite(old)):
                delta = current-old
                position = np.max(np.abs(delta)/(self.config.lateral_step_m+
                                                  self.config.lateral_step_per_m*stations))
                slope = np.diff(delta)/np.diff(stations)
                midpoints = (stations[1:]+stations[:-1])/2
                heading = np.max(np.abs(slope))/self.config.heading_step_rad
                second = np.max(np.abs(np.diff(slope)/np.diff(midpoints)))/self.config.second_derivative_step
                innovation_ratio = float(max(position, heading, second))
        if self.mode == 'recovery':
            cfg = self.recovery_config
            stable = (self._can_exit_recovery() and residual is not None and residual < cfg.exit_residual_m and
                      innovation_ratio is not None and innovation_ratio < cfg.exit_budget_fraction)
            self.exit_streak = self.exit_streak + 1 if stable else 0
            if self.exit_streak >= cfg.exit_confirmation_frames:
                self.mode = 'normal'
                self.position_streak = self.direction_streak = self.exit_streak = 0
                self._exited = True
        self.diagnostics.update(
            recovery_mode=self.mode, recovery_entered=self._entered, recovery_exited=self._exited,
            recovery_reason=self._reason, recovery_entries=self.recovery_entries,
            recovery_exit_streak=self.exit_streak, observed_reference_available=self._monitor is not None,
            observed_prior_residual_m=self._prior_residual, observed_output_residual_m=residual,
            observed_innovation_budget_ratio=innovation_ratio,
            position_contradiction_streak=self.position_streak,
            direction_contradiction_streak=self.direction_streak)
        return geometry
