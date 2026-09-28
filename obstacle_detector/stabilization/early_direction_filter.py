"""Anticipate a bend while near rails are still straight, using causal consensus.

The preview is a hypothesis from the raw route's farther section, not an
additional rail observation. It may choose the spline's sign only when that
sign does not contradict reliable observed curvature. It never sets position,
heading, reach or the motion estimate: those still come from D's ten routes.
"""
from dataclasses import dataclass

import numpy as np

from .recovery_filter import RecoveryRouteFilter, RecoveryConfig
from .route_filter import FilterConfig, STANDING


@dataclass(frozen=True)
class EarlyDirectionConfig:
    horizons_m: tuple = (150., 120., 90.)
    minimum_rail_reach_m: float = 30.
    tangent_span_m: float = 20.
    maximum_tangent_end_m: float = 55.
    minimum_forecast_span_m: float = 20.
    minimum_support: int = 7
    bend_threshold_m: float = .4
    consensus_quantile: float = .2
    maximum_mad_floor_m: float = .25
    maximum_relative_mad: float = .5
    confirmation_frames: int = 3
    evidence_lifetime_frames: int = 10


class EarlyDirectionRouteFilter(RecoveryRouteFilter):
    def __init__(self, config=FilterConfig(reach_quantile=1.), recovery=RecoveryConfig(),
                 early=EarlyDirectionConfig()):
        self.early_config = early
        super().__init__(config, recovery)

    def reset(self):
        super().reset()
        self.preview_streak = 0
        self.preview_candidate = 0
        self.preview_sign = 0
        self.preview_age = self.early_config.evidence_lifetime_frames + 1
        self.awaiting_observed_turn = False
        self.preview_evidence = None

    def _preview(self):
        cfg = self.early_config
        for horizon in cfg.horizons_m:
            values, ages = [], []
            for measurement in self.history:
                if (measurement.reach < cfg.minimum_rail_reach_m or measurement.extent < horizon or
                        horizon-measurement.reach < cfg.minimum_forecast_span_m):
                    continue
                end = min(measurement.reach-2., cfg.maximum_tangent_end_m)
                start = max(10., end-cfg.tangent_span_m)
                stations = np.linspace(start, end, 9)
                x = measurement.values(stations)[0]
                far = measurement.values(np.array([horizon]))[0].item()
                if not np.isfinite(x).all() or not np.isfinite(far):
                    continue
                slope, offset = np.polyfit(stations, x, 1)
                values.append(far-(slope*horizon+offset))
                ages.append(measurement.age)
            if len(values) < cfg.minimum_support or 0 not in ages:
                continue
            values = np.asarray(values)
            center = float(np.median(values))
            sign = int(np.sign(center))
            mad = float(np.median(abs(values-center)))
            lower = float(np.quantile(sign*values, cfg.consensus_quantile))
            current = values[ages.index(0)]
            if (lower > cfg.bend_threshold_m and sign*current > cfg.bend_threshold_m and
                    mad <= max(cfg.maximum_mad_floor_m, abs(center)*cfg.maximum_relative_mad)):
                return dict(sign=sign, horizon_m=horizon, support=len(values),
                            bend_m=center, mad_m=mad, signed_lower_quantile_m=lower)
        return None

    def _prepare_estimate(self, raw):
        super()._prepare_estimate(raw)
        cfg = self.early_config
        self.preview_age += 1
        self.preview_evidence = self._preview() if self._monitor is not None else None
        hint = 0 if self.preview_evidence is None else self.preview_evidence['sign']
        # A confident observed turn has priority over a distant hypothesis.
        compatible = (hint and self._local_curvature is not None and
                      hint*self._local_curvature >= -self.recovery_config.straight_threshold_per_m)
        if compatible:
            self.preview_streak = self.preview_streak+1 if hint == self.preview_candidate else 1
            self.preview_candidate = hint
        else:
            self.preview_streak = self.preview_candidate = 0
        if self.preview_streak >= cfg.confirmation_frames:
            self.preview_sign = hint
            self.preview_age = 0
            if self.mode == 'normal' and self.turn_sign and hint != self.turn_sign:
                self.mode = 'recovery'
                self.recovery_entries += 1
                self.exit_streak = 0
                self._entered = True
                self._reason = 'early_direction'
                self.awaiting_observed_turn = True
        active = self.preview_sign and self.preview_age < cfg.evidence_lifetime_frames
        if active and self.mode == 'recovery' and self._local_curvature is not None:
            if self.preview_sign*self._local_curvature >= -self.recovery_config.straight_threshold_per_m:
                self.turn_sign = self.preview_sign
            else:
                self.awaiting_observed_turn = False
        if not active or (self._local_curvature is not None and
                          self.preview_sign*self._local_curvature >= self.recovery_config.direction_threshold_per_m):
            self.awaiting_observed_turn = False

    def _can_exit_recovery(self):
        return not self.awaiting_observed_turn

    def update(self, raw, motion=STANDING):
        geometry = super().update(raw, motion)
        if geometry is None:
            self.preview_streak = self.preview_candidate = self.preview_sign = 0
            self.preview_age = self.early_config.evidence_lifetime_frames+1
            self.awaiting_observed_turn = False
            self.preview_evidence = None
        self.diagnostics.update(preview_evidence=self.preview_evidence,
                                preview_streak=self.preview_streak,
                                preview_sign=self.preview_sign, preview_age=self.preview_age,
                                awaiting_observed_turn=self.awaiting_observed_turn)
        return geometry
