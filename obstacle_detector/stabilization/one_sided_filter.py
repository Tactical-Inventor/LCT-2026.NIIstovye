"""Route filter with a hard, causal, one-direction curvature prior.

For a cubic spline x(s), x'' is linear in every knot span. Constraining x'' at
every unique knot therefore constrains its sign everywhere, not only on a
sampling grid. Position, tangent and curvature remain continuous at the rail /
forecast boundary. Only measured-rail curvature chooses the turn direction.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, nnls

from .route_filter import RouteFilter, FilterConfig, STANDING


@dataclass(frozen=True)
class OneSidedConfig:
    min_observations: int = 3
    curvature_deadband: float = 5e-5
    agreement: float = .7
    tolerance: float = 1e-10


class OneSidedRouteFilter(RouteFilter):
    def __init__(self, config=FilterConfig(), shape_config=OneSidedConfig()):
        self.shape_config = shape_config
        super().__init__(config)

    def reset(self):
        super().reset()
        self.direction_history = deque(maxlen=self.config.window)
        self.turn_sign = 0
        self.projection_fallbacks = 0

    def _choose_direction(self, raw):
        # One entry per update, including missing observations, makes the sign
        # history use the same causal ten-frame window as the route history.
        value = np.nan
        if raw is not None and raw.rails_end_m is not None and raw.rails_end_m >= 15.:
            value = float(raw.rails_end_curvature_per_m)
        self.direction_history.append(value)
        values = np.array(self.direction_history)
        finite = values[np.isfinite(values)]
        self.rail_curvature_median = float(np.median(finite)) if len(finite) else None
        if self.turn_sign:
            return
        decisive = finite[np.abs(finite) >= self.shape_config.curvature_deadband]
        if len(decisive) >= self.shape_config.min_observations:
            direction = int(np.sign(np.median(decisive)))
            if np.mean(np.sign(decisive) == direction) >= self.shape_config.agreement:
                # The user's prior is one turn direction for this route. A
                # single noisy opposite measurement cannot flip it; reset()
                # starts acquisition for a different route/recording.
                self.turn_sign = direction

    @staticmethod
    def _knots(basis):
        return np.unique(basis.t)

    def _shape_valid(self, basis, coefficients):
        second = basis(self._knots(basis), 2) @ coefficients
        if self.turn_sign == 0:
            return bool(np.max(np.abs(second)) <= self.shape_config.tolerance)
        return bool(np.min(self.turn_sign * second) >= -self.shape_config.tolerance)

    def _shape_projection(self, coefficients, basis, matrix, weight):
        if self._shape_valid(basis, coefficients):
            return coefficients
        knots = self._knots(basis)
        spacing = self.config.knot_spacing_m
        n = len(coefficients)
        # Value and slope at the sensor + second derivatives at every knot
        # uniquely parameterize this C2 cubic spline (n independent values).
        system = np.vstack((basis([0.]), basis([0.], 1)*spacing,
                            basis(knots, 2)*spacing**2))
        inverse = np.linalg.solve(system, np.eye(n))
        if self.turn_sign == 0:
            # Until a direction is supported, forecast a straight line fitted
            # to measured rails only. Distant predictions cannot pick its yaw.
            near = weight > 1.
            if np.count_nonzero(near) < 2:
                near = np.ones(len(weight), bool)
            affine = matrix @ inverse[:, :2]
            anchor = np.linalg.lstsq(affine[near], (matrix @ coefficients)[near], rcond=None)[0]
            return inverse[:, :2] @ anchor
        anchor = (system @ coefficients)[:2]
        base = inverse[:, :2] @ anchor
        curvature_map = self.turn_sign * inverse[:, 2:]
        # Positive unknowns are magnitudes of x'' at the knots. They cannot
        # introduce an opposite bend anywhere in a knot span.
        weighted = np.sqrt(weight)[:, None] * matrix
        smooth = np.sqrt(self.config.smoothness) * basis(
            np.linspace(0., knots[-1], max(4, int(knots[-1]))), 3) * spacing**2
        design = np.vstack((weighted @ curvature_map, smooth @ curvature_map))
        target = np.r_[weighted @ (coefficients-base), -smooth @ base]
        try:
            magnitudes, _ = nnls(design, target, maxiter=30*n)
            projected = base + curvature_map @ magnitudes
        except RuntimeError:
            # The anchored straight line is always a feasible geometric prior.
            self.projection_fallbacks += 1
            projected = base
        if not self._shape_valid(basis, projected):
            raise ArithmeticError('Curvature projection violated its sign constraint')
        return projected

    def _shape_constraint(self, basis, previous_coefficients):
        second = basis(self._knots(basis), 2) * self.config.knot_spacing_m**2
        previous = second @ previous_coefficients
        if self.turn_sign == 0:
            return LinearConstraint(second, -previous, -previous)
        return LinearConstraint(self.turn_sign*second, -self.turn_sign*previous, np.inf)

    def update(self, raw, motion=STANDING):
        self._choose_direction(raw)
        geometry = super().update(raw, motion)
        if geometry is not None:
            spline = BSpline(geometry.knots, geometry.lateral_coefficients, 3)
            stations = np.unique(geometry.knots)
            second = spline(stations, 2)
            if self.turn_sign:
                assert np.min(self.turn_sign*second) >= -self.shape_config.tolerance
            else:
                assert np.max(np.abs(second)) <= self.shape_config.tolerance
            self.diagnostics.update(turn_sign=self.turn_sign,
                                    rail_curvature_median=self.rail_curvature_median,
                                    min_second_derivative=float(second.min()),
                                    max_second_derivative=float(second.max()),
                                    projection_fallbacks=self.projection_fallbacks)
        return geometry
