"""Causal route stabilization on top of the per-frame route geometry.

History holds original measurements, never filtered outputs. Every update moves
history and the previous estimate into the current sensor frame before taking
statistics. The returned RouteGeometry is used by the real corridor selector.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize

from obstacle_detector.route.corridor import RouteGeometry, select_with_geometry
from obstacle_detector.tracking.ego_motion import STANDING
from .qp import solve_qp

# Columns are forward (-Y), lateral (+X), vertical (+Z).
SENSOR_BASIS = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


@dataclass(frozen=True)
class FilterConfig:
    window: int = 10
    reach_quantile: float = .8
    limit: bool = True
    lateral_step_m: float = .06
    lateral_step_per_m: float = .0012
    heading_step_rad: float = .002
    second_derivative_step: float = .00010
    reach_retreat_m: float = 1.0
    reach_growth_m: float = 3.0
    # Minimum sensor-relative growth when fresh rails replenish fast travel.
    reach_replenish_m: float = .5
    knot_spacing_m: float = 12.0
    smoothness: float = 12.0

    def __post_init__(self):
        if self.window < 1 or not 0 <= self.reach_quantile <= 1:
            raise ValueError('Invalid history window or reach quantile')
        for name in ('lateral_step_m', 'lateral_step_per_m', 'heading_step_rad',
                     'second_derivative_step', 'reach_retreat_m', 'reach_growth_m',
                     'reach_replenish_m', 'knot_spacing_m'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.smoothness) or self.smoothness < 0:
            raise ValueError('smoothness must be finite and nonnegative')


def sensor_points(g, stations, offset=0.):
    s = np.atleast_1d(stations).astype(float)
    lat = BSpline(g.knots, g.lateral_coefficients, g.degree)
    z = BSpline(g.knots, g.height_coefficients, g.degree)
    slope = lat(s, 1)
    scale = np.sqrt(1 + slope * slope)
    local = np.column_stack((s - offset * slope / scale,
                             lat(s) + offset / scale, z(s)))
    return local @ g.basis.T + g.origin


@dataclass
class Measurement:
    points: np.ndarray
    rail_end: np.ndarray
    heads_end: np.ndarray
    age: int = 0
    # Points ordered by forward distance, duplicates dropped; rebuilt after carry().
    _ordered: tuple | None = None

    @classmethod
    def from_geometry(cls, g):
        stations = np.unique(np.r_[np.arange(0., g.extent_m, .5), g.extent_m,
                                    np.clip(g.rails_end_m or 0., 0., g.extent_m)])
        return cls(sensor_points(g, stations),
                   sensor_points(g, [np.clip(g.rails_end_m or 0., 0., g.extent_m)])[0],
                   sensor_points(g, [np.clip(g.heads_end_m or 0., 0., g.extent_m)])[0])

    def carry(self, motion):
        self.points = motion.to_current(self.points)
        self.rail_end = motion.to_current(self.rail_end[None])[0]
        self.heads_end = motion.to_current(self.heads_end[None])[0]
        self.age += 1
        self._ordered = None

    @property
    def reach(self):
        return max(0., float(-self.rail_end[1]))

    @property
    def extent(self):
        return float(np.max(-self.points[:, 1]))

    def values(self, s):
        if self._ordered is None:
            forward = -self.points[:, 1]
            order = np.argsort(forward)
            forward, points = forward[order], self.points[order]
            keep = np.r_[True, np.diff(forward) > 1e-7]
            self._ordered = (forward[keep], points[keep, 0].copy(), points[keep, 2].copy())
        forward, lateral, height = self._ordered
        return (np.interp(s, forward, lateral, left=np.nan, right=np.nan),
                np.interp(s, forward, height, left=np.nan, right=np.nan))


def median(values):
    """np.nanmedian(values, axis=0), bit for bit, without its per-column loop.

    NaNs sort last; the median of the n finite values of a column is the mean
    of its two middle ones, which for odd n is the middle one itself. All-NaN
    columns stay NaN. Checked against np.nanmedian on 3000 calls of a run.
    """
    ordered = np.sort(np.asarray(values, float), axis=0)
    count = np.count_nonzero(~np.isnan(ordered), axis=0)
    low = np.take_along_axis(ordered, np.expand_dims(np.maximum((count-1)//2, 0), 0), axis=0)[0]
    high = np.take_along_axis(ordered, np.expand_dims(np.maximum(count//2, 0), 0), axis=0)[0]
    return (low + high) / 2


class RouteFilter:
    def __init__(self, config=FilterConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.history = deque()
        self.previous = None
        self.diagnostics = {}

    def _shape_projection(self, coefficients, basis, matrix, weight):
        """Optional geometric prior; the default leaves coefficients unchanged."""
        return coefficients

    def _shape_constraint(self, basis, previous_coefficients):
        return None

    def _shape_valid(self, basis, coefficients):
        return True

    def _prepare_estimate(self, raw):
        """Optional state validation after observations reach the current frame."""

    def _previous_shape_reference(self):
        return self.previous

    def update(self, raw, motion=STANDING):
        cfg = self.config
        for measurement in self.history:
            measurement.carry(motion)
        while self.history and self.history[0].age >= cfg.window:
            self.history.popleft()
        reach_motion_m = 0.
        if self.previous is not None:
            previous_reach = self.previous.reach
            self.previous.carry(motion)
            # Preserve the ordinary growth limit where it can replenish travel.
            # At higher speed fresh rails must still be able to extend sensor-
            # relative visibility. Missing measurements get no such allowance.
            if raw is not None:
                reach_motion_m = max(0., previous_reach - self.previous.reach)
        if raw is not None:
            self.history.append(Measurement.from_geometry(raw))
        # An expired observation is not kept indefinitely by the filtered state.
        if not self.history:
            self.previous = None
            self.diagnostics = {'available': False}
            return None

        self._prepare_estimate(raw)

        latest = self.history[-1]
        extent = max(3., max(m.extent for m in self.history))
        rail_target = float(np.quantile([m.reach for m in self.history], cfg.reach_quantile))
        reach = rail_target
        if cfg.limit and self.previous is not None:
            growth = cfg.reach_growth_m
            if raw is not None:
                growth = max(growth, reach_motion_m + cfg.reach_replenish_m)
            reach = self.previous.reach + np.clip(reach - self.previous.reach,
                                                  -cfg.reach_retreat_m, growth)
        reach = float(np.clip(reach, 0., extent))
        s = np.unique(np.r_[np.arange(0., extent, 1.), reach, extent])
        xs, zs, observed = [], [], []
        for m in self.history:
            x, z = m.values(s)
            xs.append(x)
            zs.append(z)
            observed.append(np.where(s <= m.reach, x, np.nan))
        xs, zs, observed = map(np.asarray, (xs, zs, observed))
        measured = median(observed)
        forecast = median(xs)
        # True rail observations take precedence in the historical rail section.
        target = np.where((s <= reach) & np.isfinite(measured), measured, forecast)
        valid = np.isfinite(target)
        if valid.sum() < 4:
            self.previous = None
            return raw
        target = np.interp(s, s[valid], target[valid])
        ztarget = median(zs)
        zvalid = np.isfinite(ztarget)
        ztarget = np.interp(s, s[zvalid], ztarget[zvalid])

        # One cubic spline through both sections gives C2 continuity at their
        # moving boundary. A third-derivative penalty removes median kinks while
        # preserving straight tracks and constant-curvature approximations.
        # A horizon almost coinciding with a regular knot must not create a
        # microscopic terminal span: second-derivative basis entries scale as
        # 1/span**2 and their cancellation can defeat the curvature constraint.
        # Absorb spans below 0.1% of the regular spacing into the last interval.
        minimum_span = max(1e-6, cfg.knot_spacing_m * .001)
        interior = np.arange(cfg.knot_spacing_m, extent - minimum_span, cfg.knot_spacing_m)
        knots = np.r_[np.zeros(4), interior, np.repeat(extent, 4)]
        n = len(knots) - 4
        basis = BSpline(knots, np.eye(n), 3)
        matrix = basis(s)
        weight = np.where((s <= reach) & np.isfinite(measured), 5., 1.)
        regularizer = basis(s, 3) * cfg.knot_spacing_m**2
        normal = matrix.T @ (weight[:, None] * matrix)
        normal += cfg.smoothness * regularizer.T @ regularizer + np.eye(n) * 1e-9
        fit_fallback = False
        try:
            coefficients = np.linalg.solve(normal, matrix.T @ (weight * target))
        except np.linalg.LinAlgError:
            # Strong endpoint regularization can lose rank when squared into
            # normal equations. Solve the identical least-squares objective
            # directly, avoiding that additional condition-number squaring.
            design = np.vstack((np.sqrt(weight)[:, None]*matrix,
                                np.sqrt(cfg.smoothness)*regularizer,
                                np.sqrt(1e-9)*np.eye(n)))
            response = np.r_[np.sqrt(weight)*target, np.zeros(len(regularizer)+n)]
            coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
            fit_fallback = True
        coefficients = self._shape_projection(coefficients, basis, matrix, weight)
        zcoefficients = np.linalg.lstsq(matrix, ztarget, rcond=None)[0]
        alpha = 1.
        limiter_fallback = False
        max_move = max_heading = max_second = 0.
        previous_shape = self._previous_shape_reference()
        if previous_shape is not None:
            old, _ = previous_shape.values(s)
            overlap = np.isfinite(old)
            if overlap.sum() >= 4:
                # Refit the carried previous curve into the same basis. Both
                # ends use linear continuation only outside the overlap.
                old_filled = np.interp(s, s[overlap], old[overlap])
                for at_start in (True, False):
                    idx = np.flatnonzero(overlap)
                    a, b = (idx[0], idx[min(3, len(idx)-1)]) if at_start else (idx[max(0,len(idx)-4)], idx[-1])
                    slope = (old[b] - old[a]) / max(s[b] - s[a], 1e-9)
                    missing = s < s[a] if at_start else s > s[b]
                    anchor = a if at_start else b
                    old_filled[missing] = old[anchor] + slope * (s[missing] - s[anchor])
                old_coeff = np.linalg.lstsq(matrix, old_filled, rcond=None)[0]
                old_coeff = self._shape_projection(old_coeff, basis, matrix, weight)
                delta = coefficients - old_coeff
                check = np.linspace(s[overlap][0], s[overlap][-1], max(32, int(extent * 4)))
                dense = [basis(check, nu) for nu in range(3)]
                changes = [matrix_nu @ delta for matrix_nu in dense]
                if cfg.limit:
                    budgets = [cfg.lateral_step_m + cfg.lateral_step_per_m * check,
                               np.full(len(check), cfg.heading_step_rad),
                               np.full(len(check), cfg.second_derivative_step)]
                    for change, budget in zip(changes, budgets):
                        alpha = min(alpha, float(np.min(budget / np.maximum(np.abs(change), 1e-12))))
                    if alpha < 1.:
                        # Constrain corrections locally. A single global blend
                        # would let a noisy far tail freeze the well-observed
                        # near rails. The QP keeps the whole curve C2 while each
                        # distance independently uses its available correction.
                        # Solve on a 2 m grid, then verify and if necessary
                        # contract the correction on the dense 0.25 m grid.
                        # This avoids thousands of redundant QP constraints.
                        solve_grid = np.unique(np.r_[check[::8], check[-1]])
                        solve_budgets = [cfg.lateral_step_m + cfg.lateral_step_per_m * solve_grid,
                                         np.full(len(solve_grid), cfg.heading_step_rad),
                                         np.full(len(solve_grid), cfg.second_derivative_step)]
                        constraints = np.vstack([basis(solve_grid, nu) / budget[:, None]
                                                 for nu, budget in enumerate(solve_budgets)])
                        # Corrections may approach the statistical target, but
                        # must not pull a well-supported section away from it
                        # merely to accommodate a large error in the far tail.
                        desired_position = basis(solve_grid) @ delta
                        lower_position = np.maximum(-solve_budgets[0],
                                                    np.minimum(0., desired_position) - .0005)
                        upper_position = np.minimum(solve_budgets[0],
                                                    np.maximum(0., desired_position) + .0005)
                        lower = np.r_[lower_position / solve_budgets[0],
                                      np.full(2*len(solve_grid), -1.)]
                        upper = np.r_[upper_position / solve_budgets[0],
                                      np.full(2*len(solve_grid), 1.)]
                        scale = float(np.max(np.diag(normal)))
                        hessian = normal / scale
                        # Fit the reachable correction, not the full distant
                        # outlier: otherwise a 20 m forecast error dominates a
                        # centimetre-scale correction of the observed rails.
                        sample_budget = cfg.lateral_step_m + cfg.lateral_step_per_m*s
                        reachable = np.clip(matrix @ delta, -sample_budget, sample_budget)
                        linear = matrix.T @ (weight * reachable) / scale
                        restrictions = [LinearConstraint(constraints, lower, upper)]
                        shape = self._shape_constraint(basis, old_coeff)
                        if shape is not None:
                            restrictions.append(shape)
                        # The QP is convex with linear bounds: solve it exactly,
                        # and keep SLSQP only for a result that cannot be trusted.
                        exact = solve_qp(hessian, linear, [(c.A, c.lb, c.ub) for c in restrictions])
                        if exact is not None:
                            solved = SimpleNamespace(x=exact, success=True)
                        else:
                            solved = minimize(lambda d: .5*d@hessian@d - linear@d,
                                              alpha*delta, jac=lambda d: hessian@d-linear,
                                              method='SLSQP',
                                              constraints=restrictions,
                                              options={'ftol':1e-10, 'maxiter':100})
                        solved_values = constraints @ solved.x
                        if (solved.success and np.all(solved_values >= lower - 1e-6)
                                and np.all(solved_values <= upper + 1e-6)
                                and self._shape_valid(basis, old_coeff + solved.x)):
                            limited = solved.x
                        else:
                            limited = alpha*delta
                            limiter_fallback = True
                        dense_ratio = max(float(np.max(np.abs(dense[nu] @ limited) / budget))
                                          for nu, budget in enumerate(budgets))
                        limited /= max(1., dense_ratio)
                        coefficients = old_coeff + limited
                        alpha = float(np.linalg.norm(limited) / max(np.linalg.norm(delta), 1e-12))
                        changes = [matrix_nu @ limited for matrix_nu in dense]
                max_move, max_heading, max_second = [float(np.max(np.abs(v))) for v in changes]

        lat = BSpline(knots, coefficients, 3)
        probe = np.linspace(3., max(3., min(reach, 30.)), 30)
        # ego_motion expects curvature in the forward/lateral route frame.
        curvature = float(np.median(lat(probe, 2) / (1 + lat(probe, 1)**2)**1.5))
        g = RouteGeometry(np.zeros(3), SENSOR_BASIS.copy(), knots, coefficients,
                          zcoefficients, 3, extent, 1.15, 2.4, .7, 1e-8,
                          reach, curvature, max(0., float(-latest.heads_end[1])))
        self.previous = Measurement.from_geometry(g)
        self.diagnostics = dict(available=True, history=len(self.history),
                                measured_reach_target_m=rail_target, rails_end_m=reach,
                                reach_motion_m=reach_motion_m,
                                blend=alpha, max_correction_m=max_move,
                                max_slope_correction=max_heading,
                                max_second_derivative_correction=max_second,
                                limiter_fallback=limiter_fallback,
                                fit_fallback=fit_fallback,
                                latest_measurement_age=latest.age)
        return g

    def select(self, xyz, raw, motion=STANDING):
        """Actual analysis ROI, not a display-only smoothed line."""
        geometry = self.update(raw, motion)
        return None if geometry is None else select_with_geometry(xyz, geometry)
