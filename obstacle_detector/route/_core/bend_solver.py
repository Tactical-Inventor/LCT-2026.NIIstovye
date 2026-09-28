"""Compiled bounded least squares for the single-bend route fit.

fit_horizontal minimises the same residual from six starting points, each a
two-parameter problem with a thousand residuals. SciPy's trust-region solver
spent most of its time in Python between the evaluations (about 8 ms a start).
Here a Levenberg-Marquardt iteration with the same forward-difference Jacobian,
bounds, parameter scaling and tolerances runs inside one compiled call.
The steps differ from SciPy's reflective trust region, so a start may settle
on a slightly different point; the choice between the starts is unchanged.
"""
import numpy as np
from numba import njit
from .route_surface_fast import residual_from_powers


@njit(cache=True)
def _bend_residual(x, global_bend, stations, reference, anchors, values, weight,
                   origin, slope, extent, direction, s, low, high, inferred, preserve_tail, wall):
    amplitude = x[0]
    if global_bend:
        onset = 0.
        shape = x[1]
    else:
        onset = x[1]
        shape = 1.
    t = np.empty(len(stations)); t3 = np.empty(len(stations))
    for i in range(len(stations)):
        v = max(0., (stations[i] - onset) / (extent - onset))
        t[i] = v; t3[i] = v * v * v
    at = np.empty(len(anchors)); at3 = np.empty(len(anchors))
    for i in range(len(anchors)):
        v = max(0., (anchors[i] - onset) / (extent - onset))
        at[i] = v; at3[i] = v * v * v
    return residual_from_powers(t, t3, at, at3, stations, reference, anchors, values, weight,
                                origin, slope, amplitude, onset, shape, extent, direction,
                                s, low, high, inferred, preserve_tail, wall)


@njit(cache=True)
def _jacobian(x, f0, lower, upper, global_bend, stations, reference, anchors, values, weight,
              origin, slope, extent, direction, s, low, high, inferred, preserve_tail, wall):
    """Forward differences with SciPy's step and bound handling (route_difference)."""
    n = len(x); m = len(f0)
    jac = np.empty((m, n))
    eps = np.sqrt(np.finfo(np.float64).eps)
    for i in range(n):
        sign = 1. if x[i] >= 0 else -1.
        h = eps * sign * max(1., abs(x[i]))
        lower_dist = x[i] - lower[i]; upper_dist = upper[i] - x[i]
        violated = (x[i] + h < lower[i]) or (x[i] + h > upper[i])
        fitting = abs(h) <= max(lower_dist, upper_dist)
        if violated and fitting:
            h = -h
        elif not fitting:
            h = upper_dist if upper_dist >= lower_dist else -lower_dist
        shifted = x.copy(); shifted[i] = x[i] + h
        step = shifted[i] - x[i]
        f1 = _bend_residual(shifted, global_bend, stations, reference, anchors, values, weight,
                            origin, slope, extent, direction, s, low, high, inferred, preserve_tail, wall)
        for k in range(m):
            jac[k, i] = (f1[k] - f0[k]) / step
    return jac


@njit(cache=True, nogil=True)
def solve_bend(x0, lower, upper, x_scale, global_bend, max_nfev, ftol, xtol, gtol,
               stations, reference, anchors, values, weight, origin, slope, extent, direction,
               s, low, high, inferred, preserve_tail, wall):
    """Bounded Levenberg-Marquardt; returns x, cost (0.5 * sum of squares), success."""
    n = len(x0)
    x = np.minimum(np.maximum(x0.copy(), lower), upper)
    f = _bend_residual(x, global_bend, stations, reference, anchors, values, weight,
                       origin, slope, extent, direction, s, low, high, inferred, preserve_tail, wall)
    cost = 0.5 * np.dot(f, f)
    nfev = 1
    mu = -1.
    success = False
    while nfev < max_nfev:
        jac = _jacobian(x, f, lower, upper, global_bend, stations, reference, anchors, values, weight,
                        origin, slope, extent, direction, s, low, high, inferred, preserve_tail, wall)
        # Scaled variables z = x / x_scale, as the x_scale of least_squares.
        js = jac * x_scale
        g = js.T @ f
        a = js.T @ js
        # A bound that the gradient pushes against holds that variable.
        free = np.ones(n, np.bool_)
        for i in range(n):
            if (x[i] <= lower[i] and g[i] > 0) or (x[i] >= upper[i] and g[i] < 0):
                free[i] = False
        gnorm = 0.
        for i in range(n):
            if free[i]:
                gnorm = max(gnorm, abs(g[i]))
        if gnorm < gtol:
            success = True
            break
        if mu < 0:
            mu = 1e-3 * max(1e-12, np.max(np.diag(a)))
        accepted = False
        while nfev < max_nfev:
            system = a.copy()
            for i in range(n):
                system[i, i] += mu
                if not free[i]:
                    for j in range(n):
                        if j != i:
                            system[i, j] = 0.; system[j, i] = 0.
            rhs = -g.copy()
            for i in range(n):
                if not free[i]:
                    rhs[i] = 0.
            dz = np.linalg.solve(system, rhs)
            trial = np.minimum(np.maximum(x + dz * x_scale, lower), upper)
            f_new = _bend_residual(trial, global_bend, stations, reference, anchors, values, weight,
                                   origin, slope, extent, direction, s, low, high, inferred,
                                   preserve_tail, wall)
            nfev += 1
            cost_new = 0.5 * np.dot(f_new, f_new)
            if cost_new < cost:
                step = (trial - x) / x_scale
                reduction = cost - cost_new
                x = trial; f = f_new
                old = cost; cost = cost_new
                mu = max(mu / 3., 1e-15)
                accepted = True
                znorm = np.sqrt(np.sum((x / x_scale) ** 2))
                if reduction < ftol * old or np.sqrt(np.sum(step ** 2)) < xtol * (xtol + znorm):
                    success = True
                break
            mu *= 4.
            if mu > 1e20:
                break
        if not accepted or success:
            if not accepted:
                success = True
            break
    return x, cost, success


@njit(cache=True)
def _convex_terms(v, A, R, signed, line, rail_line, values, reference, weight, lower, upper, limit):
    """Residual and Jacobian of convex_bend.fit for curvature weights v."""
    na = R.shape[0]; ns = A.shape[0]; no = signed.shape[0]; nq = len(v)
    m = 2 * na + 3 * ns + no + nq
    f = np.empty(m); jac = np.zeros((m, nq))
    y = A @ v
    error = R @ v + rail_line - values
    for i in range(na):
        f[i] = 100. * error[i]
        excess = abs(error[i]) - .08
        f[na + i] = 50000. * max(excess, 0.)
        sign = 1. if error[i] > 0 else (-1. if error[i] < 0 else 0.)
        for j in range(nq):
            jac[i, j] = 100. * R[i, j]
            if excess > 0:
                jac[na + i, j] = 50000. * sign * R[i, j]
    base = 2 * na
    for i in range(ns):
        f[base + i] = .5 * (y[i] + line[i] - reference[i])
        f[base + ns + i] = weight[i] * min(y[i] - lower[i], 0.)
        f[base + 2 * ns + i] = weight[i] * max(y[i] - upper[i], 0.)
        for j in range(nq):
            jac[base + i, j] = .5 * A[i, j]
            if y[i] < lower[i]:
                jac[base + ns + i, j] = weight[i] * A[i, j]
            if y[i] > upper[i]:
                jac[base + 2 * ns + i, j] = weight[i] * A[i, j]
    base += 3 * ns
    if no:
        pushed = signed @ v
        for i in range(no):
            f[base + i] = 800. * max(pushed[i] - limit[i], 0.)
            if pushed[i] > limit[i]:
                for j in range(nq):
                    jac[base + i, j] = 800. * signed[i, j]
    base += no
    for j in range(nq):
        f[base + j] = .02 * v[j]
        jac[base + j, j] = .02
    return f, jac


@njit(cache=True)
def solve_convex(q0, low, high, max_nfev, A, R, signed, line, rail_line, values, reference, weight,
                 lower, upper, limit):
    """Bounded Levenberg-Marquardt for convex_bend.fit; returns q and success.

    Every term is linear in the curvature weights or a one-sided square of a
    linear term, so the Jacobian is exact and cheap; the damping carries the
    steps across the kinks where a side becomes active.
    """
    n = len(q0)
    q = np.minimum(np.maximum(q0.copy(), low), high)
    f, jac = _convex_terms(q, A, R, signed, line, rail_line, values, reference, weight, lower, upper, limit)
    cost = 0.5 * np.dot(f, f)
    nfev = 1
    mu = -1.
    success = False
    while nfev < max_nfev:
        g = jac.T @ f
        a = jac.T @ jac
        free = np.ones(n, np.bool_)
        for i in range(n):
            if (q[i] <= low and g[i] > 0) or (q[i] >= high and g[i] < 0):
                free[i] = False
        gnorm = 0.
        for i in range(n):
            if free[i]:
                gnorm = max(gnorm, abs(g[i]))
        if gnorm < 1e-8:
            success = True
            break
        if mu < 0:
            mu = 1e-3 * max(1e-12, np.max(np.diag(a)))
        accepted = False
        while nfev < max_nfev:
            system = a.copy()
            rhs = -g.copy()
            for i in range(n):
                system[i, i] += mu
                if not free[i]:
                    rhs[i] = 0.
                    for j in range(n):
                        if j != i:
                            system[i, j] = 0.; system[j, i] = 0.
            dq = np.linalg.solve(system, rhs)
            trial = np.minimum(np.maximum(q + dq, low), high)
            f_new, jac_new = _convex_terms(trial, A, R, signed, line, rail_line, values, reference,
                                           weight, lower, upper, limit)
            nfev += 1
            cost_new = 0.5 * np.dot(f_new, f_new)
            if cost_new < cost:
                step = trial - q
                reduction = cost - cost_new
                q = trial; f = f_new; jac = jac_new
                old = cost; cost = cost_new
                mu = max(mu / 3., 1e-15)
                accepted = True
                if reduction < 1e-8 * old or np.sqrt(np.sum(step ** 2)) < 1e-8 * (1e-8 + np.sqrt(np.sum(q ** 2))):
                    success = True
                break
            mu *= 4.
            if mu > 1e20:
                break
        if not accepted or success:
            if not accepted:
                success = True
            break
    return q, success
