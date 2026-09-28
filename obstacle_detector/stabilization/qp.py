"""Exact solution of the small convex QP that limits route corrections.

The route filter's correction QP - a positive definite quadratic of about
twenty spline coefficients under a few hundred two-sided linear bounds - used
to go to SLSQP, which approaches the optimum from an identity Hessian and took
about twenty iterations, 30-100 ms per frame and up to 0.5 s. With linear
constraints and a known Hessian the optimum is one least-distance problem
(Lawson and Hanson, ch. 23): factor the Hessian, map the bounds onto the
distance from the unconstrained optimum, and solve that with NNLS. On 1479
QPs recorded from three recordings it matched or improved the SLSQP objective
on every one, in 2-10 ms.
"""
import numpy as np
from scipy.linalg import null_space, solve_triangular
from scipy.optimize import nnls


def _cholesky(matrix):
    scale = max(float(np.max(np.abs(np.diag(matrix)))), 1e-300)
    for ridge in (0., 1e-12, 1e-10, 1e-8):
        try:
            return np.linalg.cholesky(matrix + ridge * scale * np.eye(len(matrix)))
        except np.linalg.LinAlgError:
            continue
    return None


def solve_qp(hessian, linear, blocks, tolerance=1e-7):
    """Minimise 0.5 x'Hx - l'x subject to lb <= A x <= ub for each (A, lb, ub).

    Rows with lb == ub are equalities. Returns x, or None when the result
    cannot be trusted: a failed factorisation or NNLS, an infeasible problem,
    or a bound violated by more than tolerance. The caller then falls back.
    """
    n = len(linear)
    eq_A, eq_b, in_G, in_h = [], [], [], []
    for A, lb, ub in blocks:
        A = np.atleast_2d(np.asarray(A, float))
        lb = np.broadcast_to(np.asarray(lb, float), (len(A),))
        ub = np.broadcast_to(np.asarray(ub, float), (len(A),))
        equal = np.isfinite(lb) & np.isfinite(ub) & (ub == lb)
        if equal.any():
            eq_A.append(A[equal]); eq_b.append(lb[equal])
        low = ~equal & np.isfinite(lb)
        high = ~equal & np.isfinite(ub)
        if low.any():
            in_G.append(A[low]); in_h.append(lb[low])
        if high.any():
            in_G.append(-A[high]); in_h.append(-ub[high])
    hessian = np.asarray(hessian, float)
    linear = np.asarray(linear, float)
    # Equalities: x = particular + null-space coordinates.
    if eq_A:
        equalities = np.vstack(eq_A)
        particular = np.linalg.lstsq(equalities, np.concatenate(eq_b), rcond=None)[0]
        free = null_space(equalities)
    else:
        particular, free = np.zeros(n), np.eye(n)
    if free.shape[1]:
        factor = _cholesky(free.T @ hessian @ free)
        if factor is None:
            return None
        # 0.5 y'Hy - l'y = 0.5 |L'y - f|^2 + const
        f = solve_triangular(factor, free.T @ (linear - hessian @ particular), lower=True)
        if in_G:
            G = np.vstack(in_G)
            h = np.concatenate(in_h)
            # z = L'y - f turns G y >= h into M z >= b; the closest such z to
            # the origin comes from NNLS on [M'; b'] u ~ (0, ..., 0, 1).
            M = solve_triangular(factor, (G @ free).T, lower=True).T
            b = h - G @ particular - M @ f
            design = np.vstack((M.T, b[None]))
            target = np.zeros(len(design))
            target[-1] = 1.
            try:
                u, _ = nnls(design, target, maxiter=50 * design.shape[1])
            except RuntimeError:
                return None
            residual = design @ u - target
            if not abs(residual[-1]) > 1e-12:
                return None
            f = f - residual[:-1] / residual[-1]
        x = particular + free @ solve_triangular(factor.T, f, lower=False)
    else:
        x = particular
    if not np.all(np.isfinite(x)):
        return None
    for A, lb, ub in blocks:
        values = np.atleast_2d(np.asarray(A, float)) @ x
        if np.any(values < np.asarray(lb, float) - tolerance) or np.any(values > np.asarray(ub, float) + tolerance):
            return None
    return x
