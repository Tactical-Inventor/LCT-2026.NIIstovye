"""Compiled cubic B-spline evaluation with SciPy's own arithmetic.

The route is two cubic splines on one knot vector, and several stages evaluate
them at every return of the frame. SciPy's BSpline answers correctly but pays a
Python call and a knot search per array; here the same de Boor recursion runs
inside the loop that needs the values. The order of every operation follows
SciPy's evaluate_spline, so the values agree to the last bit (checked in
test_spline_fast.py), and thresholds applied to them decide the same way.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def interval(t, k, x, previous):
    """Knot interval of x as SciPy finds it, starting from a previous answer."""
    n = len(t) - k - 1
    ell = previous
    if ell < k or ell >= n:
        ell = k
    while x < t[ell] and ell != k:
        ell -= 1
    while x >= t[ell + 1] and ell != n - 1:
        ell += 1
    return ell


@njit(cache=True)
def basis(t, x, k, ell, nu, h, hh):
    """Non-zero basis functions (or their nu-th derivative) at x on interval ell."""
    h[0] = 1.0
    for j in range(1, k - nu + 1):
        for i in range(j):
            hh[i] = h[i]
        h[0] = 0.0
        for n in range(1, j + 1):
            ind = ell + n
            xb = t[ind]
            xa = t[ind - j]
            if xb == xa:
                h[n] = 0.0
                continue
            w = hh[n - 1] / (xb - xa)
            h[n - 1] += w * (xb - x)
            h[n] = w * (x - xa)
    for j in range(k - nu + 1, k + 1):
        for i in range(j):
            hh[i] = h[i]
        h[0] = 0.0
        for n in range(1, j + 1):
            ind = ell + n
            xb = t[ind]
            xa = t[ind - j]
            if xb == xa:
                # SciPy clears h[nu] here, not h[n]; kept for identical values.
                h[nu] = 0.0
                continue
            w = j * hh[n - 1] / (xb - xa)
            h[n - 1] -= w
            h[n] = w


@njit(cache=True)
def evaluate(t, c, k, x, nu=0):
    """Values of one spline at x (1-D), as BSpline(t, c, k)(x, nu)."""
    out = np.empty(len(x))
    h = np.empty(2 * k + 2)
    hh = np.empty(2 * k + 2)
    ell = k
    for i in range(len(x)):
        ell = interval(t, k, x[i], ell)
        basis(t, x[i], k, ell, nu, h, hh)
        value = 0.0
        for a in range(k + 1):
            value += c[ell + a - k] * h[a]
        out[i] = value
    return out


@njit(cache=True)
def evaluate_pair(t, c0, c1, k, x, out0, out1):
    """Two splines sharing knots at one x each, as one BSpline with two columns."""
    h = np.empty(2 * k + 2)
    hh = np.empty(2 * k + 2)
    ell = k
    for i in range(len(x)):
        ell = interval(t, k, x[i], ell)
        basis(t, x[i], k, ell, 0, h, hh)
        v0 = 0.0
        v1 = 0.0
        for a in range(k + 1):
            v0 += c0[ell + a - k] * h[a]
            v1 += c1[ell + a - k] * h[a]
        out0[i] = v0
        out1[i] = v1


def warmup():
    t = np.r_[np.zeros(4), np.full(4, 1.)]
    c = np.zeros(4)
    x = np.linspace(0., 1., 3)
    evaluate(t, c, 3, x, 0)
    evaluate(t, c, 3, x, 1)
    evaluate_pair(t, c, c, 3, x, np.empty(3), np.empty(3))
