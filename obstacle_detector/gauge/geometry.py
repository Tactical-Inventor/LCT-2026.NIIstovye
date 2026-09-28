"""Distance along the route for the outer gauge margin."""
import numpy as np
from scipy.interpolate import BSpline


def path_distance(geometry, stations):
    """Arc length from route station zero, by 25 cm trapezoidal quadrature."""
    g = geometry
    grid = np.unique(np.r_[np.arange(0., g.extent_m, .25), g.extent_m])
    dy = BSpline(g.knots, g.lateral_coefficients, g.degree)(grid, 1)
    dz = BSpline(g.knots, g.height_coefficients, g.degree)(grid, 1)
    speed = np.sqrt(1+dy*dy+dz*dz)
    length = np.r_[0., np.cumsum(np.diff(grid)*(speed[1:]+speed[:-1])/2)]
    return np.interp(stations, grid, length)
