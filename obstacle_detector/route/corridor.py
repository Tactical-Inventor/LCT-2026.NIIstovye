"""Public single-cloud interface without dataset paths or output files."""
from dataclasses import dataclass, field
import time
import numpy as np
from ._core.tunnel_path_fast import process_frame, warmup_tunnel
from ._core.rail_pose_fast import warmup as warmup_rails
from ._core.rail_path_fast import warmup_path
from ._core.train_envelope import WIDTH, HEIGHT
from ._membership import (corridor_mask, corridor_local, select_prepared, prepare,
                          warmup_membership, BELOW_RAIL_M, TOLERANCE_M, DEGREE)

# Lateral uncertainty of the route centre, see RouteGeometry.lateral_uncertainty_m.
UNCERTAINTY_FLOOR_M = 0.07      # along the rails the route was fitted to
UNCERTAINTY_ON_TURN = 0.034     # metres per metre beyond rails that turn
UNCERTAINTY_ON_STRAIGHT = 0.0006  # metres per square metre beyond rails that run straight
TURN_CURVATURE = 4e-4           # 1/m: from here on the rails turn (radius 2.5 km)


@dataclass(frozen=True)
class RouteGeometry:
    """Reference frame and cubic splines that defined the selected region.

    Local coordinates of a sensor row p are x,y,z = (p - origin) @ basis. For a
    station s the route centre is y(s) and its lower profile is z0(s); evaluate
    them as BSpline(knots, lateral_coefficients, degree) and
    BSpline(knots, height_coefficients, degree). Stations run 0..extent_m; s is
    a longitudinal coordinate, not arc length. The frame is re-estimated per
    frame, so two frames are comparable only through these fields.

    rails_end_m is the station of the last own rail centre the route was fitted
    to; beyond it the route is a forecast. rails_end_curvature_per_m is the
    mean curvature of those rail centres. lateral_uncertainty_m says how far
    the track may lie beside the route centre. heads_end_m is the station of
    the last near rail head the height profile is anchored to: up to there
    z0(s) is the measured rail-head level, beyond it an extrapolation.
    """
    origin: np.ndarray = field(repr=False)
    basis: np.ndarray = field(repr=False)
    knots: np.ndarray = field(repr=False)
    lateral_coefficients: np.ndarray = field(repr=False)
    height_coefficients: np.ndarray = field(repr=False)
    degree: int
    extent_m: float
    half_width_m: float
    gauge_height_m: float
    below_rail_m: float
    tolerance_m: float
    rails_end_m: float | None = None
    rails_end_curvature_per_m: float = 0.0
    heads_end_m: float | None = None

    def lateral_uncertainty_m(self, stations):
        """How far the track may lie beside the route centre, per station.

        A 90th percentile, measured against the rails later frames saw
        (route_eval/far_truth.py, beyond.py): along the rails the centre is
        within about 0.07 m of them; beyond their end it is a forecast, and
        how fast it goes wrong depends on whether the measured rails turn
        (rails_end_curvature_per_m, one parabola through them). Measured over
        all six recordings, 2.5 to 40 m beyond the rails: where they run
        straight the error grows with the square of the distance (0.12 m at
        12.5 m, 0.46 m at 27.5 m, 1.1 m at 40 m, where some tracks have begun
        to turn), where they turn with the distance itself (0.18 m at 2.5 m,
        0.49 m at 12.5 m, 1.0 m at 27.5 m) - the track may keep its curvature,
        tighten into the curve or straighten out, and nothing measured tells
        which. None when the route does not say where its rails end.
        """
        if self.rails_end_m is None:
            return None
        beyond = np.maximum(0., np.asarray(stations, float) - self.rails_end_m)
        turning = min(1., abs(self.rails_end_curvature_per_m) / TURN_CURVATURE)
        return (UNCERTAINTY_FLOOR_M + turning * UNCERTAINTY_ON_TURN * beyond
                + (1. - turning) * UNCERTAINTY_ON_STRAIGHT * beyond ** 2)


@dataclass(frozen=True)
class CorridorResult:
    """Original input points and the status of the geometry used to select them."""
    points: np.ndarray = field(repr=False)
    indices: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    status: str
    reasons: tuple[str, ...]
    extent_m: float
    elapsed_ms: float
    local: np.ndarray | None = field(default=None, repr=False)
    gauge_mask: np.ndarray | None = field(default=None, repr=False)
    below_mask: np.ndarray | None = field(default=None, repr=False)
    geometry: RouteGeometry | None = field(default=None, repr=False)


def _checked(xyz):
    xyz=np.asarray(xyz)
    if xyz.ndim!=2 or xyz.shape[1]!=3:
        raise ValueError('xyz must have shape (N, 3)')
    if xyz.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise TypeError('xyz must contain float32 or float64 coordinates in metres')
    return xyz


def _split(local,indices):
    local=np.empty((0,3)) if local is None else local[indices]
    # The gauge starts at the profile; the two masks partition the region.
    gauge=local[:,2]>=-TOLERANCE_M
    return local,gauge,~gauge


def extract_corridor(xyz, *, return_details=False, return_geometry=False, select=True):
    """Return original XYZ rows inside and below the full-range corridor.

    Input: (N,3) float32/float64 sensor coordinates in metres; forward -Y,
    up +Z. No downsampling, output coordinate conversion or cloud-file IO.
    Selection includes obstacles, uncertain parts and points up to 0.7 m below
    the rail-anchored lower profile, within the corridor width and end faces.
    Use return_details=True to inspect REJECTED/INSUFFICIENT_EVIDENCE status.
    Use return_geometry=True to also receive route-local coordinates, the gauge
    and lower-layer masks and the route geometry; it needs return_details.
    If no route can be estimated, return an empty cloud (status UNCONFIRMED).
    select=False estimates the route and its geometry but selects no rows,
    for a caller that selects with a different (smoothed) geometry anyway.
    """
    xyz=_checked(xyz)
    if return_geometry and not return_details:
        raise ValueError('return_geometry requires return_details=True')
    tick=time.perf_counter();local=None;geometry=None
    if not len(xyz):
        mask=np.zeros(0,bool);status='UNCONFIRMED';reasons=('empty_cloud',);extent=0.
    else:
        result=process_frame(xyz,budget_ms=None)
        path=result['path'];envelope=result['envelope']
        if 'spatial_route' not in path:
            mask=np.zeros(len(xyz),bool);status='UNCONFIRMED'
            reasons=(path.get('reason','route_not_estimated'),);extent=0.
        else:
            if not select:
                mask=np.zeros(len(xyz),bool)
                if return_geometry:geometry=_geometry(path,envelope)
            elif return_geometry:
                mask,local=corridor_local(xyz,path,envelope)
                geometry=_geometry(path,envelope)
            else:mask=corridor_mask(xyz,path,envelope)
            validation=path['validation'];status=validation['status']
            reasons=tuple(validation.get('reasons',[]));extent=float(path['measured_forward_extent_m'])
    indices=np.flatnonzero(mask);points=xyz[indices]
    gauge=below=None
    if return_geometry:local,gauge,below=_split(local,indices)
    result=CorridorResult(points,indices,mask,status,reasons,extent,
                          1000*(time.perf_counter()-tick),local,gauge,below,geometry)
    return result if return_details else result.points


def select_with_geometry(xyz, geometry):
    """Select rows of a later cloud with a geometry estimated on an earlier frame.

    Nothing is estimated or validated here: the route comes from the caller, so
    the result carries the status REUSED_GEOMETRY and never a verdict of its
    own. Keep the status and reasons of the frame the geometry came from.

    The reference frame is attached to the sensor, so a geometry goes stale at
    the speed of the vehicle: standing still it stays exact, and at 80 km/h it
    is already 22 m behind after one second. How long to reuse one is the
    caller's decision, not something this package can check.
    """
    xyz=_checked(xyz);tick=time.perf_counter()
    if not len(xyz):
        mask=np.zeros(0,bool);local=np.empty((0,3))
    else:
        prepared=prepare(geometry.knots,geometry.lateral_coefficients,
                         geometry.height_coefficients,geometry.extent_m)
        mask,local=select_prepared(xyz,geometry.origin,geometry.basis,prepared)
    indices=np.flatnonzero(mask);points=xyz[indices]
    local,gauge,below=_split(local,indices)
    return CorridorResult(points,indices,mask,'REUSED_GEOMETRY',
                          ('geometry_estimated_on_an_earlier_frame',),
                          float(geometry.extent_m),1000*(time.perf_counter()-tick),
                          local,gauge,below,geometry)


def _geometry(path,envelope):
    route=path['spatial_route']
    rails=path.get('route_fit',{}).get('far_rails') or {}
    end=rails.get('observed_end_m')
    return RouteGeometry(np.asarray(envelope['origin'],float),np.asarray(envelope['basis'],float),
                         np.asarray(route['knots'],float),np.asarray(route['lateral_coefficients'],float),
                         np.asarray(route['height_coefficients'],float),DEGREE,
                         float(path['measured_forward_extent_m']),WIDTH/2,HEIGHT,BELOW_RAIL_M,TOLERANCE_M,
                         None if end is None else float(end),float(rails.get('curvature_per_m',0.)),
                         float(np.asarray(path['heads_local'])[:,0].max()))


def warmup():
    """Optionally compile kernels once before timing a stream of frames."""
    warmup_rails();warmup_path();warmup_tunnel();warmup_membership()
