"""Place the train envelope on a rail reference at the lidar-origin station."""
from __future__ import annotations
import numpy as np

WIDTH = 2.1
HEIGHT = 3.0
VERSION = 'sensor_origin_envelope_v1'


def build_envelope(rail_pose, length=None, level=False, rail_height=None):
    """Front face: plane through sensor (0,0,0), normal to the rail direction.

    The bottom remains on the fitted rail-head plane and lateral center remains
    on the fitted track axis. Only longitudinal origin changes. Default length
    reaches the end of the observed rail reference; it is a display extent,
    not an assumed physical wagon length or a far-route prediction.
    """
    if length is not None and (not np.isfinite(length) or length <= 0):
        raise ValueError('Length must be finite and positive')
    if rail_pose is None:
        return dict(status='UNCONFIRMED',reason='rail_reference_unconfirmed',
                    corners=np.empty((0,3)),origin=None,basis=None)
    original=np.asarray(rail_pose['origin'],dtype=float)
    basis=np.asarray(rail_pose['basis'],dtype=float)
    if original.shape!=(3,) or basis.shape!=(3,3) or not np.isfinite(original).all() or not np.isfinite(basis).all():
        raise ValueError('Invalid rail pose')
    if not np.allclose(basis.T@basis,np.eye(3),atol=1e-8) or np.linalg.det(basis)<.999:
        raise ValueError('Rail basis must be right-handed and orthonormal')
    if level:
        original=original.copy()
        if rail_height is not None:
            if not np.isfinite(rail_height):raise ValueError('Invalid rail height')
            original[2]=rail_height
        forward=basis[:,0].copy();forward[2]=0.
        forward/=np.linalg.norm(forward)
        up=np.array([0.,0.,1.]);basis=np.column_stack((forward,np.cross(up,forward),up))
    observed_length=float(rail_pose['length_m'])
    if not np.isfinite(observed_length) or observed_length<=0:
        raise ValueError('Invalid observed rail length')
    forward=basis[:,0]
    observed_start=float(original@forward)
    observed_end=observed_start+observed_length
    origin=original-forward*observed_start
    extent=observed_end if length is None else float(length)
    if extent<=0:
        return dict(status='UNCONFIRMED',reason='no_rail_reference_in_front',
                    corners=np.empty((0,3)),origin=None,basis=None)
    local=np.array([[s,d,h] for s in [0.,extent] for d in [-WIDTH/2,WIDTH/2] for h in [0.,HEIGHT]])
    corners=local@basis.T+origin
    lidar_local=(-origin)@basis
    return dict(version=VERSION,status='ENVELOPE_ESTIMATED',origin=origin,basis=basis,corners=corners,
                width_m=WIDTH,height_m=HEIGHT,extra_margin_m=0.,length_m=extent,
                length_source='observed_rail_end' if length is None else 'explicit_display_length',
                sensor_origin=[0.,0.,0.],front_plane_normal=forward.tolist(),front_plane_offset=0.,
                lidar_coordinates_in_envelope=lidar_local.tolist(),
                lidar_within_front_section=bool(abs(lidar_local[1])<=WIDTH/2 and 0<=lidar_local[2]<=HEIGHT),
                observed_station_range_m=[observed_start,observed_end],
                extrapolated_prefix_m=max(0.,min(extent,observed_start)),
                extrapolated_suffix_m=max(0.,extent-observed_end),
                path_model='local_straight',clearance_status='NOT_EVALUATED',
                vertical_reference='constant_sensor_z_at_near_rails' if level else 'near_rail_head_plane',
                note='Front face passes through sensor origin; level height from near rail heads.' if level else
                'Front face passes through sensor origin; bottom/center follow rail heads. No obstacle avoidance.')
