"""Height slice centered in the rail-aligned train envelope; no lateral clipping."""
import numpy as np
from numba import njit


@njit(cache=True)
def _slice(xyz,origin,basis,low,high):
    mask=np.zeros(len(xyz),np.bool_)
    indices=np.empty(len(xyz),np.int64)
    local=np.empty((len(xyz),3),np.float64); n=0; extent=0.
    for i in range(len(xyz)):
        x=xyz[i,0]-origin[0]; y=xyz[i,1]-origin[1]; z=xyz[i,2]-origin[2]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)): continue
        s=x*basis[0,0]+y*basis[1,0]+z*basis[2,0]
        extent=max(extent,s)
        h=x*basis[0,2]+y*basis[1,2]+z*basis[2,2]
        if not low<=h<=high: continue
        d=x*basis[0,1]+y*basis[1,1]+z*basis[2,1]
        if s>=0 and np.isfinite(s) and np.isfinite(d):
            mask[i]=True;indices[n]=i;local[n,0]=s;local[n,1]=d;local[n,2]=h;n+=1
    return mask,indices[:n],local[:n],extent


def warmup_bev():
    for dtype in (np.float32,np.float64):
        _slice(np.zeros((1,3),dtype=dtype),np.zeros(3),np.eye(3),.8,1.6)
        _source_points(np.zeros((1,3),dtype=dtype),np.zeros(1,np.int64))


@njit(cache=True)
def _source_points(xyz,indices):
    points=np.empty((len(indices),3),dtype=xyz.dtype)
    for j in range(len(indices)):
        for axis in range(3): points[j,axis]=xyz[indices[j],axis]
    return points


def extract_bev(xyz, envelope, thickness=.8):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Expected Nx3 points')
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError('Thickness must be finite and positive')
    if envelope['status'] != 'ENVELOPE_ESTIMATED':
        raise ValueError('Rail reference is unconfirmed')
    center = envelope['height_m'] / 2
    low, high = center - thickness / 2, center + thickness / 2
    mask,indices,local,extent = _slice(xyz,np.asarray(envelope['origin']),np.asarray(envelope['basis']),low,high)
    return dict(indices=indices, mask=mask, points=_source_points(xyz,indices),
                local=local, height_bounds_m=np.array([low, high]),measured_forward_extent_m=extent)
