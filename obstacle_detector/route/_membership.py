"""Membership in the corridor with a 0.7 m margin below its rail-head profile."""
import numpy as np
from numba import njit, prange
from scipy.interpolate import BSpline, PPoly
from ._core.train_envelope import WIDTH, HEIGHT

BELOW_RAIL_M = 0.7
TOLERANCE_M = 1e-8
DEGREE = 3
BIN_M = 0.25


@njit(cache=True)
def _value(s,knots,coefficients):
    j=min(len(knots)-2,max(0,np.searchsorted(knots,s,side='right')-1))
    u=s-knots[j];a,b,c,d=coefficients[:,j]
    return ((a*u+b)*u+c)*u+d,(3*a*u+2*b)*u+c,6*a*u+2*b


@njit(cache=True,parallel=True)
def _select(xyz,origin,basis,extent,knots,cy,cz,ylo,yhi,zlo,zhi,local):
    mask=np.zeros(len(xyz),np.bool_)
    record=len(local)>0
    half=WIDTH/2;tol=1e-8;bins=len(ylo)
    for i in prange(len(xyz)):
        px,py,pz=xyz[i]
        if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(pz)):continue
        if px==0 and py==0 and pz==0:continue  # sensor placeholder
        px-=origin[0];py-=origin[1];pz-=origin[2]
        x=px*basis[0,0]+py*basis[1,0]+pz*basis[2,0]
        y=px*basis[0,1]+py*basis[1,1]+pz*basis[2,1]
        z=px*basis[0,2]+py*basis[1,2]+pz*basis[2,2]
        # A point in a normal section differs from its center station by <=half.
        lo=max(0.,x-half-tol);hi=min(extent,x+half+tol)
        if lo>hi:continue
        # Necessary bounds over that station window. The foot of the normal is
        # exactly |d| away from the point, so a point inside cannot be further
        # than half from every profile value the window can take. Rejecting
        # here only skips work; survivors still face the exact test below.
        b=min(bins-1,max(0,int(x/BIN_M)))
        if y<ylo[b]-half-tol or y>yhi[b]+half+tol:continue
        if z<zlo[b]-BELOW_RAIL_M-tol or z>zhi[b]+HEIGHT+tol:continue
        yl,dl,_=_value(lo,knots,cy);yh,dh,_=_value(hi,knots,cy)
        fl=(lo-x)+(yl-y)*dl;fh=(hi-x)+(yh-y)*dh
        if fl>tol or fh< -tol:continue  # before/after the normal end faces
        station=min(hi,max(lo,x))
        for _ in range(48):
            center,slope,second=_value(station,knots,cy)
            f=(station-x)+(center-y)*slope
            if abs(f)<=1e-10:break
            if f>0:hi=station
            else:lo=station
            denominator=1+slope*slope+(center-y)*second
            candidate=station-f/denominator if denominator>1e-12 else (lo+hi)/2
            if candidate<=lo or candidate>=hi:candidate=(lo+hi)/2
            station=candidate
        center,slope,_=_value(station,knots,cy)
        normal=((y-center)-(x-station)*slope)/np.sqrt(1+slope*slope)
        bottom,_,_=_value(station,knots,cz)
        # The fitted lower profile follows the rail heads along the route.
        if abs(normal)<=half+tol and bottom-BELOW_RAIL_M-tol<=z<=bottom+HEIGHT+tol:
            mask[i]=True
            # Same station solution that accepted the point; no second search.
            if record:local[i,0]=station;local[i,1]=normal;local[i,2]=z-bottom
    return mask


def _extrema(poly,edges):
    """Exact min and max of the piecewise cubic over each bin.

    Sampling could miss an extremum between samples, so the turning points are
    taken from the derivative's own roots instead of from a grid.
    """
    values=poly(edges)
    low=np.minimum(values[:-1],values[1:]);high=np.maximum(values[:-1],values[1:])
    turning=np.atleast_1d(poly.derivative().roots(extrapolate=False)).ravel()
    turning=turning[np.isfinite(turning)]
    turning=turning[(turning>=edges[0])&(turning<=edges[-1])]
    if len(turning):
        at=poly(turning)
        index=np.clip(np.searchsorted(edges,turning,side='right')-1,0,len(low)-1)
        np.minimum.at(low,index,at);np.maximum.at(high,index,at)
    return low,high


def _window(low,high,reach):
    """Widen each bin to the station window a point in it can project onto."""
    pad=np.full(reach,np.inf)
    span=np.lib.stride_tricks.sliding_window_view
    return (span(np.concatenate((pad,low,pad)),2*reach+1).min(axis=1),
            span(np.concatenate((-pad,high,-pad)),2*reach+1).max(axis=1))


def prepare(knots,lateral_coefficients,height_coefficients,extent):
    """Piecewise form plus the per-bin bounds the selection kernel screens with."""
    knots=np.asarray(knots,float)
    lateral=PPoly.from_spline(BSpline(knots,np.asarray(lateral_coefficients,float),DEGREE))
    height=PPoly.from_spline(BSpline(knots,np.asarray(height_coefficients,float),DEGREE))
    extent=float(extent)
    edges=np.arange(0.,extent+BIN_M,BIN_M)
    if len(edges)<2:edges=np.array([0.,max(BIN_M,extent)])
    reach=int(np.ceil((WIDTH/2+TOLERANCE_M)/BIN_M))
    ylo,yhi=_window(*_extrema(lateral,edges),reach)
    zlo,zhi=_window(*_extrema(height,edges),reach)
    return extent,lateral.x,lateral.c,height.c,ylo,yhi,zlo,zhi


def _terms(path,envelope):
    route=path['spatial_route']
    prepared=prepare(route['knots'],route['lateral_coefficients'],
                     route['height_coefficients'],path['measured_forward_extent_m'])
    return (np.asarray(envelope['origin']),np.asarray(envelope['basis']))+prepared


def corridor_mask(xyz,path,envelope):
    return _select(xyz,*_terms(path,envelope),np.empty((0,3)))


def corridor_local(xyz,path,envelope):
    """Mask plus the route-local station, normal offset and height per row.

    Rows outside the corridor stay NaN; selected rows carry the very values the
    membership test used, so the two can never disagree.
    """
    local=np.full((len(xyz),3),np.nan)
    return _select(xyz,*_terms(path,envelope),local),local


def select_prepared(xyz,origin,basis,prepared,with_local=True):
    """Apply an already prepared geometry to another cloud, estimating nothing."""
    local=np.full((len(xyz),3),np.nan) if with_local else np.empty((0,3))
    return _select(xyz,np.asarray(origin,float),np.asarray(basis,float),*prepared,local),local


def warmup_membership():
    prepared=prepare(np.array([0.,0.,0.,0.,1.,1.,1.,1.]),np.zeros(4),np.zeros(4),1.)
    for dtype in [np.float32,np.float64]:
        for out in [np.empty((0,3)),np.full((1,3),np.nan)]:
            _select(np.zeros((1,3),dtype),np.zeros(3),np.eye(3),*prepared,out)
