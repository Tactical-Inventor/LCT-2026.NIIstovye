"""Compiled lateral surface evaluation for the initial single-bend fit."""
import numpy as np
from numba import njit
from .train_envelope import WIDTH


@njit(cache=True)
def residual_from_powers(t,t3,at,at3,stations,reference,anchors,values,weight,
                         origin,slope,amplitude,onset,shape,extent,direction,
                         s,low,high,inferred,preserve_tail,wall):
    """Fuse residual arithmetic; NumPy supplies cubes to retain its rounding."""
    n=len(stations);na=len(anchors)
    y=np.empty(n);half=np.empty(n);left=np.empty(n);right=np.empty(n)
    for i in range(n):
        phi=(1-shape)*t[i]*t[i]+shape*t3[i]
        dy=slope+amplitude*((2*(1-shape)*t[i]+3*shape*t[i]*t[i])/(extent-onset))
        y[i]=origin+slope*stations[i]+amplitude*phi
        norm=np.sqrt(1+dy*dy)
        shift=WIDTH/2*dy/norm;half[i]=WIDTH/2/norm
        left[i]=stations[i]+shift;right[i]=stations[i]-shift
    lo,_=lateral_surface(left,s,low,high,inferred,preserve_tail,wall)
    _,hi=lateral_surface(right,s,low,high,inferred,preserve_tail,wall)
    nw=0
    if len(wall):
        for i in range(n):
            if stations[i]>=wall[5]:nw+=1
    out=np.empty(na+3*n+nw)
    for i in range(na):
        phi=(1-shape)*at[i]*at[i]+shape*at3[i]
        out[i]=100.*(origin+slope*anchors[i]+amplitude*phi-values[i])
    j=na+3*n
    for i in range(n):
        out[na+i]=.04*(y[i]-reference[i])
        out[na+n+i]=weight[i]*min(y[i]-half[i]-lo[i]-.02,0.)
        out[na+2*n+i]=weight[i]*max(y[i]+half[i]-hi[i]+.02,0.)
        if len(wall) and stations[i]>=wall[5]:
            station=left[i] if direction>0 else right[i]
            d=y[i]-half[i] if direction>0 else y[i]+half[i]
            z=(station-wall[3])/wall[4]
            measured=wall[0]+wall[1]*z+wall[2]*z*z
            out[j]=600.*min(direction*(d-measured)-.025,0.);j+=1
    return out


@njit(cache=True,error_model='numpy')
def _interp3(x,s,low,high,inferred):
    """np.interp of three value arrays at once, one interval search per point.

    Slopes are computed once per interval as NumPy does, the search starts
    from the previous point's interval, and a point on a station or a
    non-finite slope takes NumPy's own branch, so the values are NumPy's.
    """
    n=len(x);last=len(s)-1
    lo=np.empty(n);hi=np.empty(n);missing=np.empty(n)
    k=max(last,1)
    sl=np.empty(k);sh=np.empty(k);sm=np.empty(k)
    for j in range(last):
        width=s[j+1]-s[j]
        sl[j]=(low[j+1]-low[j])/width;sh[j]=(high[j+1]-high[j])/width
        sm[j]=(inferred[j+1]-inferred[j])/width
    a=0
    for i in range(n):
        v=x[i]
        if last==0 or v<s[0]:
            lo[i]=low[0];hi[i]=high[0];missing[i]=inferred[0]
            if last==0 and v>s[0]:
                lo[i]=low[last];hi[i]=high[last];missing[i]=inferred[last]
            continue
        if v>=s[last]:
            lo[i]=low[last];hi[i]=high[last];missing[i]=inferred[last]
            continue
        while a>0 and v<s[a]:a-=1
        while a<last-1 and v>=s[a+1]:a+=1
        if v==s[a]:
            lo[i]=low[a];hi[i]=high[a];missing[i]=inferred[a]
            continue
        d=v-s[a]
        lo[i]=sl[a]*d+low[a];hi[i]=sh[a]*d+high[a];missing[i]=sm[a]*d+inferred[a]
        if lo[i]!=lo[i] or hi[i]!=hi[i] or missing[i]!=missing[i]:
            _other_side(v,a,s,low,sl,lo,i);_other_side(v,a,s,high,sh,hi,i)
            _other_side(v,a,s,inferred,sm,missing,i)
    return lo,hi,missing


@njit(cache=True,error_model='numpy')
def _other_side(v,a,s,values,slopes,out,i):
    """NumPy's retry from the right end of an interval whose slope is not finite."""
    if out[i]==out[i]:return
    out[i]=slopes[a]*(v-s[a+1])+values[a+1]
    if out[i]!=out[i] and values[a]==values[a+1]:out[i]=values[a]


@njit(cache=True)
def lateral_surface(x,s,low,high,inferred,preserve_tail,wall):
    lo,hi,missing=_interp3(x,s,low,high,inferred)
    if len(s)>1:
        dl=(low[-1]-low[-2])/(s[-1]-s[-2])
        dh=(high[-1]-high[-2])/(s[-1]-s[-2])
        for i in range(len(x)):
            if x[i]>s[-1]:
                lo[i]=low[-2]+dl*(x[i]-s[-2])
                hi[i]=high[-2]+dh*(x[i]-s[-2])
    for i in range(len(x)):
        if preserve_tail and x[i]>s[-1]:
            middle=(lo[i]+hi[i])/2
            lo[i]=middle-(high[-1]-low[-1])/2
            hi[i]=middle+(high[-1]-low[-1])/2
        if len(wall) and x[i]>=wall[5]:
            t=(x[i]-wall[3])/wall[4]
            d=wall[0]+wall[1]*t+wall[2]*t*t
            slope=(wall[1]+2*wall[2]*t)/wall[4]
            factor=np.sqrt(1+slope*slope)
            if wall[6]>0:
                lo[i]=max(lo[i],d)
                if missing[i]>0:hi[i]=max(hi[i],d+wall[7]*factor)
            else:
                hi[i]=min(hi[i],d)
                if missing[i]>0:lo[i]=min(lo[i],d-wall[7]*factor)
    return lo,hi


def prepare(bounds):
    """Prepare immutable arrays once per fit; unsupported policies use reference."""
    if 'lateral_evidence_ranges' in bounds:return None
    s=np.asarray(bounds['stations_m'],float)
    low=np.ascontiguousarray(np.asarray(bounds['lower_m'])[:,0])
    high=np.ascontiguousarray(np.asarray(bounds['upper_m'])[:,0])
    inferred=np.asarray(bounds.get('inferred_axes',np.zeros((len(s),2))))[:,0].astype(float)
    w=bounds.get('far_wall')
    wall=np.array([]) if w is None else np.array([*w['coefficients'],w['station_origin_m'],
        w['station_scale_m'],w['observed_range_m'][0],w['direction'],w['opposite_wall_reference_width_m']])
    return s,low,high,inferred,bool(bounds.get('preserve_tail_width')),wall


def warmup():
    s=np.array([0.,1.]);z=np.zeros(2)
    lateral_surface(s,s,z,z,z,True,np.empty(0))
    anchors=np.zeros((2,3))
    residual_from_powers(s,s,s,s,s,z,anchors[:,0],anchors[:,1],s,0.,0.,1.,0.,1.,2.,1,
                         s,z,z,z,True,np.empty(0))


@njit(cache=True,error_model='numpy')
def outer_surface(x,s,low,high,inferred,preserve_tail,wall_coefficients,wall_origin,wall_scale,
                  wall_start,wall_direction,wall_width,has_wall,ranges,allowance,has_ranges):
    """cloud_route.surface_at for both axes in one pass, with its arithmetic.

    low/high are (stations, axes); the interpolation, tail continuation,
    preserved tail width, turn wall and unobserved-side allowance follow the
    array version step by step, so the bounds are the same to the bit.
    """
    n=len(x);axes=low.shape[1];last=len(s)-1
    lo=np.empty((n,axes));hi=np.empty((n,axes))
    for a in range(axes):
        la,ha,_=_interp3(x,s,np.ascontiguousarray(low[:,a]),np.ascontiguousarray(high[:,a]),inferred)
        if last>0:
            dl=(low[last,a]-low[last-1,a])/(s[last]-s[last-1])
            dh=(high[last,a]-high[last-1,a])/(s[last]-s[last-1])
            for i in range(n):
                if x[i]>s[last]:
                    la[i]=low[last-1,a]+dl*(x[i]-s[last-1])
                    ha[i]=high[last-1,a]+dh*(x[i]-s[last-1])
        for i in range(n):
            lo[i,a]=la[i];hi[i,a]=ha[i]
    if preserve_tail:
        for i in range(n):
            if x[i]>s[last]:
                for a in range(axes):
                    middle=(lo[i,a]+hi[i,a])/2
                    lo[i,a]=middle-(high[last,a]-low[last,a])/2
                    hi[i,a]=middle+(high[last,a]-low[last,a])/2
    if has_wall:
        _,_,missing=_interp3(x,s,inferred,inferred,inferred)
        c=wall_coefficients
        for i in range(n):
            t=(x[i]-wall_origin)/wall_scale
            d=c[0]+c[1]*t+c[2]*t*t
            slope=(c[1]+2*c[2]*t)/wall_scale
            factor=np.sqrt(1+slope*slope)
            if not x[i]>=wall_start:continue
            gone=missing[i]>0
            if wall_direction>0:
                lo[i,0]=max(lo[i,0],d)
                if gone:hi[i,0]=max(hi[i,0],d+wall_width*factor)
            else:
                hi[i,0]=min(hi[i,0],d)
                if gone:lo[i,0]=min(lo[i,0],d-wall_width*factor)
    if has_ranges:
        for i in range(n):
            left=False;right=False
            for r in range(len(ranges)):
                if x[i]>=ranges[r,0]-2 and x[i]<=ranges[r,1]+2:
                    if ranges[r,2]<0:left=True
                    else:right=True
            if has_wall and x[i]>=wall_start:
                if wall_direction>0:left=True
                else:right=True
            if not left:lo[i,0]-=allowance
            if not right:hi[i,0]+=allowance
    return lo,hi
