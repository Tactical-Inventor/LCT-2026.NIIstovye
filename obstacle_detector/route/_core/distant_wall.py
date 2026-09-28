"""Range-adaptive outer wall evidence; never center a route on a lone wall."""
import numpy as np
from numba import njit
from scipy.optimize import least_squares
from .train_envelope import WIDTH, HEIGHT


def distance_edges(extent):
    edges=[0.]
    while edges[-1]<extent:
        s=edges[-1]
        width=8. if s<48 else (12. if s<96 else (16. if s<160 else 24.))
        edges.append(min(extent,s+width))
    if len(edges)>2 and edges[-1]-edges[-2]<.4*(edges[-2]-edges[-3]):
        edges.pop(-2)
    return np.asarray(edges)


@njit(cache=True)
def collect(local,edges,z0,dz):
    n=len(edges)-1
    low=np.full((n,2),np.inf);high=np.full((n,2),-np.inf)
    smin=np.full(n,np.inf);smax=np.full(n,-np.inf)
    wall_ids=np.full((n,2),-1,np.int64)
    walls=np.zeros((n,2,2))
    count=np.zeros(n,np.int64);height_bits=np.zeros(n,np.uint32)
    for j in range(len(local)):
        s,d,h=local[j]
        if not np.isfinite(s+d+h) or s<0 or s>edges[-1]:continue
        i=min(n-1,np.searchsorted(edges,s,side='right')-1)
        smin[i]=min(smin[i],s);smax[i]=max(smax[i],s)
        low[i,0]=min(low[i,0],d);high[i,0]=max(high[i,0],d)
        low[i,1]=min(low[i,1],h);high[i,1]=max(high[i,1],h)
        height=h-z0-dz*s
        if not .2<=height<=HEIGHT+1.6:continue
        count[i]+=1
        height_bits[i]|=np.uint32(1<<min(15,int(height/.3)))
        for side in range(2):
            if wall_ids[i,side]<0 or (side==0 and d<walls[i,side,1]) or (side==1 and d>walls[i,side,1]):
                wall_ids[i,side]=j;walls[i,side,0]=s;walls[i,side,1]=d
    return low,high,smin,smax,walls,wall_ids,count,height_bits


def turn_direction(stations,low,high,slope):
    # Compare the approach to the terminal chain, beyond nearby platforms.
    near=(stations>=stations[-1]*.4)&(stations<=stations[-1]*.65)
    far=stations>=stations[-1]*.8
    if near.sum()<2 or far.sum()<2:return 0
    corrected_low=low[:,0]-slope*stations;corrected_high=high[:,0]-slope*stations
    left=float(np.median(corrected_low[far])-np.median(corrected_low[near]))
    right=float(np.median(corrected_high[far])-np.median(corrected_high[near]))
    if left>1.2 and right> .35:return 1
    if right< -1.2 and left< -.35:return -1
    return 0


def fit_wall(edges,walls,ids,count,height_bits,direction,extent):
    if not direction:return dict(status='NO_COHERENT_TURN',direction=0)
    side=0 if direction>0 else 1
    rows=[]
    for i in range(len(count)):
        s,d=walls[i,side]
        required=6 if edges[i]<60 else (3 if edges[i]<120 else 2)
        layers=int(height_bits[i]).bit_count()
        if ids[i,side]<0 or count[i]<required or layers<2 or s<max(32.,.4*extent):continue
        rows.append(dict(s=float(s),d=float(d),local_index=int(ids[i,side]),
                         point_count=int(count[i]),height_layers=layers,
                         window_m=float(edges[i+1]-edges[i]),minimum_count=required))
    if len(rows)<3 or rows[-1]['s']-rows[0]['s']<16:
        return dict(status='INSUFFICIENT_WALL_EVIDENCE',direction=direction,observations=rows)
    s=np.array([r['s'] for r in rows]);d=np.array([r['d'] for r in rows])
    origin=float(s[0]);scale=max(1.,float(s[-1]-s[0]));t=(s-origin)/scale
    design=np.column_stack((np.ones(len(s)),t,t*t))
    start=np.linalg.lstsq(design,d,rcond=None)[0]
    fit=least_squares(lambda c:design@c-d,start,jac=lambda c:design,loss='soft_l1',f_scale=.15)
    errors=design@fit.x-d
    tolerance=.2+.0025*extent
    c=fit.x;end=(extent-origin)/scale
    slope=(c[1]+2*c[2]*end)/scale
    ok=bool(np.quantile(abs(errors),.8)<tolerance and direction*slope>.015)
    return dict(status='OBSERVED_TURN_WALL' if ok else 'WALL_FIT_CONFLICT',direction=direction,
                side=-1 if direction>0 else 1,observations=rows,coefficients=c.tolist(),
                station_origin_m=origin,station_scale_m=scale,observed_range_m=[float(s[0]),float(s[-1])],
                residual_p80_m=float(np.quantile(abs(errors),.8)),residual_limit_m=tolerance,
                end_slope=float(slope),semantics='range-adaptive height-supported observed side, not a centerline')


def wall_at(wall,s):
    c=np.asarray(wall['coefficients']);t=(np.asarray(s)-wall['station_origin_m'])/wall['station_scale_m']
    return c[0]+c[1]*t+c[2]*t*t,(c[1]+2*c[2]*t)/wall['station_scale_m']


@njit(cache=True)
def opposite_support(local,edges,coefficients,origin,scale,direction,z0,dz):
    n=len(edges)-1;count=np.zeros(n,np.int64);bits=np.zeros(n,np.uint32);width=np.zeros(n)
    for s,d,h in local:
        if not np.isfinite(s+d+h) or s<0 or s>edges[-1]:continue
        height=h-z0-dz*s
        if not .2<=height<=HEIGHT+1.6:continue
        t=(s-origin)/scale
        wall=coefficients[0]+coefficients[1]*t+coefficients[2]*t*t
        slope=(coefficients[1]+2*coefficients[2]*t)/scale
        distance=direction*(d-wall)/np.sqrt(1+slope*slope)
        if distance<WIDTH+.4:continue
        i=min(n-1,np.searchsorted(edges,s,side='right')-1)
        count[i]+=1;bits[i]|=np.uint32(1<<min(15,int(height/.3)))
        width[i]=max(width[i],distance)
    return count,bits,width


def apply_wall(bounds,raw_low,raw_high,wall,local,edges,z0,dz):
    if wall['status']!='OBSERVED_TURN_WALL':return
    stations=np.asarray(bounds['stations_m']);low=np.asarray(bounds['lower_m']);high=np.asarray(bounds['upper_m'])
    inferred=np.asarray(bounds['inferred_axes'])
    measured,slope=wall_at(wall,stations);normal_factor=np.sqrt(1+slope*slope)
    counts,bits,widths=opposite_support(local,edges,np.asarray(wall['coefficients']),
                                      wall['station_origin_m'],wall['station_scale_m'],wall['direction'],z0,dz)
    required=np.where(edges[:-1]<60,6,np.where(edges[:-1]<120,3,2))
    supported=(counts>=required)&np.array([int(b).bit_count()>=2 for b in bits])
    bins=np.minimum(len(counts)-1,np.searchsorted(edges,stations,side='right')-1)
    usable=supported[bins]&(stations>=.35*stations[-1])
    reference=float(np.median(widths[bins[usable]])) if usable.any() else WIDTH+.8
    reference=max(WIDTH+.4,min(reference,8.))
    active=stations>=wall['observed_range_m'][0]
    if wall['direction']>0:
        low[active,0]=np.maximum(low[active,0],measured[active])
        missing=active&(~supported[bins]|((raw_high[:,0]-measured)/normal_factor<WIDTH+.4)|inferred[:,0])
        high[missing,0]=low[missing,0]+reference*normal_factor[missing]
    else:
        high[active,0]=np.minimum(high[active,0],measured[active])
        missing=active&(~supported[bins]|((measured-raw_low[:,0])/normal_factor<WIDTH+.4)|inferred[:,0])
        low[missing,0]=high[missing,0]-reference*normal_factor[missing]
    inferred[missing,0]=True
    bounds.update(lower_m=low.tolist(),upper_m=high.tolist(),inferred_axes=inferred.tolist(),far_wall=wall)
    wall['opposite_wall_reference_width_m']=reference
    wall['opposite_wall_inferred_windows']=int(missing.sum())
    wall['opposite_support_counts']=counts.tolist()
