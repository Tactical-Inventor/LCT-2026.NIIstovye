"""Independent necessary checks. Never move a path to avoid an observed return.

The raw footprint is deliberately an outer bound, not a free-space map. Being
inside it cannot establish correctness; leaving it contradicts measured support.
"""
import time
import numpy as np
from numba import njit
from .rail_path_fast import signed_distance


@njit(cache=True)
def _footprint(local,extent):
    n=min(4096,max(1,int(np.ceil(extent/4))))
    step=max(4.,extent/n)
    lo=np.full(n,np.inf);hi=np.full(n,-np.inf)
    count=np.zeros(n,np.int64);along=np.zeros(n,np.uint8)
    for p in local:
        if not np.isfinite(p[0]+p[1]+p[2]) or p[0]<0 or p[0]>extent:continue
        i=min(n-1,int(p[0]/step))
        lo[i]=min(lo[i],p[1]);hi[i]=max(hi[i],p[1]);count[i]+=1
        along[i]|=np.uint8(1<<min(3,int((p[0]-i*step)*4/step)))
    return lo,hi,count,along,step


def validate_path(path,local):
    tick=time.perf_counter()
    if 'spatial_route' in path:
        from .cloud_route import validate_route
        result=validate_route(path)
        result['elapsed_ms']=1000*(time.perf_counter()-tick)
        return result
    if path.get('status')!='PATH_ESTIMATED':
        return dict(status='INSUFFICIENT_EVIDENCE',reason='no_path',elapsed_ms=0.)
    from .tunnel_path_fast import sample_full_path
    curves=sample_full_path(path)
    extent=path['measured_forward_extent_m']
    checks=[];outside=[];narrow=[]
    refinement=path.get('corridor_refinement',{})
    if refinement.get('bands') and refinement.get('status') not in ('BUDGET_LIMITED','NO_EVIDENCE'):
        bands=np.asarray(refinement['bands'])
        boundaries=np.stack((bands[:,[0,1]],bands[:,[0,2]]),axis=1).reshape(-1,2)
        distances=signed_distance(boundaries,path['parameters']).reshape(-1,2)
        excesses=np.maximum(0.,np.maximum(1.15+distances[:,0],1.15-distances[:,1]))
        for band,normal_excess in zip(refinement['bands'],excesses):
            station,left,right=band[:3]
            values=[float(np.interp(station,curves[key][:,0],curves[key][:,1])) for key in ('left','right')]
            excess=float(normal_excess)
            row=dict(s=station,left_m=left,right_m=right,envelope_left_m=min(values),envelope_right_m=max(values),outside_m=excess)
            checks.append(row)
            if excess>.10001:outside.append(row)
    elif local is not None:
        lo,hi,count,along,step=_footprint(np.asarray(local),extent)
        for i in range(len(lo)):
            # Repeated identical returns cannot make a longitudinal interval observed.
            if count[i]<8 or int(along[i]).bit_count()<2 or hi[i]-lo[i]<.5:continue
            station=min((i+.5)*step,extent)
            values=[float(np.interp(station,curves[key][:,0],curves[key][:,1])) for key in ('left','right')]
            excess=max(lo[i]-min(values),max(values)-hi[i],0.)
            row=dict(s=float(station),left_m=float(lo[i]),right_m=float(hi[i]),
                     envelope_left_m=min(values),envelope_right_m=max(values),outside_m=float(excess))
            # A strip narrower than the train cannot bound its full corridor.
            # Often only one surface returns at distance. Do not force a curve
            # into that strip or mistake missing coverage for a narrow tunnel.
            if hi[i]-lo[i]<2.3:
                narrow.append(row);continue
            checks.append(row)
            # 10 cm is measurement tolerance, not a margin added to the envelope.
            if excess>.10:outside.append(row)
    wall_checks=[];crossings=[]
    for side in (-1,1):
        rows=sorted([r for r in path.get('wall_observations',[]) if r['side']==side and r.get('constraint_used',True)],key=lambda r:r['s'])
        if not rows:continue
        distances=side*signed_distance(np.array([[r['s'],r['d']] for r in rows]),path['parameters'])
        for i,(r,d) in enumerate(zip(rows,distances)):
            violation=1.15-float(d)
            item=dict(s=float(r['s']),d=float(r['d']),side=side,clearance_m=float(d)-1.15,
                      center_outside=bool(d<0),source_indices=r.get('source_indices',[]))
            wall_checks.append(item)
            # Require a neighbouring observation of the same side to corroborate.
            neighbours=[j for j in (i-1,i+1) if 0<=j<len(rows) and abs(rows[j]['s']-r['s'])<=24]
            if violation>.10 and any(distances[j]<1.05 for j in neighbours):crossings.append(item)
    reasons=[]
    if outside:reasons.append('outside_raw_cloud_footprint')
    if crossings:reasons.append('crosses_observed_side_chain')
    if not path.get('rail_compatible',True):reasons.append('incompatible_with_rail_supports')
    if path.get('far_wall_model_conflicts') and path.get('wall_fit_role')!='proposal_only':reasons.append('wall_model_conflict')
    if refinement.get('status') in ('BEST_INCONSISTENT_ARC','NO_FEASIBLE_ARC_FOUND'):reasons.append('no_consistent_single_arc_found')
    limited=path.get('runtime_budget',{}).get('degraded',False)
    if limited:reasons.append('compute_budget_limited')
    status='REJECTED' if any(r!='compute_budget_limited' for r in reasons) else (
        'INSUFFICIENT_EVIDENCE' if limited or len(checks)<3 else 'NO_DETECTED_CONTRADICTION')
    return dict(status=status,reasons=reasons,raw_footprint_checks=checks,
                raw_outside_distance_metric='normal_to_path' if refinement.get('bands') else 'lateral_in_bev',
                insufficient_width_observations=narrow,
                raw_outside=outside,side_checks=wall_checks,side_crossings=crossings,
                max_raw_outside_m=max((r['outside_m'] for r in outside),default=0.),
                max_side_overlap_m=max((-r['clearance_m'] for r in crossings),default=0.),
                first_contradiction_s_m=min([r['s'] for r in outside+crossings],default=None),
                semantics='necessary checks only; inside outer footprint does not prove track identity or 3D clearance',
                elapsed_ms=1000*(time.perf_counter()-tick))


def warmup_validation():
    _footprint(np.zeros((2,3)),10.)
