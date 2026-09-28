"""Single-frame sparse rail tracking and one straight/constant-curvature path.

No obstacle scores or BEV occupancy enter trajectory fitting. Coordinates are
forward/lateral/up in the accepted rail reference, with s=0 at the sensor plane.
"""
import time
import numpy as np
from numba import njit
from scipy.optimize import least_squares
from .rail_pose_fast import estimate_rail_pose, robust_fit, _peaks, _voxel_first
from ._reference import pairs_from_peaks
from ._reference import default_config
from .train_envelope import build_envelope

VERSION = 'sparse_rail_path_v1'


@njit(cache=True)
def _far_points(xyz, origin, basis):
    ids=np.empty(len(xyz),np.int64); q=np.empty((len(xyz),3),np.float64); n=0
    for i in range(len(xyz)):
        x=xyz[i,0]-origin[0]; y=xyz[i,1]-origin[1]; z=xyz[i,2]-origin[2]
        s=x*basis[0,0]+y*basis[1,0]+z*basis[2,0]
        d=x*basis[0,1]+y*basis[1,1]+z*basis[2,1]
        h=x*basis[0,2]+y*basis[1,2]+z*basis[2,2]
        if np.isfinite(s+d+h) and 13<=s<=83 and abs(d)<12 and abs(h)<.8:
            ids[n]=i; q[n,0]=s; q[n,1]=d; q[n,2]=h; n+=1
    return ids[:n],q[:n]


def signed_distance(points, parameters):
    """Stable exact signed perpendicular distance to an oriented circle/line."""
    offset, heading, curvature=parameters
    c,s=np.cos(heading),np.sin(heading)
    x=points[:,0]*c+(points[:,1]-offset)*s
    y=-points[:,0]*s+(points[:,1]-offset)*c
    return (2*y-curvature*(x*x+y*y))/(1+np.sqrt((1-curvature*y)**2+(curvature*x)**2))


@njit(cache=True)
def _rail_terms(points,parameters,targets):
    offset,yaw,k=parameters;c=np.cos(yaw);s=np.sin(yaw)
    residual=np.empty(len(points));jac=np.empty((len(points),3))
    for i in range(len(points)):
        x=points[i,0];y=points[i,1]-offset
        a=x*c+y*s;b=-x*s+y*c
        rho=np.sqrt((1-k*b)**2+(k*a)**2)
        distance=(2*b-k*(a*a+b*b))/(1+rho)
        residual[i]=distance-targets[i]
        jac[i,0]=(k*a*s-(1-k*b)*c)/rho
        jac[i,1]=-a/rho
        jac[i,2]=(distance*distance-a*a-b*b)/(2*rho)
    return residual,jac


def fit_path(heads, gauge):
    """Compare an exact pair of parallel lines to concentric circular rails."""
    heads=np.asarray(heads,float)
    targets=np.tile([-gauge/2,gauge/2],len(heads)//2)
    terms=lambda p:_rail_terms(heads,np.asarray(p),targets)
    residual=lambda p:terms(p)[0]
    line=least_squares(lambda p:residual([*p,0]),[0.,0.],
                       jac=lambda p:terms([*p,0])[1][:,:2],
                       bounds=([-.6,-.3],[.6,.3]),loss='soft_l1',f_scale=.035,max_nfev=25)
    straight=np.r_[line.x,0.]
    arc=least_squares(residual,straight,jac=lambda p:terms(p)[1],bounds=([-.6,-.3,-.025],[.6,.3,.025]),
                      loss='soft_l1',f_scale=.035,x_scale=[.1,.03,.002],max_nfev=35)
    # A small apparent bend in the near supports is not sufficient evidence.
    span=float(np.ptp(heads[:,0]))
    line_rms=float(np.sqrt(np.mean(np.minimum(residual(straight)**2,.15**2))))
    arc_rms=float(np.sqrt(np.mean(np.minimum(residual(arc.x)**2,.15**2))))
    use_arc=(len(heads)>=10 and span>=15 and arc_rms<.7*line_rms and
             line_rms-arc_rms>.012 and abs(arc.x[2])*span*span/8>.045)
    parameters=arc.x if use_arc else straight
    errors=residual(parameters)
    return dict(model='arc' if use_arc else 'straight',parameters=parameters.tolist(),
                offset_m=float(parameters[0]),heading_rad=float(parameters[1]),
                curvature_per_m=float(parameters[2]),
                radius_m=None if parameters[2]==0 else float(1/abs(parameters[2])),
                line_rms_m=line_rms,arc_rms_m=arc_rms,
                residual_rms_m=float(np.sqrt(np.mean(errors**2))),
                residual_max_m=float(np.max(abs(errors))),
                model_fit_consistent=bool(np.quantile(abs(errors),.9)<.12),
                optimizer_success=bool(arc.success if use_arc else line.success))


def sample_path(path, length=60., step=.25):
    """Metric offsets: boundaries are exactly 1.15 m normal to the centerline."""
    if 'spatial_route' in path:
        from .cloud_route import sample_route
        return sample_route(path,length,step)
    u=np.arange(0,length+step/2,step)
    offset,heading,k=path['parameters']
    theta=heading+k*u
    # np.sinc avoids cancellation for both small curvature and straight lines.
    chord=u*np.sinc(k*u/(2*np.pi))
    center=np.column_stack((chord*np.cos(heading+k*u/2),offset+chord*np.sin(heading+k*u/2)))
    normal=np.column_stack((-np.sin(theta),np.cos(theta)))
    return dict(distance=u,center=center,left=center-1.15*normal,right=center+1.15*normal)


def estimate_path(xyz, rail=None, envelope=None,deadline=None):
    """Accept XYZ alone; optional within-call rail/envelope reuse avoids duplicate work."""
    tick=time.perf_counter()
    if rail is None: rail=estimate_rail_pose(xyz)
    if envelope is None: envelope=build_envelope(rail['pose'])
    if envelope['status']!='ENVELOPE_ESTIMATED':
        return dict(version=VERSION,status='UNCONFIRMED',confidence=0.,
                    reason='rail_reference_unconfirmed',elapsed_ms=1000*(time.perf_counter()-tick))
    cfg=default_config()
    origin=np.asarray(envelope['origin']); basis=np.asarray(envelope['basis'])
    near=(rail['head_centers']-origin)@basis
    heads=list(near.copy()); source=list(rail['indices']); sections=[]; rejected=[]
    gauge=float(np.median(np.linalg.norm(near[1::2,:2]-near[::2,:2],axis=1)))
    ids,q=_far_points(np.asarray(xyz),origin,basis)
    if len(q):
        selected=_voxel_first(q,.035); q=q[selected]; ids=ids[selected]
    centers=(near[::2]+near[1::2])/2
    history=list(centers)
    misses=0; ambiguous=0;budget_limited=False
    start=max(19.5,float(centers[:,0].max())+3)
    for station in np.arange(start,80,3):
        if deadline is not None and time.perf_counter()>=deadline:
            budget_limited=True;break
        recent=np.array(history[-6:])
        # Local quadratic is only a search predictor; output is ONE circle/line.
        t=recent[:,0]-station
        degree=2 if len(recent)>=5 else 1
        a=np.column_stack([t**j for j in range(degree, -1, -1)])
        coef=robust_fit(a,recent[:,1],.035)
        center=float(coef[-1]); heading=float(np.arctan(coef[-2]))
        if abs(heading)>.6 or abs(center)>10: break
        window=min(6.,3.+max(0.,station-18)*.12)
        chunk=np.flatnonzero(abs(q[:,0]-station)<window/2+2)
        local=q[chunk].copy()
        if len(local)<30:
            misses+=1; rejected.append(dict(s=float(station),reason='too_few_points'))
            if misses>=3: break
            continue
        tangent=np.array([np.cos(heading),np.sin(heading)])
        lateral=np.array([-tangent[1],tangent[0]])
        delta=local[:,:2]-[station,center]
        along=delta@tangent; across=delta@lateral
        profile=local.copy(); profile[:,0]=along*3/window; profile[:,1]=across
        peaks,heights,floor=_peaks(profile,0.,0.,0.,cfg)
        candidates=[p for p in pairs_from_peaks(peaks,gauge,.10,cfg) if abs(p[0])<.4]
        candidates.sort(key=lambda p:abs(p[0])+2*abs(p[1]-gauge))
        accepted=[]
        for pair in candidates[:3]:
            pair_heads=[]; pair_ids=[]
            for peak in pair[2:]:
                height=heights[np.argmin(abs(peaks-peak))]+peak*floor[0]+floor[1]
                chosen=np.flatnonzero((abs(along)<window/2)&(abs(across-peak)<.10)&
                                      (local[:,2]>height-.055)&(local[:,2]<height+.035))
                if len(chosen)<6 or np.ptp(along[chosen])<1.: break
                design=np.column_stack((along[chosen],np.ones(len(chosen))))
                dc=robust_fit(design,across[chosen],.025)
                hc=robust_fit(design,local[chosen,2],.018)
                if (abs(dc[0])>.2 or abs(hc[0])>.15 or
                    np.median(abs(design@dc-across[chosen]))>.045 or
                    np.median(abs(design@hc-local[chosen,2]))>.035): break
                point=np.r_[np.array([station,center])+lateral*dc[1],hc[1]]
                best=chosen[np.argmin(np.sum((local[chosen]-point)**2,axis=1))]
                pair_heads.append(point); pair_ids.append(int(ids[chunk[best]]))
            if len(pair_heads)==2:
                measured=np.linalg.norm(np.diff(np.array(pair_heads)[:,:2],axis=0))
                if abs(measured-gauge)<=.10:
                    accepted.append((pair_heads,pair_ids,abs(pair[0])+2*abs(measured-gauge)))
        if not accepted:
            misses+=1; rejected.append(dict(s=float(station),reason='no_verified_pair'))
            if misses>=3: break
            continue
        accepted.sort(key=lambda x:x[2])
        is_ambiguous=bool(len(accepted)>1 and accepted[1][2]-accepted[0][2]<.10)
        ambiguous+=int(is_ambiguous)
        hh,ii,_=accepted[0]; heads.extend(hh); source.extend(ii)
        history.append(np.mean(hh,axis=0)); misses=0
        sections.append(dict(s=float(station),alternatives=len(accepted),ambiguous=is_ambiguous))
    heads=np.array(heads)
    path=fit_path(heads,gauge)
    lo=float(heads[:,0].min()); hi=float(heads[:,0].max())
    ambiguity=ambiguous+rail['report'].get('ambiguous_sections',0)
    confidence=float(rail['report']['confidence']*np.exp(-path['residual_rms_m']/.10)*
                     (0.65 if ambiguity else 1.)*(1. if path['model_fit_consistent'] else .35))
    if budget_limited:confidence*=.7
    path.update(version=VERSION,status='PATH_ESTIMATED',confidence=confidence,
                confidence_semantics='heuristic geometric evidence, not probability or clearance',
                gauge_m=gauge,heads_local=heads.tolist(),support_indices=[int(i) for i in source],
                observed_forward_range_m=[lo,hi],sections=sections,rejected_sections=rejected,
                ambiguous_sections=ambiguity,branch_policy='one locally continuous verified rail pair',
                height_reference='near rail-head plane; extrapolated beyond near supports',
                rail_search_budget_limited=budget_limited,
                clearance_status='NOT_EVALUATED',elapsed_ms=1000*(time.perf_counter()-tick))
    return path


def warmup_path():
    _rail_terms(np.zeros((2,3)),np.zeros(3),np.zeros(2))
    for dtype in (np.float32,np.float64):
        _far_points(np.zeros((1,3),dtype=dtype),np.zeros(3),np.eye(3))
