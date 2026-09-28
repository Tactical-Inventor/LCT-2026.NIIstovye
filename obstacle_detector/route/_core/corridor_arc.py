"""Bounded adjustment of ONE rail-anchored arc to observed outer BEV bounds.

Only outer longitudinal support is used; internal obstacles cannot steer it.
Missing or too-narrow support is explicitly unknown, not free space or a wall.
"""
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize
from .rail_path_fast import _rail_terms,signed_distance,sample_path


@njit(cache=True)
def observed_bands(local,extent):
    n=min(4096,max(1,int(np.ceil(extent/4))))
    step=max(4.,extent/n)
    lo=np.full(n,np.inf);hi=np.full(n,-np.inf)
    low_id=np.full(n,-1,np.int64);high_id=np.full(n,-1,np.int64)
    bits=np.zeros(n,np.uint8);count=np.zeros(n,np.int64)
    for j in range(len(local)):
        s,d,h=local[j]
        if not np.isfinite(s+d+h) or s<0 or s>extent:continue
        i=min(n-1,int(s/step));count[i]+=1
        bits[i]|=np.uint8(1<<min(3,int((s-i*step)/step*4)))
        if d<lo[i]:lo[i]=d;low_id[i]=j
        if d>hi[i]:hi[i]=d;high_id[i]=j
    bands=np.zeros((n,8));used=0;i=0
    while i<n:
        width=1 if i*step<60 else (2 if i*step<120 else 3)
        end=min(n,i+width);l=np.inf;r=-np.inf;li=-1;ri=-1;population=0;occupied=0
        for j in range(i,end):
            population+=count[j]
            b=int(bits[j]);occupied+=((b>>0)&1)+((b>>1)&1)+((b>>2)&1)+((b>>3)&1)
            if lo[j]<l:l=lo[j];li=low_id[j]
            if hi[j]>r:r=hi[j];ri=high_id[j]
        if population>=8 and occupied>=2 and r-l>=2.5:
            bands[used]=np.array([(i+end)*step/2,l,r,li,ri,i*step,min(end*step,extent),population]);used+=1
        i=end
    return bands[:used]


class BudgetExpired(Exception):pass


def continuous_sides(path):
    """Retain side identity only while an observed chain stays continuous.

    A weak side converging onto a better observed side is occlusion/identity
    ambiguity. It cannot constrain a train to the gap between two returns on
    the same surface. Raw points remain available for independent checks.
    """
    chains={}
    for side in (-1,1):
        rows=sorted([r for r in path.get('wall_observations',[]) if r['side']==side],key=lambda r:r['s'])
        kept=[]
        for row in rows:
            if not kept and row['s']>24:break
            if kept and row['s']-kept[-1]['s']>max(16.,1.5*row.get('window_m',4.)):break
            kept.append(row)
        chains[side]=kept
    left,right=chains[-1],chains[1]
    if left and right:
        ls=np.array([r['s'] for r in left]);ld=np.array([r['d'] for r in left])
        rs=np.array([r['s'] for r in right]);rd=np.array([r['d'] for r in right])
        stations=np.arange(max(ls[0],rs[0]),min(ls[-1],rs[-1]),4.)
        widths=np.interp(stations,rs,rd)-np.interp(stations,ls,ld)
        near=widths[stations<40]
        if len(near)>=3:
            collapse=stations[(stations>40)&(widths<max(2.5,.55*np.median(near)))]
            if len(collapse):
                start=float(collapse[0]);scores={side:sum(r.get('quality',1.)*r.get('window_m',4.) for r in chains[side] if r['s']>=40) for side in (-1,1)}
                weaker=min(scores,key=scores.get)
                chains[weaker]=[r for r in chains[weaker] if r['s']<start]
    return chains[-1]+chains[1]


def refine_arc(path,local,deadline=None):
    from .tunnel_path_fast import length_to_plane
    tick=time.perf_counter()
    if path.get('status')!='PATH_ESTIMATED' or local is None:return dict(status='NO_EVIDENCE')
    bands=observed_bands(local,path['measured_forward_extent_m'])
    if len(bands)<3:return dict(status='NO_EVIDENCE',bands=bands.tolist())
    extent=path['measured_forward_extent_m'];cap=min(.025,.8/extent)
    heads=np.asarray(path['heads_local'])[:,:2];targets=np.tile([-path['gauge_m']/2,path['gauge_m']/2],len(heads)//2)
    edges=np.stack([bands[:,[0,1]],bands[:,[0,2]]],axis=1).reshape(-1,2)
    signs=np.tile([-1.,1.],len(bands))
    walls=continuous_sides(path)
    if walls:
        edges=np.r_[edges,np.array([[r['s'],r['d']] for r in walls])]
        signs=np.r_[signs,[r['side'] for r in walls]]
    old=np.asarray(path['parameters']);near=np.asarray(path['rail_only_parameters'])
    # Do not relax an already-good rail anchor to satisfy a distant surface.
    tolerance=max(.10,float(np.max(abs(signed_distance(heads,near)-targets)))+.01)
    def unpack(x):return np.array([x[0],x[1],x[2]*cap])
    def constraints(x,jac=False):
        p=unpack(x);rr,rj=_rail_terms(heads,p,targets)
        d,dj=_rail_terms(edges,p,np.zeros(len(edges)))
        values=np.r_[tolerance-rr,tolerance+rr,signs*d-1.15+.10]
        if not jac:return values
        result=np.r_[-rj,rj,signs[:,None]*dj];result[:,2]*=cap
        return result
    x0=old.copy();x0[2]/=cap
    before=constraints(x0);max_before=max(0.,-float(before.min()))
    status='UNCHANGED';selected=old.copy();iterations=0;success=True
    if max_before>1e-6:
        if deadline is not None and time.perf_counter()>deadline-.014:
            return dict(status='BUDGET_LIMITED',bands=bands.tolist(),max_violation_before_m=max_before,
                        elapsed_ms=1000*(time.perf_counter()-tick))
        # Preserve the original arc wherever constraints already agree with it.
        reference=sample_path(path,path['length_m'],max(1.,path['length_m']/24))['center']
        def objective(x,jac=False):
            rr,jj=_rail_terms(reference,unpack(x),np.zeros(len(reference)))
            if jac:
                jj[:,2]*=cap
                return 2*rr@jj/len(rr)
            return float(rr@rr/len(rr))
        def callback(x):
            if deadline is not None and time.perf_counter()>deadline-.010:raise BudgetExpired()
        try:
            fit=minimize(objective,x0,jac=lambda x:objective(x,True),method='SLSQP',
                         bounds=[(near[0]-.06,near[0]+.06),(-.16,.16),(-1.,1.)],
                         constraints=[dict(type='ineq',fun=constraints,jac=lambda x:constraints(x,True))],
                         callback=callback,options=dict(maxiter=16,ftol=1e-8))
            iterations=int(fit.nit);success=bool(fit.success)
            if np.min(constraints(fit.x))>=-1e-5:
                selected=unpack(fit.x);status='REFINED'
            else:
                status='NO_FEASIBLE_ARC_FOUND'
                if deadline is not None and time.perf_counter()>=deadline-.016:
                    raise BudgetExpired()
                else:
                    # If one circle cannot meet every wall bound, keep the rail
                    # constraint hard and minimize the worst remaining side error.
                    # This remains explicitly rejected; no hidden relaxed success.
                    nrail=2*len(heads)
                    def relaxed(z,jac=False):
                        value=constraints(z[:3],jac)
                        if jac:
                            return np.column_stack((value,np.r_[np.zeros(nrail),np.ones(len(edges))]))
                        value[nrail:]+=z[3];return value
                    seed=near.copy();seed[2]/=cap
                    z0=np.r_[seed,max(0.,-float(constraints(seed)[nrail:].min()))+.001]
                    fallback=minimize(lambda z:z[3]+1e-4*objective(z[:3]),z0,
                                      jac=lambda z:np.r_[1e-4*objective(z[:3],True),1.],method='SLSQP',
                                      bounds=[(near[0]-.06,near[0]+.06),(-.16,.16),(-1.,1.),(0.,100.)],
                                      constraints=[dict(type='ineq',fun=relaxed,jac=lambda z:relaxed(z,True))],
                                      callback=callback,options=dict(maxiter=12,ftol=1e-7))
                    iterations+=int(fallback.nit)
                    margins=constraints(fallback.x[:3])
                    if margins[:nrail].min()>=-1e-5 and max(0.,-float(margins.min()))<max_before-1e-4:
                        displacement=np.max(abs(_rail_terms(reference,unpack(fallback.x[:3]),np.zeros(len(reference)))[0]))
                        # Large changes are justified only by a feasible solution.
                        # An inconsistent fit must not rewrite an entire turn.
                        candidate=fallback.x[:3]
                        if displacement>.75:candidate=x0+(candidate-x0)*(.75/displacement)
                        limited_margins=constraints(candidate)
                        if limited_margins[:nrail].min()>=-1e-5 and max(0.,-float(limited_margins.min()))<max_before-1e-4:
                            selected=unpack(candidate);status='BEST_INCONSISTENT_ARC'
        except BudgetExpired:
            status='BUDGET_LIMITED';success=False
    x=selected.copy();x[2]/=cap;after=constraints(x)
    return dict(status=status,parameters=selected.tolist(),bands=bands.tolist(),
                side_observations=walls,
                max_violation_before_m=max_before,max_violation_after_m=max(0.,-float(after.min())),
                rail_tolerance_m=tolerance,iterations=iterations,optimizer_success=success,
                elapsed_ms=1000*(time.perf_counter()-tick),
                semantics='nearest original constant-curvature arc meeting near-rail and outer-support constraints; not obstacle avoidance or proof of track identity')


def warmup_corridor():observed_bands(np.array([[1.,-2.,1.],[2.,2.,1.]]),10.)


def apply_refinement(path,result):
    """Update all geometry-derived fields together; retain proposal diagnostics."""
    from .tunnel_path_fast import length_to_plane
    path['corridor_refinement']=result
    if result.get('status') in ('NO_EVIDENCE','BUDGET_LIMITED'):return
    original=list(path['parameters']);parameters=result['parameters'];k=parameters[2]
    path['proposal_parameters']=original
    path['parameters']=parameters;path['offset_m']=parameters[0];path['heading_rad']=parameters[1]
    path['curvature_per_m']=k;path['radius_m']=None if k==0 else 1/abs(k)
    path['model']='straight' if k==0 else 'arc'
    path['length_m']=length_to_plane(parameters,path['measured_forward_extent_m'])
    heads=np.asarray(path['heads_local'])[:,:2];targets=np.tile([-path['gauge_m']/2,path['gauge_m']/2],len(heads)//2)
    errors=signed_distance(heads,parameters)-targets
    path['rail_p95_error_m']=float(np.quantile(abs(errors),.95))
    path['residual_rms_m']=float(np.sqrt(np.mean(errors**2)));path['residual_max_m']=float(np.max(abs(errors)))
    path['rail_compatible']=bool(np.max(abs(errors))<=result['rail_tolerance_m']+1e-5)
    path['proposal_wall_model_conflicts']=path.get('far_wall_model_conflicts',[])
    # Parallel, constant-offset walls were only a proposal. They are not an
    # assumption of the final constrained arc (tunnels/platforms can widen).
    path['wall_fit_role']='proposal_only'
    used={(r['side'],r['station']) for r in result.get('side_observations',[])}
    for row in path.get('wall_observations',[]):
        row['constraint_used']=(row['side'],row['station']) in used
        row['fit_residual_m']=float(signed_distance(np.array([[row['s'],row['d']]]),parameters)[0]-path['wall_offsets_m'][0 if row['side']<0 else 1])
        row['fit_inlier']=bool(row['constraint_used'] and abs(row['fit_residual_m'])<.4)
    path['model_fit_consistent']=bool(result['max_violation_after_m']<1e-5 and path['rail_compatible'])
    path['method']='RAIL_ANCHORED_BEV_CONSTRAINED_ARC'
