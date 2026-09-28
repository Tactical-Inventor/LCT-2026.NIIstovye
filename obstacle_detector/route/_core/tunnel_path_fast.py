"""One full-range arc anchored to rails and estimated from tunnel side surfaces.

Based on wall_arc_path's separate wall offsets and original far-plane geometry.
No occupancy/collision cost, tunnel centering, or neighboring frame is used.
"""
import time
import numpy as np
from numba import njit
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from .rail_path_fast import signed_distance, sample_path
from .frame_runtime import frame_compute

VERSION='full_range_obstacle_aware_single_bend_v8'


@njit(cache=True)
def _grid(points,low,high,step,bin_size):
    """Bounded grid of unique longitudinal/height cells, not return density."""
    if len(points)==0:
        return np.zeros((0,0),np.uint32),np.zeros((0,0)),np.zeros((0,0)),0.
    dmin=points[0,1];dmax=dmin;xmax=0.
    for p in points:
        dmin=min(dmin,p[1]);dmax=max(dmax,p[1]);xmax=max(xmax,p[0])
    dmin=np.floor(dmin/bin_size)*bin_size
    nx=int(xmax/step)+1;nd=int((dmax-dmin)/bin_size)+1
    # Skip a pathological grid allocation without shortening the measured horizon.
    if nx*nd>2000000:
        return np.zeros((0,0),np.uint32),np.zeros((0,0)),np.zeros((0,0)),dmin
    bits=np.zeros((nx,nd),np.uint32);sx=np.zeros((nx,nd));sd=np.zeros((nx,nd))
    for p in points:
        i=int(p[0]/step);j=int((p[1]-dmin)/bin_size)
        along=min(3,int((p[0]-i*step)/step*4))
        height=min(3,max(0,int((p[2]-low)/(high-low)*4)))
        flag=np.uint32(1 << (along*4+height))
        if not bits[i,j]&flag:
            bits[i,j] |= flag; sx[i,j]+=p[0];sd[i,j]+=p[1]
    return bits,sx,sd,dmin


@njit(cache=True)
def _count_bits(bits):
    out=np.zeros(bits.shape,np.int64)
    for i in range(bits.shape[0]):
        for j in range(bits.shape[1]):
            v=bits[i,j];n=0
            while v:
                n+=1;v=v&(v-1)
            out[i,j]=n
    return out


@njit(cache=True)
def _ridge_values(peaks,counts,bits,sx,sd,i,end,sparse):
    out=np.empty((len(peaks),5));n=0
    for j in peaks:
        lo=max(0,j-1);hi=min(counts.shape[1],j+2)
        mask=np.uint32(0);count=0;xs=0.;ds=0.;occupied=0
        for a in range(i,end):
            station_count=0
            for b in range(lo,hi):
                mask|=bits[a,b];station_count+=counts[a,b];xs+=sx[a,b];ds+=sd[a,b]
            count+=station_count
            if station_count>0: occupied+=1
        levels=0;longitudinal=0
        for h in range(4):
            found=False
            for a in range(4):
                if mask&(1<<(a*4+h)): found=True
            if found: levels+=1
        for a in range(4):
            if mask&(15<<(a*4)): longitudinal+=1
        strong=levels>=3 and longitudinal>=2 and count>=6
        if not strong and not (sparse and count>=1): continue
        out[n,0]=xs/count;out[n,1]=ds/count;out[n,2]=count;out[n,3]=1. if strong else (.35 if count>=3 else .15);out[n,4]=j;n+=1
    return out[:n]


@njit(cache=True)
def _fit_terms(v,points,rail_count,targets,sides,weights,cap,anchor):
    """Exact circle distance and analytic Jacobian, stable through k=0."""
    n=len(points);res=np.empty(n+1);jac=np.zeros((n+1,5))
    k=v[2]*cap;c=np.cos(v[1]);s=np.sin(v[1])
    for i in range(n):
        x=points[i,0];y=points[i,1]-v[0]
        a=x*c+y*s;b=-x*s+y*c
        rho=np.sqrt((1-k*b)**2+(k*a)**2)
        distance=(2*b-k*(a*a+b*b))/(1+rho)
        if i<rail_count:
            scale=25.;target=targets[i]
        else:
            j=i-rail_count;scale=weights[j]*5.
            col=3 if sides[j]<0 else 4;target=v[col];jac[i,col]=-scale
        res[i]=(distance-target)*scale
        jac[i,0]=(k*a*s-(1-k*b)*c)/rho*scale
        jac[i,1]=-a/rho*scale
        jac[i,2]=(distance*distance-a*a-b*b)/(2*rho)*cap*scale
    res[n]=(v[0]-anchor)/.025;jac[n,0]=40.
    return res,jac


def wall_observations(bev,rail_path):
    """Trace each wall independently; gaps and alternative ridges are reported."""
    points=bev['local'];step=4.;bin_size=.25
    bits,sx,sd,dmin=_grid(points,*bev['height_bounds_m'],step,bin_size)
    counts=_count_bits(bits);tracks={-1:[],1:[]};ambiguous=0
    yaw=rail_path['heading_rad'];k=rail_path['curvature_per_m']
    i=0
    while i<len(counts):
        width=1 if i*step<60 else (2 if i*step<120 else 3)
        end=min(len(counts),i+width)
        row=counts[i:end].sum(axis=0);smoothed=np.convolve(row,[.25,.5,.25],mode='same')
        sparse=i*step>=60
        peaks,_=find_peaks(np.r_[0.,smoothed,0.],height=.25 if sparse else 2.5,distance=2)
        peaks=peaks-1
        candidates=[dict(s=float(v[0]),d=float(v[1]),cells=int(v[2]),station=i,quality=float(v[3]),window_m=(end-i)*step,grid_column=int(v[4]))
                    for v in _ridge_values(peaks,counts,bits,sx,sd,i,end,sparse)]
        proposals={}
        for sign in (-1,1):
            history=tracks[sign]
            if not history:
                if i*step>20: continue
                eligible=[c for c in candidates if rail_path['gauge_m']/2+.1<sign*(c['d']-rail_path['offset_m']-np.tan(yaw)*c['s'])<10]
                if not eligible: continue
                chosen=max(eligible,key=lambda c:c['cells'])
            else:
                last=history[-1];gap=(i+width/2)*step-last['s']
                if gap>max(24,5*width*step): continue
                slope=np.tan(yaw+k*last['s'])
                if len(history)>=3:
                    older=history[-3]
                    slope=np.clip((last['d']-older['d'])/(last['s']-older['s']),-.9,.9)
                def error(c): return abs(c['d']-(last['d']+(c['s']-last['s'])*slope))
                eligible=sorted([c for c in candidates if error(c)<.65+.045*gap],key=lambda c:error(c)-.015*c['cells'])
                if not eligible: continue
                if len(eligible)>1 and abs(error(eligible[1])-error(eligible[0]))<.12:
                    ambiguous+=1
                chosen=eligible[0]
            proposals[sign]=chosen
        # A sparse return cannot be both walls. Keep the continuously tracked
        # side; a wall lost long ago cannot steal the other side's observation.
        if len(proposals)==2:
            left,right=proposals[-1],proposals[1]
            if right['d']-left['d']<rail_path['gauge_m']*.5:
                gaps={sign:(c['s']-tracks[sign][-1]['s'] if tracks[sign] else 0.) for sign,c in proposals.items()}
                del proposals[max(gaps,key=gaps.get)]
                ambiguous+=1
        for sign,chosen in proposals.items(): tracks[sign].append(dict(chosen,side=sign))
        i=end
    # Sparse evidence gains weight only through a locally coherent chain,
    # never merely from the number of repeated returns at one location.
    for history in tracks.values():
        for j,row in enumerate(history):
            if row['s']<60 or row['quality']>=.65: continue
            neighbors=history[max(0,j-1):min(len(history),j+2)]
            if len(neighbors)<3: continue
            ss=np.array([r['s'] for r in neighbors]);dd=np.array([r['d'] for r in neighbors])
            ds=np.diff(ss)
            if ds.min()>1. and ds.max()<65 and abs(np.diff(np.diff(dd)/ds)[0])<.18:
                row['quality_from_cells']=row['quality'];row['quality']=.65
    return tracks[-1]+tracks[1],ambiguous


def length_to_plane(parameters,plane):
    """First crossing of the unchanged sensor-forward plane s=plane."""
    _,yaw,k=parameters
    if abs(k)<1e-12: return float(plane/np.cos(yaw))
    a=np.sin(yaw)+k*plane
    if abs(a)>=1: raise ValueError('Arc cannot reach measured forward plane')
    end=np.arcsin(a)
    delta_cos=k*plane*(2*np.sin(yaw)+k*plane)/(np.cos(yaw)+np.sqrt(1-a*a))
    delta=np.arctan2(k*plane*np.cos(yaw)+np.sin(yaw)*delta_cos,np.cos(end-yaw))
    return float(delta/k)


def observed_wall_conflicts(rows,errors,rail_end):
    """Test coherent local residuals on observed chains, not a fraction of range."""
    conflicts=[]
    for side in (-1,1):
        indices=[i for i,r in enumerate(rows) if r['side']==side and r['s']>rail_end]
        found=[]
        for a in range(max(0,len(indices)-2)):
            chosen=indices[a:a+4]
            if len(chosen)<3:continue
            stations=[rows[i]['s'] for i in chosen]
            if max(stations)-min(stations)<8:continue
            values=np.asarray(errors)[chosen];median=float(np.median(values))
            if abs(median)>.5 and np.mean(np.sign(values)==np.sign(median))>=.75:
                found.append((chosen,median))
        if len(indices)>=2:
            chosen=indices[-2:];values=np.asarray(errors)[chosen]
            if rows[chosen[-1]]['s']-rows[chosen[0]]['s']>=8 and min(abs(values))>1 and values[0]*values[1]>0:
                found.append((chosen,float(np.median(values))))
        if found:
            chosen,median=max(found,key=lambda item:abs(item[1]))
            conflicts.append(dict(side=side,median_signed_residual_m=median,observations=len(chosen),
                                  observed_range_m=[rows[chosen[0]]['s'],rows[chosen[-1]]['s']],
                                  reason='coherent_observed_wall_residual'))
    return conflicts


def fit_tunnel_path(bev,rail_path,observations=None,allow_fit=True):
    tick=time.perf_counter()
    if rail_path['status']!='PATH_ESTIMATED': return dict(rail_path)
    plane=max(.01,float(bev['measured_forward_extent_m']))
    rows,ambiguous=wall_observations(bev,rail_path) if observations is None else observations
    heads=np.asarray(rail_path['heads_local'])[:,:2];gauge=rail_path['gauge_m']
    targets=np.tile([-gauge/2,gauge/2],len(heads)//2)
    anchor=rail_path['offset_m']
    cap=min(.025,.8/plane)
    old=np.asarray(rail_path['parameters'])
    start=np.array([anchor,np.clip(old[1],-.159,.159),np.clip(old[2]/cap,-.999,.999),-2.5,2.5])
    wall=np.array([[r['s'],r['d']] for r in rows]).reshape(-1,2)
    sides=np.array([r['side'] for r in rows])
    weights=np.sqrt([r.get('quality',1.)*r.get('window_m',4.)/4. for r in rows])
    for sign,j in [(-1,3),(1,4)]:
        if np.any(sides==sign):
            values=signed_distance(wall[sides==sign],[anchor,start[1],start[2]*cap])
            start[j]=np.clip(np.median(values),-19 if sign<0 else .1,-.1 if sign<0 else 19)
    wall_span=float(np.ptp(wall[:,0])) if len(wall) else 0.
    supported=allow_fit and len(rows)>=8 and wall_span>=20 and (len(set(sides))==2 or observations is not None)

    points=np.concatenate([heads,wall])
    def terms(v): return _fit_terms(np.asarray(v),points,len(heads),targets,sides,weights,cap,anchor)
    def residual(v): return terms(v)[0]
    def jacobian(v): return terms(v)[1]

    # Wall offsets have a side sign, but NO constraint derived from train width.
    # Even a wall inside the envelope must not turn this into a clearance search.
    lower=[anchor-.06,-.16,-1.,-20.,.05];upper=[anchor+.06,.16,1.,-.05,20.]
    if supported:
        active=np.array([0,1,2]+([3] if np.any(sides<0) else [])+([4] if np.any(sides>0) else []))
        def unpack(x,columns,base):
            value=base.copy();value[columns]=x;return value
        fitted=least_squares(lambda x:residual(unpack(x,active,start)),start[active],
                             jac=lambda x:jacobian(unpack(x,active,start))[:,active],
                             bounds=(np.asarray(lower)[active],np.asarray(upper)[active]),loss='soft_l1',f_scale=1.,
                             max_nfev=35,x_scale=np.array([.04,.03,.3,1.,1.])[active])
        v=unpack(fitted.x,active,start);success=bool(fitted.success)
        # Compare against a true straight solution, with the SAME independent wall offsets.
        columns=active[active!=2];base=v.copy();base[2]=0.
        straight=least_squares(lambda x:residual(unpack(x,columns,base)),
                               v[columns],jac=lambda x:jacobian(unpack(x,columns,base))[:,columns],
                               bounds=(np.array(lower)[columns],np.array(upper)[columns]),
                               loss='soft_l1',f_scale=1.,max_nfev=25)
        sv=unpack(straight.x,columns,base)
        loss=lambda x:float(np.sum(np.sqrt(1+residual(x)**2)-1))
        if loss(sv)<=loss(v)+.75 or abs(v[2]*cap)*wall_span**2/8<.06:
            v=sv;success=bool(straight.success)
        parameters=[float(v[0]),float(v[1]),float(v[2]*cap)]
        offsets=[float(v[3]),float(v[4])]
        method='RAILS_AND_INDEPENDENT_WALL_OFFSETS' if len(set(sides))==2 else 'RAILS_AND_SINGLE_OBSERVED_WALL'
    else:
        parameters=[float(start[0]),float(start[1]),float(start[2]*cap)]
        offsets=[float(start[3]),float(start[4])];success=True
        method='RAIL_ONLY_FULL_RANGE_EXTRAPOLATION' if allow_fit else 'RAIL_ONLY_BUDGET_EXTRAPOLATION'
    rail_errors=signed_distance(heads,parameters)-targets
    wall_errors=signed_distance(wall,parameters)-np.where(sides<0,offsets[0],offsets[1])
    wr=float(np.median(abs(wall_errors))) if len(wall_errors) else None
    rail_p95=float(np.quantile(abs(rail_errors),.95))
    compatible=rail_p95<=max(.10,rail_path['residual_max_m']+.035)
    wall_ok=supported and wr<.35
    # The global median can hide a coherent far-wall disagreement among many
    # good near observations. Report it separately; never bend around points.
    far_conflicts=observed_wall_conflicts(rows,wall_errors,rail_path['observed_forward_range_m'][1])
    confidence=rail_path['confidence']*np.exp(-max(0.,rail_p95-.03)/.1)
    confidence*=1. if wall_ok else .55
    if wall_ok and len(set(sides))==1:confidence*=.75
    if not compatible or not success: confidence*=.3
    if ambiguous: confidence*=.85
    if far_conflicts: confidence*=.35
    if not allow_fit:confidence*=.5
    for row,error in zip(rows,wall_errors):
        row['fit_residual_m']=float(error)
        row['fit_inlier']=bool(wall_ok and compatible and abs(error)<.4)
    supported_stations=[r['s'] for r in rows if r['fit_inlier']]
    evidence_end=max(rail_path['observed_forward_range_m'][1],max(supported_stations,default=0.))
    evidence_confidence=float(confidence)
    confidence*=.35+.65*min(1.,evidence_end/plane)
    result=dict(rail_path)
    k=parameters[2]
    result.update(version=VERSION,parameters=parameters,offset_m=parameters[0],heading_rad=parameters[1],
                  curvature_per_m=k,radius_m=None if k==0 else 1/abs(k),model='straight' if k==0 else 'arc',
                  confidence=float(confidence),method=method,measured_forward_extent_m=plane,
                  observed_evidence_confidence=evidence_confidence,
                  confidence_semantics='full-range heuristic: rail/wall consistency, ambiguity, and observed range coverage; not probability',
                  last_supported_forward_m=float(evidence_end),
                  length_m=length_to_plane(parameters,plane),wall_observations=rows,
                  wall_offsets_m=offsets,wall_median_residual_m=wr,wall_ambiguous_sections=ambiguous,
                  wall_supported=bool(wall_ok),rail_compatible=bool(compatible),
                  wall_support_sides=sorted(set(int(side) for side in sides)),
                  rail_p95_error_m=rail_p95,wall_observed_range_m=None if not rows else [float(wall[:,0].min()),float(wall[:,0].max())],
                  rail_only_parameters=rail_path['parameters'],optimizer_success=success,
                  residual_rms_m=float(np.sqrt(np.mean(rail_errors**2))),residual_max_m=float(np.max(abs(rail_errors))),
                  far_wall_model_conflicts=far_conflicts,
                  wall_observed_end_by_side_m={str(side):max((r['s'] for r in rows if r['side']==side),default=0.) for side in (-1,1)},
                  wall_input='multilayer_source_evidence' if observations is not None else 'legacy_height_slice',
                  model_fit_consistent=bool(compatible and success and not far_conflicts),
                  horizon_policy='maximum forward coordinate of ALL finite source points; not shortened by rail/wall support',
                  elapsed_ms=1000*(time.perf_counter()-tick))
    return result


@frame_compute
def process_frame(xyz,thickness=.8,wall_mode='multilayer',budget_ms=100.,level=True,refine=True):
    """Complete independent-frame computation, excluding IO/JIT/rendering."""
    from .rail_pose_fast import estimate_rail_pose
    from .train_envelope import build_envelope
    from .rail_path_fast import estimate_path
    from .rail_bev import extract_bev
    if wall_mode not in ('multilayer','legacy'):raise ValueError('Unknown wall input mode')
    if budget_ms is not None and (not np.isfinite(budget_ms) or budget_ms<=0):raise ValueError('Budget must be positive or None')
    t=time.perf_counter();rail=estimate_rail_pose(xyz)
    envelope=build_envelope(rail['pose'])
    deadline=None if budget_ms is None else t+budget_ms/1000
    a=time.perf_counter();near=estimate_path(xyz,rail,envelope,None if deadline is None else deadline-.046);b=time.perf_counter()
    bev=(extract_bev(xyz,envelope,thickness) if wall_mode=='legacy' else {}) if envelope['status']=='ENVELOPE_ESTIMATED' else None
    c=time.perf_counter();evidence=None;observations=None
    limited=[]
    if near.get('rail_search_budget_limited'):limited.append('rail_continuation')
    run_evidence=deadline is None or c<deadline-.036
    if bev is not None and near['status']=='PATH_ESTIMATED' and wall_mode=='multilayer' and run_evidence:
        from .wall_layers import wall_evidence
        rows,ambiguity,evidence=wall_evidence(xyz,envelope,near)
        observations=(rows,ambiguity)
    elif wall_mode=='multilayer' and not run_evidence:
        observations=([],0);limited.append('wall_evidence')
    e=time.perf_counter()
    allow_fit=(deadline is None or e<deadline-.028) and run_evidence
    if not allow_fit:limited.append('wall_fit')
    if wall_mode=='legacy' and not allow_fit:observations=([],0)
    if bev is not None and wall_mode=='multilayer':
        if evidence is not None:extent=evidence['measured_forward_extent_m']
        else:
            from .wall_layers import _project
            _,_,extent=_project(np.asarray(xyz),np.asarray(envelope['origin']),np.asarray(envelope['basis']))
        bev['measured_forward_extent_m']=extent
    path=fit_tunnel_path(bev,near,observations,allow_fit) if bev is not None else near;d=time.perf_counter()
    if evidence is not None:
        path['wall_evidence_status']=evidence['status']
        path['wall_candidate_overflow']=evidence.get('candidate_overflow',0)
        path['wall_evidence_grid']={key:evidence[key] for key in ['grid_origin_d_m','grid_shape','height_policy'] if key in evidence}
        path['wall_tracking_diagnostics']={key:evidence[key] for key in ['tracking_ambiguity','observed_width_changes'] if key in evidence}
    runtime=dict(budget_ms=budget_ms,limited_stages=limited,degraded=bool(limited),
                 elapsed_ms=1000*(d-t),overrun=bool(deadline is not None and d>deadline),
                 semantics='cooperative compute budget; reserves time for 3D; no hard OS scheduling guarantee')
    path['runtime_budget']=runtime
    if limited:path['model_fit_consistent']=False
    leveled=level and envelope['status']=='ENVELOPE_ESTIMATED' and rail['pose'] is not None
    fast_level=bool(deadline is not None and time.perf_counter()>deadline-.020)
    if leveled:
        from .level_reference import level_path,level_extent
        from .wall_layers import _project
        if fast_level:
            limited.append('level_projection_precision');runtime['degraded']=True;evidence=None
        envelope=level_path(path,envelope,rail,evidence,fast=fast_level)
        # The horizon still comes from all finite source returns in the output basis.
        if fast_level:
            raw_local=None;extent=level_extent(np.asarray(xyz),np.asarray(envelope['origin']),np.asarray(envelope['basis'])[:,0])
        else:raw_local,_,extent=_project(np.asarray(xyz),np.asarray(envelope['origin']),np.asarray(envelope['basis']))
        if path.get('status')=='PATH_ESTIMATED':
            path['measured_forward_extent_m']=max(.01,extent)
            p=path['parameters'];path['offset_m']=p[0];path['heading_rad']=p[1];path['curvature_per_m']=p[2]
            path['radius_m']=None if p[2]==0 else 1/abs(p[2]);path['model']='straight' if p[2]==0 else 'arc'
            path['length_m']=length_to_plane(p,path['measured_forward_extent_m'])
        bev=None  # Display slice is built lazily in the final (level) reference.
    from .path_validation import validate_path
    local=raw_local if leveled else (evidence.get('local') if evidence is not None else None)
    # Legacy mode must also be checked against raw XYZ, not its height slice.
    if local is None and envelope['status']=='ENVELOPE_ESTIMATED' and not fast_level:
        from .wall_layers import _project
        local,_,_=_project(np.asarray(xyz),np.asarray(envelope['origin']),np.asarray(envelope['basis']))
    if refine and path.get('status')=='PATH_ESTIMATED' and not limited:
        from .corridor_arc import refine_arc,apply_refinement
        adjustment=refine_arc(path,local,deadline)
        apply_refinement(path,adjustment)
        if adjustment['status']=='BUDGET_LIMITED':
            limited.append('corridor_refinement');runtime['degraded']=True;path['model_fit_consistent']=False
    elif refine and limited:
        path['corridor_refinement']=dict(status='BUDGET_LIMITED')
    refined=time.perf_counter()
    if fast_level or (deadline is not None and refined>deadline-.008):
        limited.append('validation');runtime['degraded']=True;path['model_fit_consistent']=False
        path['validation']=dict(status='INSUFFICIENT_EVIDENCE',reasons=['compute_budget_limited'],elapsed_ms=0.)
    else:path['validation']=validate_path(path,local)
    path['candidate_rejected']=path['validation']['status']=='REJECTED'
    path['full_path_validated']=False  # Necessary BEV tests cannot certify a railway route.
    if path['validation']['status']=='REJECTED':path['model_fit_consistent']=False
    # Fit a spatial route at the FULL measured range. This required stage is
    # not skipped when the optional arc proposal exhausted its compute budget.
    from .cloud_route import fit_cloud_route
    if local is None and envelope['status']=='ENVELOPE_ESTIMATED':
        from .wall_layers import _project
        local,_,_=_project(np.asarray(xyz),np.asarray(envelope['origin']),np.asarray(envelope['basis']))
    checked=time.perf_counter()
    if local is not None:
        fit_cloud_route(path,local,envelope if envelope['status']=='ENVELOPE_ESTIMATED' else None)
    finish=time.perf_counter()
    runtime['elapsed_ms']=1000*(finish-t);runtime['overrun']=bool(deadline is not None and finish>deadline)
    return dict(rail=rail,envelope=envelope,bev=bev if wall_mode=='legacy' else None,path=path,wall_evidence=evidence,elapsed_ms=1000*(finish-t),
                runtime_budget=runtime,
                stages_ms=[1000*(a-t),1000*(b-a),1000*(c-b),1000*(e-c),1000*(d-e),1000*(refined-d),1000*(checked-refined),1000*(finish-checked)],
                stage_names=['rail_pose','rail_path','legacy_slice','wall_evidence','path_fit','level_and_corridor_refinement','arc_validation','spatial_route'])


def sample_full_path(path,step=.5):
    length=path['length_m']
    # Include exact far endpoint, not a rounded arange sample.
    n=max(1,int(np.ceil(length/step)))
    return sample_path(path,length=length,step=length/n if length>0 else step)


def warmup_tunnel():
    from .cloud_route import warmup_cloud_route
    warmup_cloud_route()
    from .level_reference import warmup_level
    warmup_level()
    from .corridor_arc import warmup_corridor
    warmup_corridor()
    from .path_validation import warmup_validation
    warmup_validation()
    from .wall_layers import warmup_layers
    warmup_layers()
    from .far_rails import warmup as warmup_far_rails
    warmup_far_rails()
    bits,sx,sd,_=_grid(np.array([[1.,2.,1.2]]),.8,1.6,4.,.25)
    counts=_count_bits(bits)
    _ridge_values(np.array([0],np.int64),counts,bits,sx,sd,0,1,False)
    _fit_terms(np.zeros(5),np.zeros((2,2)),1,np.zeros(1),np.ones(1,dtype=int),np.ones(1),.01,0.)
