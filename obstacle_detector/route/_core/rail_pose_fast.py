"""Sparse single-frame rail reference for positioning a 2.3 x 2.4 m envelope.

Only five near cross-sections are fitted. Returned support indices are original
returns, NOT a dense segmentation and NOT the future envelope membership mask.
Numba kernels use no fastmath and no frame-to-frame state.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.signal import find_peaks
from ._reference import frame_basis, pairs_from_peaks
from ._reference import unit
from ._reference import default_config

VERSION = 'sparse_rail_pose_v2'


@njit(cache=True)
def robust_fit(a,b,scale):
    """Small convex soft-L1 regression: Newton with analytic derivatives.

    Same loss as the reference scipy fit; standardized columns and backtracking
    avoid finite-difference Jacobians and generic trust-region solver overhead.
    Last design column must be all ones. No outlier threshold is changed.
    """
    n,k=a.shape
    means=np.zeros(k); scales=np.ones(k)
    x=a.copy(); offset=np.mean(b); y=b-offset
    for j in range(k-1):
        means[j]=np.mean(a[:,j]); scales[j]=max(np.std(a[:,j]),1e-12)
        x[:,j]=(a[:,j]-means[j])/scales[j]
    h=x.T@x
    for j in range(k): h[j,j]+=1e-12
    c=np.linalg.solve(h,x.T@y)
    for _ in range(40):
        r=x@c-y
        v=np.sqrt(1+(r/scale)**2)
        gradient=x.T@(r/v)
        if np.max(np.abs(gradient))<1e-10:
            break
        hessian=np.zeros((k,k))
        for i in range(n):
            weight=1/(v[i]*v[i]*v[i])
            for u in range(k):
                for w in range(k):
                    hessian[u,w]+=x[i,u]*x[i,w]*weight
        for j in range(k): hessian[j,j]+=1e-12
        step=np.linalg.solve(hessian,gradient)
        loss=np.sum(r*r/(v+1))
        rate=1.
        for _ in range(25):
            trial=c-rate*step
            residual=x@trial-y
            trial_loss=np.sum(residual*residual/(np.sqrt(1+(residual/scale)**2)+1))
            if trial_loss<=loss:
                break
            rate*=.5
        c=trial
        if np.max(np.abs(rate*step))<1e-10:
            break
    answer=c/scales
    answer[-1]+=offset-np.sum(answer[:-1]*means[:-1])
    return answer


@njit(cache=True)
def _gather(xyz, basis):
    ids=np.empty(len(xyz),np.int64)
    coords=np.empty((len(xyz),3),np.float64)
    n=0
    for i in range(len(xyz)):
        x,y,z=float(xyz[i,0]),float(xyz[i,1]),float(xyz[i,2])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        s=x*basis[0,0]+y*basis[1,0]+z*basis[2,0]
        d=x*basis[0,1]+y*basis[1,1]+z*basis[2,1]
        h=x*basis[0,2]+y*basis[1,2]+z*basis[2,2]
        if 1.<=s<=20. and abs(d)<=4. and -5.<=h<=2.:
            ids[n]=i
            coords[n,0]=s; coords[n,1]=d; coords[n,2]=h
            n+=1
    return ids[:n],coords[:n]


@njit(cache=True)
def _cell_medians(q, size):
    """Median of every occupied cell, cells in key order.

    A counting sort on the cell key: the cells come out in the same order as
    by sorting the keys, and a median does not depend on the order within.
    """
    a=np.floor(q[:,0]/size).astype(np.int64)
    b=np.floor(q[:,1]/size).astype(np.int64)
    stride=b.max()-b.min()+1
    keys=(a-a.min())*stride+b-b.min()
    cells=keys.max()+1
    sizes=np.zeros(cells,np.int64)
    for k in keys: sizes[k]+=1
    starts=np.empty(cells,np.int64);total=0
    for k in range(cells):
        starts[k]=total;total+=sizes[k]
    fill=starts.copy();order=np.empty(len(q),np.int64)
    for i in range(len(q)):
        order[fill[keys[i]]]=i;fill[keys[i]]+=1
    out=np.empty_like(q)
    count=0;values=np.empty(len(q))
    for k in range(cells):
        n=sizes[k]
        if n==0:continue
        for axis in range(3):
            for j in range(n):values[j]=q[order[starts[k]+j],axis]
            out[count,axis]=np.median(values[:n])
        count+=1
    return out[:count]


@njit(cache=True)
def _ransac(samples, picks, tolerance, tilt):
    best=np.zeros(len(samples),np.bool_); count=0
    for pick in picks:
        a,b,c=samples[pick[0]],samples[pick[1]],samples[pick[2]]
        dx1,dy1,dz1=b-a
        dx2,dy2,dz2=c-a
        det=dx1*dy2-dx2*dy1
        if abs(det)<1e-6:
            continue
        slope_s=(dz1*dy2-dz2*dy1)/det
        slope_d=(dx1*dz2-dx2*dz1)/det
        if slope_s*slope_s+slope_d*slope_d>tilt*tilt:
            continue
        intercept=a[2]-slope_s*a[0]-slope_d*a[1]
        mask=np.abs(samples[:,0]*slope_s+samples[:,1]*slope_d+intercept-samples[:,2])<tolerance
        n=mask.sum()
        if n>count:
            best=mask; count=n
    return best


@njit(cache=True)
def _levelled_box(xyz, source, origin, basis):
    """Rows of source inside the near-rail box of the levelled frame.

    A compiled pass picks the candidates with a nanometre of slack, and only
    they go through the same BLAS product as (xyz[source] - origin) @ basis,
    so the kept coordinates are those of the array version to the bit.
    """
    n=len(source);cand=np.empty(n,np.int64);m=0
    slack=1e-9
    for k in range(n):
        i=source[k]
        a=float(xyz[i,0])-origin[0];b=float(xyz[i,1])-origin[1];c=float(xyz[i,2])-origin[2]
        s=a*basis[0,0]+b*basis[1,0]+c*basis[2,0]
        d=a*basis[0,1]+b*basis[1,1]+c*basis[2,1]
        h=a*basis[0,2]+b*basis[1,2]+c*basis[2,2]
        if s>=2-slack and s<=19+slack and abs(d)<3.5+slack and abs(h)<1.+slack:
            cand[m]=k;m+=1
    y=np.empty((m,3))
    for r in range(m):
        i=source[cand[r]]
        for j in range(3):y[r,j]=float(xyz[i,j])-origin[j]
    q=np.dot(y,basis)
    keep=np.empty(m,np.bool_);kept=0
    for r in range(m):
        keep[r]=q[r,0]>=2 and q[r,0]<=19 and abs(q[r,1])<3.5 and abs(q[r,2])<1.
        kept+=keep[r]
    out=np.empty((kept,3));ids=np.empty(kept,np.int64);kept=0
    for r in range(m):
        if keep[r]:
            out[kept]=q[r];ids[kept]=source[cand[r]];kept+=1
    return out,ids


def _floor(mount, cfg):
    attempts=[]
    for width in [cfg['calibration_half_width'],1.4,2.6]:
        keep=((mount[:,0]>=3)&(mount[:,0]<=18)&(abs(mount[:,1])<=width)&
              (mount[:,2]>=cfg['floor_height_limits'][0])&(mount[:,2]<=cfg['floor_height_limits'][1]))
        points=mount[keep]
        try:
            if len(points)<cfg['floor_min_cells']:
                raise ValueError('insufficient_low_points')
            samples=_cell_medians(points,cfg['floor_cell'])
            if len(samples)<cfg['floor_min_cells']:
                raise ValueError('insufficient_floor_cells')
            rng=np.random.default_rng(cfg['seed'])
            picks=np.array([rng.choice(len(samples),3,replace=False) for _ in range(cfg['ransac_iterations'])])
            limit=np.tan(np.radians(cfg['floor_max_tilt_deg']))
            best=_ransac(samples,picks,cfg['floor_tolerance'],limit)
            if best.sum()<cfg['floor_min_cells'] or best.mean()<cfg['floor_min_fraction']:
                raise ValueError('no_dominant_low_plane')
            design=np.column_stack((samples[:,:2],np.ones(len(samples))))
            coef=robust_fit(design[best],samples[best,2],cfg['floor_tolerance'])
            best=abs(design@coef-samples[:,2])<cfg['floor_tolerance']
            if np.ptp(samples[best,0])<cfg['floor_min_span'] or np.linalg.norm(coef[:2])>limit:
                raise ValueError('floor_span_or_tilt')
            attempts.append(dict(width=width,status='OK'))
            return coef,dict(attempts=attempts,fraction=float(best.mean()),cells=len(samples))
        except ValueError as error:
            attempts.append(dict(width=width,status=str(error)))
    raise ValueError('floor_unconfirmed: '+str(attempts))


@njit(cache=True)
def _voxel_first(q, size):
    cells=np.floor(q/size).astype(np.int64)
    lo=np.empty(3,np.int64); hi=np.empty(3,np.int64)
    for axis in range(3):
        lo[axis]=cells[:,axis].min(); hi[axis]=cells[:,axis].max()
    shape=hi-lo+1
    keys=((cells[:,0]-lo[0])*shape[1]+cells[:,1]-lo[1])*shape[2]+cells[:,2]-lo[2]
    # Same first return and sorted voxel order, sorting only unique keys.
    capacity=1
    while capacity<2*len(q):capacity*=2
    table=np.full(capacity,-1,np.int64)
    unique=np.empty(len(q),np.int64);first=np.empty(len(q),np.int64);n=0
    for i in range(len(keys)):
        key=keys[i]
        slot=np.int64((np.uint64(key)*np.uint64(11400714819323198485))&np.uint64(capacity-1))
        while table[slot]>=0 and unique[table[slot]]!=key:slot=(slot+1)&(capacity-1)
        if table[slot]<0:
            table[slot]=n;unique[n]=key;first[n]=i;n+=1
    return first[:n][np.argsort(unique[:n])]


@njit(cache=True)
def _profile(q, s, center, heading, edges, quantile):
    bins=len(edges)-1
    count=np.zeros(bins,np.int64)
    heights=np.full(bins,np.nan)
    base=np.full(bins,np.nan)
    cosine=np.cos(heading); sine=np.sin(heading)
    membership=np.full(len(q),-1,np.int64)
    for i in range(len(q)):
        point=q[i]
        along=(point[0]-s)*cosine+(point[1]-center)*sine
        across=-(point[0]-s)*sine+(point[1]-center)*cosine
        if abs(along)<1.5 and abs(across)<1.7 and -.55<point[2]<.65:
            j=np.searchsorted(edges,across,side='right')-1
            if 0<=j<bins:membership[i]=j;count[j]+=1
    offsets=np.zeros(bins+1,np.int64)
    for j in range(bins):offsets[j+1]=offsets[j]+count[j]
    values=np.empty(offsets[-1],np.float64);filled=np.zeros(bins,np.int64)
    for i in range(len(q)):
        j=membership[i]
        if j>=0:values[offsets[j]+filled[j]]=q[i,2];filled[j]+=1
    for j in range(bins):
        if count[j]>=3:
            row=np.sort(values[offsets[j]:offsets[j+1]])
            for k,fraction in enumerate((quantile,.25)):
                pos=(len(row)-1)*fraction
                low=int(pos); high=min(low+1,len(row)-1)
                value=row[low]+(pos-low)*(row[high]-row[low])
                if k==0: heights[j]=value
                else: base[j]=value
    return heights,base


def _peaks(q,s,center,heading,cfg):
    edges=np.arange(-1.7,1.7+cfg['profile_bin'],cfg['profile_bin'])
    positions=(edges[:-1]+edges[1:])/2
    heights,base=_profile(q,s,center,heading,edges,cfg['profile_quantile']/100.)
    support=np.isfinite(base)
    if support.sum()<cfg['local_floor_min_bins']:
        return [],[],np.zeros(2)
    design=np.column_stack((positions[support],np.ones(support.sum())))
    floor=robust_fit(design,base[support],cfg['floor_tolerance'])
    inliers=abs(base[support]-design@floor)<cfg['floor_tolerance']
    if inliers.sum()>=cfg['local_floor_min_bins']:
        floor=robust_fit(design[inliers],base[support][inliers],cfg['floor_tolerance'])
    if abs(floor[0])>cfg['local_floor_max_slope'] or abs(floor[1])>cfg['local_floor_max_shift']:
        return [],[],floor
    heights-=positions*floor[0]+floor[1]
    valid=np.flatnonzero(np.isfinite(heights))
    if len(valid)<3:
        return [],[],floor
    filled=np.interp(positions,positions[valid],heights[valid])
    proposed,_=find_peaks(filled,height=cfg['rail_height_limits'][0],prominence=cfg['rail_prominence'],
                          distance=max(1,int(cfg['rail_band']/cfg['profile_bin'])))
    peaks=[]
    for j in proposed:
        distance=positions[valid]-positions[j]
        left=heights[valid[(distance<0)&(distance>=-cfg['peak_flank_width'])]]
        right=heights[valid[(distance>0)&(distance<=cfg['peak_flank_width'])]]
        if (np.min(abs(distance))<=cfg['peak_nearest_observation'] and len(left) and len(right)
            and filled[j]-left.min()>=cfg['rail_prominence'] and filled[j]-right.min()>=cfg['rail_prominence']):
            peaks.append(j)
    return positions[peaks],filled[peaks],floor


# How much less centred a pair may be before it is a different track, not the
# same one measured differently. Track spacing is metres; peak quantisation is
# centimetres, so this only ever admits a neighbouring peak of the own rails.
SEED_CENTRE_MARGIN = 0.20


def _agree_on_gauge(options, tolerance, margin=SEED_CENTRE_MARGIN):
    """Pair per station agreeing on one gauge, without leaving the own track.

    Gauge agreement alone is not a criterion: the adjacent track has the same
    gauge, so it would happily walk onto it. Candidates are therefore limited
    to pairs nearly as centred as the most centred one at their own station.
    """
    allowed = [(s, [p for p in pairs if abs(p[0]) <= abs(pairs[0][0]) + margin])
               for s, pairs in options]
    best = None
    for _, pairs in allowed:
        for candidate in pairs:
            chosen = []
            for s, near in allowed:
                fit = [p for p in near if abs(p[1] - candidate[1]) <= tolerance]
                if fit:
                    chosen.append([s, *min(fit, key=lambda p: abs(p[0]))])
            key = (len(chosen), -sum(abs(row[1]) for row in chosen))
            if best is None or key > best[0]:
                best = (key, chosen)
    return [] if best is None else best[1]


def _drop_outliers(design, centres, stations):
    """Refit the near direction on the largest subset of stations that agrees.

    Stations on the own track agree on a straight line to centimetres, so a
    station pairing the neighbouring track stands out by half a metre. It
    cannot be found by discarding the largest residual, though: a gross outlier
    drags the fitted line towards itself and makes a good station look like the
    offender instead. With at most five stations every subset can simply be
    tried, so the one that actually agrees is found rather than guessed.

    Three stations spanning the usual 6 m must survive. Nothing is lost by
    dropping one: the second pass revisits every station in the fitted
    direction and can still take its own rails there.
    """
    best = None
    for mask in range(1, 1 << len(centres)):
        keep = np.array([(mask >> i) & 1 for i in range(len(centres))], bool)
        if keep.sum() < 3 or np.ptp(stations[keep]) < 5.9:
            continue
        direction = robust_fit(design[keep], centres[keep], .045)
        residual = float(abs(design[keep] @ direction - centres[keep]).max())
        if residual > .3 or abs(np.degrees(np.arctan(direction[0]))) > 18:
            continue
        key = (int(keep.sum()), -residual)
        if best is None or key > best[0]:
            best = (key, direction, keep)
    return (None, None) if best is None else (best[1], best[2])


def _drop_section(heads):
    """The refined sections without the one that spoils the fit, or None.

    Every section is tried in turn, farthest first, and the first fit that
    passes the residual limits with at least three sections over the usual
    6 m is kept. A diverging track leaves the own one with distance, so when
    dropping either of two sections passes, the farther one is the suspect:
    on frame 768 the section at 16.5 m sat 0.23 m off the line of the four
    nearer ones, and dropping 13.5 m instead also passed, bent towards it.
    """
    pairs=heads.reshape(-1,2,3)
    if len(pairs)<4:return None
    for drop in np.argsort(-pairs[:,:,0].mean(axis=1)):
        keep=[k for k in range(len(pairs)) if k!=drop]
        sub=pairs[keep].reshape(-1,3);centers=(sub[::2]+sub[1::2])/2
        if np.ptp(centers[:,0])<5.9:continue
        center_design=np.column_stack((centers[:,0],np.ones(len(centers))))
        center_coef=robust_fit(center_design,centers[:,1],.03)
        height_design=np.column_stack((sub[:,0],sub[:,1],np.ones(len(sub))))
        height_coef=robust_fit(height_design,sub[:,2],.02)
        residual_d=float(np.max(abs(center_design@center_coef-centers[:,1])))
        residual_h=float(np.max(abs(height_design@height_coef-sub[:,2])))
        if residual_d>.12 or residual_h>.08:continue
        return keep,center_coef,height_coef,residual_d,residual_h
    return None


def estimate_rail_pose(xyz,cfg=None):
    """Return 6..10 source supports, fitted rail-head pose and local envelope.

    The short straight envelope only shows mounting/height reference over the
    observed near span. It is not the final constant-curvature route estimate.
    """
    tick=time.perf_counter()
    xyz=np.asarray(xyz)
    if xyz.ndim!=2 or xyz.shape[1]!=3:
        raise ValueError('Expected XYZ shape (N,3)')
    cfg=default_config() if cfg is None else dict(cfg)
    report=dict(version=VERSION,status='UNCONFIRMED',confidence=0.,stages_ms={},
                scope='Near rail-head reference only; no far route or clearance test',
                dimensions_m=dict(width=2.3,height=2.4,extra_margin=0.))
    result=dict(indices=np.empty(0,np.int64),points=np.empty((0,3)),labels=np.empty(0,np.uint8),
                head_centers=np.empty((0,3)),pose=None,envelope_corners=np.empty((0,3)),report=report)
    def finish(reason):
        report.update(reason=reason,support_points=len(result['indices']),elapsed_ms=1000*(time.perf_counter()-tick))
        return result
    mount_basis=frame_basis(cfg['forward_hint'],cfg['up_hint'])
    source,mount=_gather(xyz,mount_basis)
    report['stages_ms']['gather']=1000*(time.perf_counter()-tick)
    if len(mount)<cfg['floor_min_cells']:
        return finish('insufficient_near_points')
    stage=time.perf_counter()
    try:
        coef,floor_info=_floor(mount,cfg)
    except ValueError as error:
        return finish(str(error))
    normal=unit(mount_basis@[-coef[0],-coef[1],1.])
    origin=mount_basis@[0.,0.,coef[2]]
    basis=frame_basis(mount_basis[:,0],normal)
    if xyz.dtype in (np.float32,np.float64):
        q,source=_levelled_box(xyz,source,np.ascontiguousarray(origin,float),np.ascontiguousarray(basis,float))
    else:
        q=(np.asarray(xyz[source],float)-origin)@basis
        keep=(q[:,0]>=2)&(q[:,0]<=19)&(abs(q[:,1])<3.5)&(abs(q[:,2])<1.)
        q=q[keep]; source=source[keep]
    if not len(q):
        return finish('no_near_rail_points')
    ids=_voxel_first(q,cfg['rail_voxel'])
    fit=q[ids]; fit_source=source[ids]
    report['stages_ms']['floor_and_voxels']=1000*(time.perf_counter()-stage)
    stage=time.perf_counter()
    options=[]; candidates=[]
    # Every station reads only returns within 1.5 m along a heading of at most
    # 18 degrees and 1.8 m across: under 2.5 m of station away. A 3.3 m window
    # keeps the same returns in the same order, and scans a sixth of them.
    windows={s:np.abs(fit[:,0]-s)<3.3 for s in (4.5,7.5,10.5,13.5,16.5)}
    for s in (4.5,7.5,10.5,13.5,16.5):
        peaks,heights,floor=_peaks(fit[windows[s]],s,0.,0.,cfg)
        pairs=[p for p in pairs_from_peaks(peaks,None,0,cfg)
               if abs(p[0])<=cfg['calibration_center_limit'] and p[2]<0<p[3]]
        pairs.sort(key=lambda p:abs(p[0]))
        if pairs:
            options.append((s,pairs))
            candidates.append(dict(s=s,peaks=np.asarray(peaks).tolist(),pairs=[list(p) for p in pairs]))
    report['seed_candidates']=candidates
    if len(options)<3:
        return finish('insufficient_near_pairs')
    observations=[[s,*pairs[0]] for s,pairs in options]
    obs=np.array(observations)
    gauge=np.median(obs[:,2])
    consistent=abs(obs[:,2]-gauge)<=cfg['gauge_tolerance']
    rejected=obs[~consistent].tolist()
    obs=obs[consistent]
    if len(obs)<3 or np.ptp(obs[:,0])<5.9:
        # The most centred pair at a station can disagree on the gauge because
        # of a neighbouring peak, while a pair a few centimetres further out
        # agrees with every other station. Retry once with that swap allowed.
        # It stays a fallback: a frame the rule above accepts never reaches
        # here, so the accepted geometry is untouched wherever it worked.
        retry=_agree_on_gauge(options,cfg['gauge_tolerance'])
        if len(retry)<3 or np.ptp([row[0] for row in retry])<5.9:
            report['rejected_pairs']=rejected
            return finish('inconsistent_pairs_or_short_span')
        obs=np.array(retry);gauge=np.median(obs[:,2])
        rejected=[[s,*pairs[0]] for s,pairs in options if s not in {row[0] for row in retry}]
        report['seed_gauge_retry']=dict(stations=len(options),supported=len(retry),
                                        gauge_m=float(gauge),centre_margin_m=SEED_CENTRE_MARGIN)
    report['rejected_pairs']=rejected
    design=np.column_stack((obs[:,0],np.ones(len(obs))))
    direction=robust_fit(design,obs[:,1],.045)
    if max(abs(design@direction-obs[:,1]))>.3 or abs(np.degrees(np.arctan(direction[0])))>18:
        # One seed station can land on the neighbouring track: past a curve the
        # own rails leave the centred seed window and only the neighbour is
        # left to pair there. Drop such a station instead of refusing the whole
        # frame. Fallback only: a frame the check above accepts never gets here.
        direction,keep=_drop_outliers(design,obs[:,1],obs[:,0])
        if direction is None:
            return finish('inconsistent_near_direction')
        report['direction_outliers']=obs[~keep].tolist()
        obs=obs[keep];gauge=float(np.median(obs[:,2]))
    # Recenter and orient each profile once; supports are measured, not extrapolated.
    heads=[]; indices=[]; labels=[]; sections=[]
    heading=float(np.arctan(direction[0]))
    tangent=np.array([np.cos(heading),np.sin(heading)])
    lateral=np.array([-tangent[1],tangent[0]])
    # A turning rail may leave the initial sensor-centred profile. Revisit all
    # five stations in the fitted direction, not only successful seed stations.
    for s in (4.5,7.5,10.5,13.5,16.5):
        center=float(direction@[s,1])
        near=fit[windows[s]];near_source=fit_source[windows[s]]
        peaks,heights,floor=_peaks(near,s,center,heading,cfg)
        pairs=[p for p in pairs_from_peaks(peaks,gauge,cfg['gauge_tolerance'],cfg) if abs(p[0])<=.3]
        pairs.sort(key=lambda p:abs(p[0])+abs(p[1]-gauge))
        if not pairs:
            continue
        pair=pairs[0]
        along=(near[:,:2]-[s,center])@tangent
        across=(near[:,:2]-[s,center])@lateral
        section_heads=[]; section_indices=[]
        for peak in pair[2:]:
            height=heights[np.argmin(abs(peaks-peak))]+peak*floor[0]+floor[1]
            ids=np.flatnonzero((abs(along)<1.5)&(abs(across-peak)<.12)&
                               (near[:,2]>height-.055)&(near[:,2]<height+.035))
            if len(ids)<6 or np.ptp(along[ids])<1.:
                break
            a=np.column_stack((along[ids],np.ones(len(ids))))
            d_coef=robust_fit(a,across[ids],.025)
            h_coef=robust_fit(a,near[ids,2],.018)
            head=np.r_[np.array([s,center])+lateral*d_coef[1],h_coef[1]]
            nearest=ids[np.argmin(np.sum((near[ids]-head)**2,axis=1))]
            section_heads.append(head)
            section_indices.append(int(near_source[nearest]))
        if len(section_heads)==2:
            heads.extend(section_heads); indices.extend(section_indices); labels.extend([1,2])
            sections.append(dict(s=float(s),alternatives=len(pairs),fit_heads_local=np.array(section_heads).tolist()))
    report['stages_ms']['rail_profiles']=1000*(time.perf_counter()-stage)
    if len(heads)<6:
        return finish('insufficient_refined_heads')
    heads=np.array(heads); centers=(heads[::2]+heads[1::2])/2
    span=float(np.ptp(centers[:,0]))
    if span<5.9:
        return finish('insufficient_refined_span')
    center_design=np.column_stack((centers[:,0],np.ones(len(centers))))
    center_coef=robust_fit(center_design,centers[:,1],.03)
    height_design=np.column_stack((heads[:,0],heads[:,1],np.ones(len(heads))))
    height_coef=robust_fit(height_design,heads[:,2],.02)
    residual_d=float(np.max(abs(center_design@center_coef-centers[:,1])))
    residual_h=float(np.max(abs(height_design@height_coef-heads[:,2])))
    if residual_d>.12 or residual_h>.08:
        # At a switch one station can pair a blade or the diverging rail: on
        # squareT_platform_squareT_switch frames 753 and 768 a single section
        # lay 0.16-0.25 m off the line of the others, or one head 0.085 m
        # above their plane, and the whole frame went without a route. Drop
        # the one section that disagrees if the rest agree over the same span.
        # Fallback only: a frame the check above accepts never gets here.
        refit=_drop_section(heads)
        if refit is None:
            return finish('rail_pose_residual_too_large')
        keep,center_coef,height_coef,residual_d,residual_h=refit
        report['section_outliers']=[sections[k]['s'] for k in range(len(sections)) if k not in keep]
        heads=heads.reshape(-1,2,3)[keep].reshape(-1,3);centers=(heads[::2]+heads[1::2])/2
        indices=[indices[2*k+j] for k in keep for j in (0,1)];labels=[labels[2*k+j] for k in keep for j in (0,1)]
        sections=[sections[k] for k in keep];span=float(np.ptp(centers[:,0]))
    # Frame fitted to the HEADS, not to a constant lidar Z or the floor plane.
    up=unit(basis@[-height_coef[0],-height_coef[1],1.])
    forward=basis@np.array([1.,center_coef[0],height_coef[0]+height_coef[1]*center_coef[0]])
    rail_basis=frame_basis(forward,up)
    station=float(centers[:,0].min())
    d=float(center_coef@[station,1])
    h=float(height_coef@[station,d,1])
    rail_origin=origin+basis@[station,d,h]
    length=float(span*np.sqrt(1+center_coef[0]**2+(height_coef[0]+height_coef[1]*center_coef[0])**2))
    corners=np.array([[s,d,h] for s in [0.,length] for d in [-1.15,1.15] for h in [0.,2.4]])
    corners=corners@rail_basis.T+rail_origin
    ambiguity=sum(section['alternatives']>1 for section in sections)
    confidence=.9*min(1.,span/12)*min(1.,len(sections)/5)*np.exp(-residual_d/.1-residual_h/.08)
    if ambiguity or report['rejected_pairs']:
        confidence*=.8
    if len(floor_info['attempts'])>1:
        confidence*=.8
    report.update(status='RAIL_REFERENCE_ESTIMATED',confidence=float(confidence),floor=floor_info,
                  observed_span_m=span,sections=sections,ambiguous_sections=ambiguity,
                  residual_lateral_m=residual_d,residual_height_m=residual_h,
                  peak_separation_m=float(gauge),near_points_considered=len(fit),
                  confidence_semantics='geometric heuristic, not a probability')
    result.update(indices=np.array(indices,np.int64),points=xyz[indices].copy(),labels=np.array(labels,np.uint8),
                  head_centers=heads@basis.T+origin,
                  pose=dict(origin=rail_origin.tolist(),basis=rail_basis.tolist(),length_m=length),
                  envelope_corners=corners)
    return finish('observed_near_head_pairs')


def warmup():
    """Compile/load kernels before timing or opening interactive frame processing."""
    rng=np.random.default_rng(91)
    q=rng.normal(size=(300,3)); q[:,0]=rng.uniform(3,18,300)
    _gather(q,np.eye(3)); _cell_medians(q,.25)
    _gather(q.astype(np.float32),np.eye(3))
    ids=np.arange(len(q))
    _levelled_box(q,ids,np.zeros(3),np.eye(3));_levelled_box(q.astype(np.float32),ids,np.zeros(3),np.eye(3))
    # Frames arrive as views over a message buffer: strided rows.
    strided=np.zeros((len(q),4),np.float32);strided[:,:3]=q
    _levelled_box(strided[:,:3],ids,np.zeros(3),np.eye(3))
    _ransac(q,np.array([[0,1,2]],np.int64),.045,.27)
    _voxel_first(q,.025)
    _profile(q,4.5,0.,0.,np.arange(-1.7,1.735,.035),.8)
    robust_fit(np.column_stack((q[:,:2],np.ones(len(q)))),q[:,2],.045)
