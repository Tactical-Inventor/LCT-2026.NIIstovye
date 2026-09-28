"""Height-supported occupied surfaces, independent of outer cloud bounds."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from numba import njit, prange
from .train_envelope import WIDTH, HEIGHT
from .spline_fast import interval, basis


@njit(cache=True)
def _forward_indices(q,extent):
    ids=np.empty(len(q),np.int64);n=0
    for i in range(len(q)):
        if (np.isfinite(q[i,0]) and np.isfinite(q[i,1]) and np.isfinite(q[i,2])
                and q[i,0]>=0 and q[i,0]<=extent):
            ids[n]=i;n+=1
    return ids[:n]


@njit(cache=True)
def _closest_per_group(inverse,side,p,n):
    chosen=np.full(n,-1,np.int64);best=np.full(n,np.inf)
    for i in range(len(p)):
        k=inverse[i];value=side[i]*p[i,1]
        if chosen[k]<0 or value<best[k]:chosen[k]=i;best[k]=value
    return chosen


@njit(cache=True)
def _limits(key):
    minimum=np.empty(key.shape[1],np.int64);maximum=np.empty(key.shape[1],np.int64)
    for j in range(key.shape[1]):minimum[j]=key[0,j];maximum[j]=key[0,j]
    for i in range(1,len(key)):
        for j in range(key.shape[1]):
            if key[i,j]<minimum[j]:minimum[j]=key[i,j]
            if key[i,j]>maximum[j]:maximum[j]=key[i,j]
    return minimum,maximum-minimum+1


@njit(cache=True)
def _height_supported(inverse,relative,n):
    lo=np.full(n,np.inf);hi=np.full(n,-np.inf)
    for i in range(len(relative)):
        k=inverse[i];v=relative[i]
        if v<lo[k]:lo[k]=v
        if v>hi[k]:hi[k]=v
    keep=np.empty(len(relative),np.bool_)
    for i in range(len(relative)):keep[i]=hi[inverse[i]]-lo[inverse[i]]>=.4
    return keep


@njit(cache=True)
def _dense_groups(key, minimum, widths, size):
    first=np.full(size,-1,np.int64)
    codes=np.empty(len(key),np.int64)
    for i in range(len(key)):
        code=0
        for j in range(key.shape[1]):
            code=code*widths[j]+(key[i,j]-minimum[j])
        codes[i]=code
        if first[code]<0:first[code]=i
    count=0
    for i in range(size):
        if first[i]>=0:count+=1
    sources=np.empty(count,np.int64);count=0
    for i in range(size):
        if first[i]>=0:
            sources[count]=first[i];first[i]=count;count+=1
    groups=np.empty(len(key),np.int64)
    for i in range(len(key)):groups[i]=first[codes[i]]
    return sources,groups


@njit(cache=True)
def _components(points,offset,pairs):
    """Same undirected edges and first-vertex labels, without a sparse matrix."""
    parent=np.arange(len(points));rank=np.zeros(len(points),np.int64)
    for i in range(len(pairs)):
        a,b=pairs[i]
        if not (np.sign(offset[a])==np.sign(offset[b]) or
                (abs(offset[a])<.6 and abs(offset[b])<.6 and abs(points[a,1]-points[b,1])<.5)):
            continue
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        while parent[b]!=b:
            parent[b]=parent[parent[b]];b=parent[b]
        if a==b:continue
        if rank[a]<rank[b]:parent[a]=b
        else:
            parent[b]=a
            if rank[a]==rank[b]:rank[a]+=1
    labels=np.empty(len(points),np.int32);numbers=np.full(len(points),-1,np.int32);count=0
    for i in range(len(points)):
        a=i
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        if numbers[a]<0:numbers[a]=count;count+=1
        labels[i]=numbers[a]
    return count,labels


def _unique_rows(key, inverse=False):
    """Integer voxel groups in unique order, with stable first source indices."""
    if not len(key):
        empty=np.empty(0,np.int64)
        return (empty,empty.copy()) if inverse else empty
    minimum,widths=_limits(key)
    size=1
    for width in widths:size*=int(width)
    # Bounded temporary storage; sparse or exceptionally wide clouds retain
    # the stable sort fallback. Inputs are integer-valued voxel coordinates.
    if 0<size<=2_000_000:
        first,groups=_dense_groups(key.astype(np.int64),minimum,widths,size)
        return (first,groups) if inverse else first
    order=np.lexsort(key.T[::-1])
    ordered=key[order]
    start=np.r_[True,np.any(ordered[1:]!=ordered[:-1],axis=1)]
    first=order[start]
    if not inverse:return first
    groups=np.empty(len(key),np.intp)
    groups[order]=np.cumsum(start)-1
    return first,groups


RAISED_BIN_M = 1.0     # station bins the profiles' bounds are taken over
RAISED_SLACK_M = 1e-6  # the bounds' own rounding, far below any threshold here


def _profile_bounds(lateral,height,extent):
    """Exact min/max of both profiles over every RAISED_BIN_M of station."""
    from scipy.interpolate import PPoly
    from .._membership import _extrema
    edges=np.arange(0.,extent+RAISED_BIN_M,RAISED_BIN_M)
    if len(edges)<2:edges=np.array([0.,RAISED_BIN_M])
    ylo,yhi=_extrema(PPoly.from_spline(lateral),edges)
    zlo,zhi=_extrema(PPoly.from_spline(height),edges)
    return ylo,yhi,zlo,zhi


@njit(cache=True,parallel=True)
def _raised(q,ids,t,lateral,height,k,ceiling,ylo,yhi,zlo,zhi):
    """Returns 0.3 m over the profile and under the ceiling, within 12 m across.

    One pass over the frame instead of indexing it, evaluating both profiles
    and masking in turn; the profiles use SciPy's arithmetic (spline_fast).
    Returns are tested in parallel blocks and kept in their original order.
    Most returns - the track bed, the vault - are settled by the profiles'
    bounds over their station bin alone and need no spline evaluation; the
    bounds only skip returns the exact test rejects as well.
    """
    m=len(ids);blocks=64;size=(m+blocks-1)//blocks;bins=len(zlo)
    relative=np.empty(m);keep=np.zeros(m,np.bool_)
    for b in prange(blocks):
        h=np.empty(2*k+2);hh=np.empty(2*k+2);ell=k
        for j in range(b*size,min(m,(b+1)*size)):
            i=ids[j];s=q[i,0]
            c=min(bins-1,max(0,int(s/RAISED_BIN_M)))
            if (q[i,2]-zlo[c]<=.3-RAISED_SLACK_M or q[i,2]-zhi[c]>=ceiling+RAISED_SLACK_M
                    or q[i,1]<=ylo[c]-12.-RAISED_SLACK_M or q[i,1]>=yhi[c]+12.+RAISED_SLACK_M):
                continue
            ell=interval(t,k,s,ell)
            basis(t,s,k,ell,0,h,hh)
            across=0.;bottom=0.
            for a in range(k+1):
                across+=lateral[ell+a-k]*h[a]
                bottom+=height[ell+a-k]*h[a]
            r=q[i,2]-bottom
            relative[j]=r
            keep[j]=r>.3 and r<ceiling and abs(q[i,1]-across)<12.
    n=0
    for j in range(m):n+=keep[j]
    out=np.empty(n,np.int64);kept=np.empty(n);n=0
    for j in range(m):
        if keep[j]:
            out[n]=ids[j];kept[n]=relative[j];n+=1
    return out,kept


@njit(cache=True,parallel=True)
def _column_keys(q,ids,offset,k0,k1):
    for j in prange(len(ids)):
        i=ids[j];s=q[i,0]
        sx=s if s<60 else (60+(s-60)/2 if s<120 else 90+(s-120)/3)
        k0[j]=int(np.floor(sx+offset))
        k1[j]=int(np.floor(q[i,1]/.35+offset))


@njit(cache=True,parallel=True)
def _mark_columns(k0,k1,lo0,lo1,w1,low,high,keep):
    for j in prange(len(k0)):
        code=(k0[j]-lo0)*w1+(k1[j]-lo1)
        if high[code]-low[code]>=.4:keep[j]=True


@njit(cache=True)
def _supported_voxels(q,ids,relative):
    """Returns in height-supported columns, one per voxel, in voxel order.

    The same grouping as height_keys + _unique_rows + _height_supported for
    both staggered grids, then voxel_keys + _unique_rows, read straight from
    the frame instead of copying the rows at every step. Returns None when a
    column grid would need more than 2,000,000 cells (the array path handles
    it); voxels go through a hash table of any extent.
    """
    n=len(ids);keep=np.zeros(n,np.bool_)
    k0=np.empty(n,np.int64);k1=np.empty(n,np.int64)
    for offset in (0.,.5):
        _column_keys(q,ids,offset,k0,k1)
        lo0=k0.min();lo1=k1.min();w0=k0.max()-lo0+1;w1=k1.max()-lo1+1
        if w0*w1>2000000:return None
        low=np.full(w0*w1,np.inf);high=np.full(w0*w1,-np.inf)
        for j in range(n):
            code=(k0[j]-lo0)*w1+(k1[j]-lo1)
            if relative[j]<low[code]:low[code]=relative[j]
            if relative[j]>high[code]:high[code]=relative[j]
        _mark_columns(k0,k1,lo0,lo1,w1,low,high,keep)
    m=0
    for j in range(n):m+=keep[j]
    if m==0:return np.empty(0,np.int64)
    kept=np.empty(m,np.int64);m=0
    for j in range(n):
        if keep[j]:kept[m]=ids[j];m+=1
    return _first_per_voxel(q,kept)


@njit(cache=True,parallel=True)
def _voxel_keys(q,kept,ka,kb,kc):
    for j in prange(len(kept)):
        i=kept[j]
        ka[j]=int(np.floor(q[i,0]/.5));kb[j]=int(np.floor(q[i,1]/.12));kc[j]=int(np.floor(q[i,2]/.3))


@njit(cache=True,parallel=True)
def _voxel_codes(ka,kb,kc,lo0,lo1,lo2,w1,w2,codes):
    for j in prange(len(ka)):
        codes[j]=((ka[j]-lo0)*w1+(kb[j]-lo1))*w2+(kc[j]-lo2)


@njit(cache=True,parallel=True)
def _block_firsts(codes,blocks,size,unique,first,counts):
    """Per block of consecutive returns, its voxels and their first return."""
    m=len(codes)
    for b in prange(blocks):
        start=b*size;stop=min(m,start+size)
        capacity=1
        while capacity<2*max(1,stop-start):capacity*=2
        table=np.full(capacity,-1,np.int64);u=0
        for j in range(start,stop):
            code=codes[j]
            slot=np.int64((np.uint64(code)*np.uint64(11400714819323198485))&np.uint64(capacity-1))
            while table[slot]>=0 and unique[start+table[slot]]!=code:slot=(slot+1)&(capacity-1)
            if table[slot]<0:
                table[slot]=u;unique[start+u]=code;first[start+u]=j;u+=1
        counts[b]=u


@njit(cache=True)
def _first_per_voxel(q,kept):
    """The first of the kept returns in every occupied voxel, in voxel order.

    Blocks of returns find their own voxels in parallel; merged in block
    order, the earlier block's return wins, which is the first overall.
    """
    m=len(kept)
    ka=np.empty(m,np.int64);kb=np.empty(m,np.int64);kc=np.empty(m,np.int64)
    _voxel_keys(q,kept,ka,kb,kc)
    lo0=ka.min();lo1=kb.min();lo2=kc.min()
    w1=kb.max()-lo1+1;w2=kc.max()-lo2+1
    codes=np.empty(m,np.int64)
    _voxel_codes(ka,kb,kc,lo0,lo1,lo2,w1,w2,codes)
    blocks=32;size=(m+blocks-1)//blocks
    unique=np.empty(m,np.int64);first=np.empty(m,np.int64);counts=np.zeros(blocks,np.int64)
    _block_firsts(codes,blocks,size,unique,first,counts)
    total=0
    for b in range(blocks):total+=counts[b]
    capacity=1
    while capacity<2*total:capacity*=2
    table=np.full(capacity,-1,np.int64)
    merged=np.empty(total,np.int64);chosen=np.empty(total,np.int64);u=0
    for b in range(blocks):
        start=b*size
        for k in range(counts[b]):
            code=unique[start+k]
            slot=np.int64((np.uint64(code)*np.uint64(11400714819323198485))&np.uint64(capacity-1))
            while table[slot]>=0 and merged[table[slot]]!=code:slot=(slot+1)&(capacity-1)
            if table[slot]<0:
                table[slot]=u;merged[u]=code;chosen[u]=first[start+k];u+=1
    order=np.argsort(merged[:u])
    out=np.empty(u,np.int64)
    for c in range(u):out[c]=kept[chosen[order[c]]]
    return out


def _supported_voxels_arrays(p,ids,relative):
    """Array path of _supported_voxels, for any grid size and dtype."""
    from .route_point_fast import height_keys,voxel_keys
    # Continuous index with widths 1, 2, 3 m (no overlapping range boundaries).
    keep=np.zeros(len(p),bool)
    for offset in [0.,.5]:
        key=height_keys(p,offset)
        _,inverse=_unique_rows(key,inverse=True)
        n=int(inverse.max())+1
        # A .4 m span necessarily occupies at least two .25 m height bins.
        # Thus the old distinct-layer count adds no condition to this mask.
        keep|=_height_supported(inverse,relative,n)
    p=p[keep];ids=ids[keep]
    if not len(p):return p,ids
    # One source point per small occupied voxel keeps dense near returns from
    # overwhelming sparse distant evidence. No synthetic centroids are used.
    first=_unique_rows(voxel_keys(p))
    return p[first],ids[first]


def _shared_knots(lateral,height):
    return (isinstance(lateral,BSpline) and isinstance(height,BSpline) and lateral.k==height.k
            and lateral.extrapolate==height.extrapolate and np.array_equal(lateral.t,height.t)
            and lateral.c.ndim==height.c.ndim==1 and len(lateral.c)==len(height.c))


def extract(local, lateral, height, extent,prepared=None):
    """Retain actual returns in vertical cells; duplicate returns add no evidence.

    Two staggered grids prevent a narrow post at a cell border from being lost.
    Larger distant cells gather sparse layers, but constraints keep each return's
    own forward coordinate rather than inflating a sloping wall to a rectangle.
    """
    q=np.asarray(local)
    ids=_forward_indices(q,extent) if prepared is None else prepared
    compiled=q.dtype==np.float64 and q.flags.c_contiguous
    if compiled and _shared_knots(lateral,height):
        ids,relative=_raised(q,ids,np.ascontiguousarray(lateral.t,float),
                             np.ascontiguousarray(lateral.c,float),np.ascontiguousarray(height.c,float),
                             int(lateral.k),HEIGHT-.1,*_profile_bounds(lateral,height,extent))
    else:
        p=q[ids]
        across=lateral(p[:,0]);bottom=height(p[:,0])
        relative=p[:,2]-bottom
        mask=(relative>.3)&(relative<HEIGHT-.1)&(abs(p[:,1]-across)<12.)
        ids=ids[mask];relative=relative[mask]
    empty=dict(points=[],local_indices=[],components=[],side=[],column=[])
    if not len(ids):return empty
    from .route_point_fast import grid_components
    chosen=_supported_voxels(q,ids,relative) if compiled else None
    if chosen is not None:
        ids=chosen;p=q[ids]
    else:
        p,ids=_supported_voxels_arrays(q[ids],ids,relative)
    if not len(p):return empty
    # A nearby platform or beam must not join the two corridor sides into
    # one component and assign both walls the same avoidance side.
    offset=p[:,1]-lateral(p[:,0])
    n,labels=grid_components(p,offset)
    if n<0:
        pairs=cKDTree(p[:,:2]).query_pairs(3.,output_type='ndarray')
        n,labels=_components(p,offset,pairs)
    side=np.zeros(len(p));column=np.zeros(len(p),bool);movable=np.zeros(len(p),bool);ranges=[]
    # Every component's returns in one stable pass instead of a mask per
    # component; the same returns, so the same medians and extents.
    order=np.argsort(labels,kind='stable')
    bounds=np.searchsorted(labels[order],np.arange(n+1))
    for k in range(n):
        members=order[bounds[k]:bounds[k+1]];part=p[members]
        side[members]=1 if np.median(offset[members])>=0 else -1
        is_column=bool(np.ptp(part[:,0])<2.5 and np.ptp(part[:,1])<1.)
        column[members]=is_column
        movable[members]=is_column and np.any(abs(offset[members])<WIDTH/2+.25)
        if not is_column and np.ptp(part[:,0])>3.:
            ranges.append([float(part[:,0].min()),float(part[:,0].max()),int(side[members[0]])])
    # The lists are the evidence as reported; 'arrays' holds the same values
    # for the checks that follow, so they need not convert the lists back.
    return dict(points=p.tolist(),local_indices=ids.tolist(),components=labels.tolist(),
                side=side.tolist(),column=column.tolist(),movable=movable.tolist(),side_ranges=ranges,
                semantics='observed height-supported occupied returns; unobserved volume is unknown',
                arrays=dict(points=p,components=labels,side=side,column=column,movable=movable))


def evidence_array(evidence,name):
    """One field of extract()'s evidence as an array, without converting its list again."""
    arrays=evidence.get('arrays')
    return arrays[name] if arrays is not None and name in arrays else np.asarray(evidence[name])


@njit(cache=True,parallel=True)
def _clearance(p,t,lateral,height,k,extent,half,ceiling):
    """Per return, the same five Newton steps as the array version below."""
    n=len(p);gap=np.empty(n);station=np.empty(n);active=np.empty(n,np.bool_)
    blocks=64;size=(n+blocks-1)//blocks
    for b in prange(blocks):
        h=np.empty(2*k+2);hh=np.empty(2*k+2)
        for i in range(b*size,min(n,(b+1)*size)):
            s=min(max(p[i,0],0.),extent)
            for _ in range(5):
                ell=interval(t,k,s,k)
                basis(t,s,k,ell,0,h,hh);y=0.
                for a in range(k+1):y+=lateral[ell+a-k]*h[a]
                basis(t,s,k,ell,1,h,hh);dy=0.
                for a in range(k+1):dy+=lateral[ell+a-k]*h[a]
                s=min(max(s-((s-p[i,0])+(y-p[i,1])*dy)/(1+dy*dy),0.),extent)
            ell=interval(t,k,s,k)
            basis(t,s,k,ell,1,h,hh);dy=0.
            for a in range(k+1):dy+=lateral[ell+a-k]*h[a]
            basis(t,s,k,ell,0,h,hh);y=0.;z=0.
            for a in range(k+1):
                y+=lateral[ell+a-k]*h[a]
                z+=height[ell+a-k]*h[a]
            normal=((p[i,1]-y)-(p[i,0]-s)*dy)/np.sqrt(1+dy*dy)
            relative=p[i,2]-z
            gap[i]=abs(normal)-half;station[i]=s
            active[i]=relative>.15 and relative<ceiling
    return gap,station,active


def clearance(lateral,height,points,extent):
    p=np.asarray(points)
    if not len(p):return np.empty(0),np.empty(0),np.empty(0,dtype=bool)
    if _shared_knots(lateral,height) and p.ndim==2 and p.shape[1]==3:
        return _clearance(np.ascontiguousarray(p,float),np.ascontiguousarray(lateral.t,float),
                          np.ascontiguousarray(lateral.c,float),np.ascontiguousarray(height.c,float),
                          int(lateral.k),float(extent),WIDTH/2,HEIGHT-.05)
    s=np.clip(p[:,0],0.,extent)
    for _ in range(5):
        y=lateral(s);dy=lateral(s,1)
        s=np.clip(s-((s-p[:,0])+(y-p[:,1])*dy)/(1+dy*dy),0.,extent)
    dy=lateral(s,1)
    normal=((p[:,1]-lateral(s))-(p[:,0]-s)*dy)/np.sqrt(1+dy*dy)
    relative=p[:,2]-height(s)
    active=(relative>.15)&(relative<HEIGHT-.05)
    return abs(normal)-WIDTH/2,s,active


def constraints(evidence,column_side=0):
    p=evidence_array(evidence,'points')
    if not len(p):return np.empty((0,3))
    side=evidence_array(evidence,'side').copy()
    if column_side:side[evidence_array(evidence,'movable' if 'movable' in evidence else 'column')]=column_side
    for row in evidence.get('column_rows',[]):
        if row.get('enforced',True):
            side[np.isin(evidence_array(evidence,'components'),row['components'])]=row['side']
    # Keep the closest face on each side in each half-metre station, using
    # actual source coordinates. These are sufficient one-sided constraints.
    keys=np.column_stack((np.floor(p[:,0]/.5),side))
    _,inverse=_unique_rows(keys,inverse=True)
    chosen=_closest_per_group(inverse,side,p,int(inverse.max())+1)
    return np.column_stack((p[chosen,:2],side[chosen]))


def anchor_column_rows(evidence,lateral,cache=None):
    """Keep a well-supported divider on the side measured beside near rails.

    A quadratic is only an association model, never synthetic occupied returns.
    Require four posts over 50 m and a near post outside the vehicle. Ambiguous
    associations are left unchanged. Isolated posts remain independently tested.
    """
    p=evidence_array(evidence,'points');labels=evidence_array(evidence,'components')
    if not len(p):return []
    column=evidence_array(evidence,'column');posts=[]
    for k in np.unique(labels[column]):
        part=p[labels==k]
        posts.append((int(k),*np.median(part[:,:2],axis=0)))
    posts=np.asarray(sorted(posts,key=lambda a:a[1]))
    if len(posts)<4 or len(posts)>30:return []
    cache_key=(posts.shape,posts.tobytes())
    if cache is not None and cache_key in cache:
        associations=cache[cache_key]
    else:
        from .route_point_fast import column_hypotheses
        hypotheses,triples=column_hypotheses(posts)
        associations=[];seen=set()
        for bits,triplet in zip(hypotheses,triples):
            if bits==-1:continue
            if bits==-2:
                sample=posts[triplet]
                coef=np.polyfit(sample[:,1],sample[:,2],2)
                mask=abs(np.polyval(coef,posts[:,1])-posts[:,2])<.5
                bits=sum(1<<int(j) for j in np.flatnonzero(mask))
            if bits in seen:continue
            seen.add(bits)
            mask=np.array([bool(bits&(1<<j)) for j in range(len(posts))])
            row=posts[mask]
            if len(row)<4 or np.ptp(row[:,1])<50 or row[0,1]>25:continue
            if np.max(np.diff(row[:,1]))>45:continue
            coef=np.polyfit(row[:,1],row[:,2],2)
            if abs(2*coef[0])>.01 or abs(coef[1])>.08:continue
            if np.max(abs(np.polyval(coef,row[:,1])-row[:,2]))>.4:continue
            associations.append((row,coef))
        if cache is not None:cache[cache_key]=associations
    candidates={}
    for row,coef in associations:
        # Geometry can be reused, but the current route must determine the near
        # side afresh. Never reuse a final decision after a trajectory change.
        offset=row[0,2]-float(lateral(row[0,1]))
        if not WIDTH/2+.25<abs(offset)<4:continue
        key=tuple(int(k) for k in row[:,0])
        candidates[key]=dict(components=list(key),side=1 if offset>0 else -1,
            observed_range_m=[float(row[0,1]),float(row[-1,1])],
            association_coefficients=coef.tolist())
    # Precompute membership once, preserving the original insertion order.
    sets={key:frozenset(key) for key in candidates}
    maximal=[(key,row) for key,row in candidates.items()
             if not any(sets[key]<other for other in sets.values())]
    rows=[row for key,row in maximal if not any(sets[key]&sets[other]
           for other,_ in maximal if other!=key)]
    if rows:evidence['column_rows']=rows
    return rows


def audit(route,extent,points=None):
    """Observed returns the route's cross-section intersects; points may pass
    the evidence already as an array (it is read from the route otherwise)."""
    evidence=route.get('obstacles',{})
    if points is None:points=evidence.get('points',[])
    lateral=BSpline(route['knots'],route['lateral_coefficients'],3)
    height=BSpline(route['knots'],route['height_coefficients'],3)
    gap,s,active=clearance(lateral,height,points,extent)
    bad=active&(gap<-.03)
    return dict(observed_return_count=len(points),intersecting_return_count=int(bad.sum()),
                max_penetration_m=float(max(0.,-gap[active].min())) if active.any() else 0.,
                conflicting_stations_m=s[bad].tolist(),
                status='OBSERVED_COLLISION' if bad.any() else 'NO_OBSERVED_COLLISION',
                semantics='finite observed surfaces only; absence of returns is not free-space evidence')

