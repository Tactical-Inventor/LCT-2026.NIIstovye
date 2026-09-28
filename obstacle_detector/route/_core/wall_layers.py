"""Bounded multilayer wall evidence and joint side tracking from source XYZ.

The grid changes support weights, never the raw point selection. Height >=.2 m
is an explicit evidence policy; all measured higher layers are considered.
"""
import numpy as np
from numba import njit, prange

STEP=4.
BIN=.25
MAX_CELLS=1500000


@njit(cache=True,parallel=True)
def _project(xyz, origin, basis):
    """Returns ahead of the plane s=0 in the frame, in their original order.

    Blocks of rows are counted in parallel and then written to their own
    offsets, so the rows, their order and the extent are those of one pass.
    """
    m=len(xyz);blocks=64;size=(m+blocks-1)//blocks
    counts=np.zeros(blocks,np.int64);extents=np.zeros(blocks)
    for b in prange(blocks):
        c=0;e=0.
        for i in range(b*size,min(m,(b+1)*size)):
            if not (np.isfinite(xyz[i,0]) and np.isfinite(xyz[i,1]) and np.isfinite(xyz[i,2])): continue
            if xyz[i,0]==0 and xyz[i,1]==0 and xyz[i,2]==0: continue
            x=xyz[i,0]-origin[0];y=xyz[i,1]-origin[1];z=xyz[i,2]-origin[2]
            s=x*basis[0,0]+y*basis[1,0]+z*basis[2,0]
            e=max(e,s)
            if s<0:continue
            c+=1
        counts[b]=c;extents[b]=e
    starts=np.zeros(blocks,np.int64);n=0
    for b in range(blocks):
        starts[b]=n;n+=counts[b]
    local=np.empty((n,3),np.float64)
    ids=np.empty(n,np.int64)
    for b in prange(blocks):
        k=starts[b]
        for i in range(b*size,min(m,(b+1)*size)):
            if not (np.isfinite(xyz[i,0]) and np.isfinite(xyz[i,1]) and np.isfinite(xyz[i,2])): continue
            if xyz[i,0]==0 and xyz[i,1]==0 and xyz[i,2]==0: continue
            x=xyz[i,0]-origin[0];y=xyz[i,1]-origin[1];z=xyz[i,2]-origin[2]
            s=x*basis[0,0]+y*basis[1,0]+z*basis[2,0]
            if s<0:continue
            local[k,0]=s
            local[k,1]=x*basis[0,1]+y*basis[1,1]+z*basis[2,1]
            local[k,2]=x*basis[0,2]+y*basis[1,2]+z*basis[2,2]
            ids[k]=i;k+=1
    extent=0.
    for b in range(blocks):extent=max(extent,extents[b])
    return local,ids,extent


@njit(cache=True)
def _layer(h):
    return 0 if h<.8 else int(h/.8)


@njit(cache=True)
def _pop(v):
    n=0
    while v:
        v=v&(v-1);n+=1
    return n


@njit(cache=True)
def _layers(local):
    dmin=np.inf;dmax=-np.inf;far=0.;top=.8
    for p in local:
        if p[2]<.2:continue
        dmin=min(dmin,p[1]);dmax=max(dmax,p[1]);far=max(far,p[0]);top=max(top,p[2])
    if not np.isfinite(dmin):
        return np.zeros((0,0,0),np.uint16),np.zeros((0,0,0,3)),np.zeros(len(local),np.bool_),0.,False
    dmin=np.floor(dmin/BIN)*BIN
    if far>STEP*MAX_CELLS or dmax-dmin>BIN*MAX_CELLS or top>.8*MAX_CELLS:
        return np.zeros((0,0,0),np.uint16),np.zeros((0,0,0,3)),np.zeros(len(local),np.bool_),dmin,True
    nx=int(far/STEP)+1;nd=int((dmax-dmin)/BIN)+1;nh=_layer(top)+1
    if float(nx)*nd*nh>MAX_CELLS:
        return np.zeros((0,0,0),np.uint16),np.zeros((0,0,0,3)),np.zeros(len(local),np.bool_),dmin,True
    bits=np.zeros((nx,nh,nd),np.uint16)
    sums=np.zeros((nx,nh,nd,3))
    representative=np.zeros(len(local),np.bool_)
    for a in range(len(local)):
        p=local[a]
        if p[2]<.2:continue
        i=int(p[0]/STEP);j=int((p[1]-dmin)/BIN);h=_layer(p[2])
        low=.2 if h==0 else h*.8
        high=.8 if h==0 else (h+1)*.8
        along=min(3,int((p[0]-i*STEP)/STEP*4))
        height=min(3,max(0,int((p[2]-low)/(high-low)*4)))
        flag=np.uint16(1<<(along*4+height))
        if bits[i,h,j]&flag:continue
        bits[i,h,j]|=flag;representative[a]=True
        for axis in range(3):sums[i,h,j,axis]+=p[axis]
    return bits,sums,representative,dmin,False


@njit(cache=True)
def _candidates(bits,sums):
    # Columns: s,d,h,unique cells,quality,first layer,last layer,grid column.
    # Keep at most 32 lateral candidates/window, ranked by vertical evidence.
    nx,nh,nd=bits.shape
    out=np.zeros((nx,32,8));counts=np.zeros(nx,np.int64)
    members=np.full(bits.shape,-1,np.int32);overflow=0
    starts=np.zeros(nx,np.int64);ends=np.zeros(nx,np.int64);nw=0;i=0
    while i<nx:
        width=1 if i*STEP<60 else (2 if i*STEP<120 else 3)
        end=min(nx,i+width)
        temp=np.zeros((max(1,nh*nd),8));nt=0
        temp_members=np.full((nh,nd),-1,np.int32)
        for h in range(nh):
            profile=np.zeros(nd)
            for j in range(nd):
                for a in range(i,end):profile[j]+=_pop(bits[a,h,j])
            smooth=np.zeros(nd)
            for j in range(nd):
                smooth[j]=.5*profile[j]
                if j>0:smooth[j]+=.25*profile[j-1]
                if j+1<nd:smooth[j]+=.25*profile[j+1]
            for j in range(nd):
                if smooth[j] < (2.5 if i*STEP<60 else .25):continue
                if j>0 and smooth[j]<=smooth[j-1]:continue
                if j+1<nd and smooth[j]<smooth[j+1]:continue
                mask=np.uint16(0);cells=0;value=np.zeros(3)
                for a in range(i,end):
                    for b in range(max(0,j-1),min(nd,j+2)):
                        mask|=bits[a,h,b];cells+=_pop(bits[a,h,b])
                        for axis in range(3):value[axis]+=sums[a,h,b,axis]
                levels=0;along=0
                for z in range(4):
                    if mask & np.uint16(0x1111<<z):levels+=1
                    if mask & np.uint16(15<<(4*z)):along+=1
                vertical_columns=0;max_vertical=0
                for z in range(4):
                    vertical=_pop(np.uint16((mask>>(4*z))&15))
                    if vertical>=2:vertical_columns+=1
                    max_vertical=max(max_vertical,vertical)
                strong=levels>=3 and along>=2 and cells>=6 and vertical_columns>=2 and max_vertical>=3
                if not strong and i*STEP<60:continue
                quality=1. if strong else (.4 if cells>=3 else .15)
                # Merge nearby ridges across heights; retain actual layer interval.
                d=value[1]/cells;merged=-1
                for b in range(nt):
                    if abs(temp[b,1]-d)<.3 and temp[b,6]<h:
                        merged=b;break
                if merged>=0:
                    old=temp[merged,3];total=old+cells
                    for axis in range(3):temp[merged,axis]=(temp[merged,axis]*old+value[axis])/total
                    temp[merged,3]=total;temp[merged,4]=max(temp[merged,4],quality)
                    if temp[merged,4]<.65:temp[merged,4]=.65
                    temp[merged,6]=h
                    temp_members[h,j]=merged
                else:
                    for axis in range(3):temp[nt,axis]=value[axis]/cells
                    temp[nt,3]=cells;temp[nt,4]=quality;temp[nt,5]=h;temp[nt,6]=h;temp[nt,7]=j;nt+=1
                    temp_members[h,j]=nt-1
        # Lateral order is deterministic; cap only the support search, with diagnostics.
        order=np.argsort(-temp[:nt,4]-.005*np.minimum(temp[:nt,3],40))[:32]
        count=len(order)
        overflow+=max(0,nt-count)
        reverse=np.full(nt,-1,np.int32)
        for b in range(count):reverse[order[b]]=b
        for h in range(nh):
            for j in range(nd):
                if temp_members[h,j]>=0:members[nw,h,j]=reverse[temp_members[h,j]]
        for b in range(count):out[nw,b]=temp[order[b]]
        counts[nw]=count;starts[nw]=i;ends[nw]=end;nw+=1;i=end
    return out[:nw],counts[:nw],starts[:nw],ends[:nw],members[:nw],overflow


@njit(cache=True)
def _track(candidates,counts,starts,ends,offset,yaw,k,gauge):
    """Joint bounded beam: side ordering, persistent seeds, local continuity.

    Local linear prediction tracks evidence only; the final output is one arc.
    """
    nw=len(counts);beam=8
    histories=np.full((beam,2,nw),-1,np.int64)
    scores=np.full(beam,-1e30);scores[0]=0.;nb=1
    for w in range(nw):
        proposals=np.full((beam,3),-1,np.int64);best=np.full(beam,-1e30)
        for b in range(nb):
            options=np.full((2,33),-1,np.int64)
            values=np.full((2,33),-1e30);sizes=np.ones(2,np.int64)
            previous=np.full(2,-1,np.int64);predicted=np.zeros(2)
            for side in range(2):
                sign=-1 if side==0 else 1
                recent=np.full(3,-1,np.int64);n=0
                for t in range(w-1,-1,-1):
                    if histories[b,side,t]>=0:
                        recent[n]=t;n+=1
                        if n==3:break
                # Missing evidence remains an option, never invent a return.
                values[side,0]=-.2*(ends[w]-starts[w]) if n else -.3
                if n:
                    t=recent[0];last=candidates[t,histories[b,side,t]];previous[side]=t
                    slope=np.tan(yaw+k*last[0])
                    if n>=2:
                        older=candidates[recent[min(n-1,2)],histories[b,side,recent[min(n-1,2)]]]
                        slope=(last[1]-older[1])/(last[0]-older[0])
                        slope=max(-.9,min(.9,slope))
                    gap=(starts[w]+ends[w])*STEP/2-last[0]
                    predicted[side]=last[1]+gap*slope
                    if gap>max(24.,3*(ends[w]-starts[w])*STEP):continue
                elif starts[w]*STEP>20:continue
                for a in range(counts[w]):
                    c=candidates[w,a]
                    # Seeds and their near continuation must intersect the known
                    # rail-relative side-height zone, not an overhead-only ridge.
                    if c[0]<40 and c[5]>2:continue
                    if n:
                        ds=c[0]-last[0]
                        if ds<=.5:continue
                        deviation=abs(c[1]-last[1]-slope*ds)
                        gate=.4+.015*ds+.0015*ds*ds
                        if deviation>gate:continue
                        score=(.6+c[4])*(ends[w]-starts[w])-2*(deviation/gate)**2
                    else:
                        distance=sign*(c[1]-offset-np.tan(yaw)*c[0])
                        if distance<=gauge/2+.1 or distance>=10 or c[4]<.65 or c[5]>2:continue
                        persistent=0
                        for future in range(w+1,min(nw,w+4)):
                            matched=False
                            for candidate in range(counts[future]):
                                other=candidates[future,candidate]
                                if other[4]<.65 or other[5]>2:continue
                                ds=other[0]-c[0]
                                if abs(other[1]-c[1]-np.tan(yaw+k*c[0])*ds)<.4+.01*ds:
                                    matched=True;break
                            if matched:persistent+=1
                        if persistent<2:continue
                        score=1.6+.5*persistent-.25*distance
                    pos=sizes[side];options[side,pos]=a;values[side,pos]=score;sizes[side]+=1
            for l in range(sizes[0]):
                li=options[0,l]
                for r in range(sizes[1]):
                    ri=options[1,r]
                    if li>=0 and ri>=0:
                        if candidates[w,ri,1]-candidates[w,li,1]<.5:continue
                    # A returning side cannot cross the other recently observed side.
                    if li>=0 and ri<0 and previous[1]>=0:
                        t=previous[1];other=candidates[t,histories[b,1,t]]
                        if other[4]>=.65 and (starts[w]+ends[w])*STEP/2-other[0]<36 and candidates[w,li,1]>=predicted[1]-.5:continue
                    if ri>=0 and li<0 and previous[0]>=0:
                        t=previous[0];other=candidates[t,histories[b,0,t]]
                        if other[4]>=.65 and (starts[w]+ends[w])*STEP/2-other[0]<36 and candidates[w,ri,1]<=predicted[0]+.5:continue
                    score=scores[b]+values[0,l]+values[1,r]
                    worst=np.argmin(best)
                    if score>best[worst]:best[worst]=score;proposals[worst]=np.array([b,li,ri])
        order=np.argsort(-best);new=np.full_like(histories,-1);new_scores=np.full(beam,-1e30);nb=0
        for a in order:
            if proposals[a,0]<0:continue
            b,li,ri=proposals[a]
            new[nb]=histories[b];new[nb,0,w]=li;new[nb,1,w]=ri;new_scores[nb]=best[a];nb+=1
        histories=new;scores=new_scores
    return histories[:nb],scores[:nb]


@njit(cache=True)
def _attach(local,representatives,members,tracks,candidates,starts,ends,dmin):
    nw=len(starts);nh=members.shape[1];nd=members.shape[2]
    window=np.zeros(ends[-1],np.int64)
    for w in range(nw):
        for i in range(starts[w],ends[w]):window[i]=w
    owners=np.full(len(local),-1,np.int32)
    sums=np.zeros((2*nw,3));counts=np.zeros(2*nw,np.int64)
    lookup=np.full(members.shape,-1,np.int32)
    for w in range(nw):
        for h in range(nh):
            for j in range(nd):
                candidate=members[w,h,j]
                if candidate<0:continue
                for side in range(2):
                    if tracks[side,w]!=candidate:continue
                    owner=side*nw+w
                    for col in range(max(0,j-1),min(nd,j+2)):
                        old=lookup[w,h,col]
                        lookup[w,h,col]=owner if old==-1 or old==owner else -2
    for a in range(len(local)):
        p=local[a]
        if p[2]<.2:continue
        i=int(p[0]/STEP)
        if i>=len(window):continue
        w=window[i];h=_layer(p[2]);j=int((p[1]-dmin)/BIN)
        if h>=nh or j<0 or j>=nd:continue
        best=lookup[w,h,j]
        if best==-2:
            best=-1;error=np.inf
            for side in range(2):
                candidate=tracks[side,w]
                if candidate<0:continue
                for col in range(max(0,j-1),min(nd,j+2)):
                    if members[w,h,col]!=candidate:continue
                    distance=abs(p[1]-candidates[w,candidate,1])
                    if distance<error:best=side*nw+w;error=distance
        if best<0:continue
        owners[a]=best
        if representatives[a]:
            counts[best]+=1
            for axis in range(3):sums[best,axis]+=p[axis]
    return owners,sums,counts


def wall_evidence(xyz,envelope,rail_path):
    local,ids,extent=_project(np.asarray(xyz),np.asarray(envelope['origin']),np.asarray(envelope['basis']))
    bits,sums,representatives,dmin,skipped=_layers(local)
    if skipped or not bits.size:
        return [],0,dict(status='GRID_LIMIT' if skipped else 'NO_EVIDENCE',source_indices=ids,
                         local=local,representative_mask=representatives,measured_forward_extent_m=extent)
    candidates,counts,starts,ends,members,overflow=_candidates(bits,sums)
    hypotheses,scores=_track(candidates,counts,starts,ends,rail_path['offset_m'],rail_path['heading_rad'],
                         rail_path['curvature_per_m'],rail_path['gauge_m'])
    tracks=hypotheses[0]
    ambiguity=0
    for alternative,score in zip(hypotheses[1:],scores[1:]):
        if scores[0]-score>3:continue
        different=0
        for side in range(2):
            for w in range(len(starts)):
                a,b=tracks[side,w],alternative[side,w]
                if a>=0 and b>=0 and abs(candidates[w,a,1]-candidates[w,b,1])>.4:different+=1
        if different>=3:ambiguity+=1
    pair_widths=[];near_widths=[]
    for w in range(len(starts)):
        a,b=tracks[:,w]
        if a<0 or b<0:continue
        width=float(candidates[w,b,1]-candidates[w,a,1])
        pair_widths.append((float(starts[w]*STEP),width))
        if starts[w]*STEP<40:near_widths.append(width)
    width_changes=[]
    if len(near_widths)>=3:
        reference=float(np.median(near_widths))
        width_changes=[dict(s=s,width_m=width,near_width_m=reference) for s,width in pair_widths
                       if s>=40 and (width<.65*reference or width>1.5*reference)]
        if len(width_changes)>=2:ambiguity+=1
    owners,totals,cell_counts=_attach(local,representatives,members,tracks,candidates,starts,ends,dmin)
    selected=np.flatnonzero((owners>=0)&representatives)
    selected=selected[np.argsort(owners[selected],kind='stable')]
    splits=np.r_[0,np.cumsum(cell_counts)]
    rows=[]
    for side in range(2):
        for w,index in enumerate(tracks[side]):
            if index<0:continue
            c=candidates[w,index]
            owner=side*len(starts)+w
            if not cell_counts[owner]:continue
            value=totals[owner]/cell_counts[owner]
            source=ids[selected[splits[owner]:splits[owner+1]]]
            regions=np.argwhere(members[w]==index)
            rows.append(dict(s=float(value[0]),d=float(value[1]),h=float(value[2]),cells=int(cell_counts[owner]),quality=float(c[4]),
                             side=-1 if side==0 else 1,station=int(starts[w]),window_m=float((ends[w]-starts[w])*STEP),
                             height_layer_range=[int(c[5]),int(c[6])],grid_column=int(c[7]),
                             source_indices=source.tolist(),source_region_cells=regions.tolist(),evidence_owner=int(owner)))
    return rows,ambiguity,dict(status='MULTILAYER_EVIDENCE',source_indices=ids,local=local,
                       representative_mask=representatives,measured_forward_extent_m=extent,
                       wall_membership=owners,candidate_overflow=int(overflow),
                       tracking_ambiguity=int(ambiguity),observed_width_changes=width_changes,
                       grid_origin_d_m=float(dmin),grid_shape=list(bits.shape),
                       candidates=candidates,candidate_counts=counts,window_starts=starts,window_ends=ends,
                       beam_scores=scores.tolist(),height_policy='evidence h>=0.2; all higher measured 0.8 m layers; first layer 0.2..0.8')


def warmup_layers():
    for dtype in (np.float32,np.float64):
        _project(np.array([[1,2,1]],dtype=dtype),np.zeros(3),np.eye(3))
    bits,sums,_,_,_=_layers(np.array([[1.,2.,1.]]))
    candidates,counts,starts,ends,members,_=_candidates(bits,sums)
    hypotheses,_=_track(candidates,counts,starts,ends,0.,0.,0.,1.6)
    _attach(np.array([[1.,2.,1.]]),np.ones(1,np.bool_),members,hypotheses[0],candidates,starts,ends,2.)
