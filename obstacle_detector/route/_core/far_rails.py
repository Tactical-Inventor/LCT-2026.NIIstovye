"""Own rails beyond the near tracker, from single lidar ring crossings.

Past ~30 m a ring crosses each rail head with one or two returns, so the near
tracker's 3.5 cm profile bins (three returns each) never fill and it stops at
25-31 m on every recording. The rails are still there to 50-60 m: on one ring
they are two bumps of about equal height one gauge apart. The ring itself is
recovered exactly from the elevation angle of a return in the sensor frame.

Candidates are offsets from a reference line; they must agree on ONE smooth
offset curve that starts at the last near rail. A single wrong pair - a
platform edge, a neighbouring track - cannot steer the rest, which following
pairs one by one cannot promise. Only lateral centres are returned.
"""
import numpy as np
from numba import njit

REACH_M=2.6          # lateral search band around the reference
BUMP_M=.06           # rail head above the median of its lateral neighbourhood
CANT_M=.25           # height difference of the two heads (superelevation)
GAUGE_TOLERANCE_M=.1
MAX_OFFSET_M=.8      # furthest a far centre may lie from the reference
MAX_RANGE_M=80.      # the near tracker's own limit; rail heads are gone by then
MAX_GAP_M=15.        # a chain broken for longer is not the same rails any more
INLIER_M=.08


@njit(cache=True)
def _pairs(d,h,s,starts,ends,gauge):
    """Per ring (runs sorted by lateral offset): equal bumps one gauge apart."""
    out=np.empty((max(16,4*len(d)),4));n=0
    peaks=np.empty(len(d),np.int64);around=np.empty(len(d))
    for r in range(len(starts)):
        a=starts[r];b=ends[r]
        if b-a<8:continue
        np_=0;lo=a;hi=a
        for i in range(a,b):
            while d[i]-d[lo]>.35:lo+=1
            while hi<b and d[hi]-d[i]<=.35:hi+=1
            k=0;top=-np.inf
            for j in range(lo,hi):
                w=abs(d[j]-d[i])
                if w>.1:around[k]=h[j];k+=1
                elif w<.07 and h[j]>top:top=h[j]
            if k<2 or h[i]<top or h[i]-np.median(around[:k])<=BUMP_M:continue
            # Dual returns repeat a point; one peak per lateral position.
            if np_ and d[peaks[np_-1]]==d[i]:continue
            peaks[np_]=i;np_+=1
        for p in range(np_):
            for q in range(p+1,np_):
                i=peaks[p];j=peaks[q];g=d[j]-d[i]
                if g>gauge+GAUGE_TOLERANCE_M:break
                if abs(g-gauge)<GAUGE_TOLERANCE_M and abs(h[i]-h[j])<CANT_M and n<len(out):
                    out[n,0]=(s[i]+s[j])/2;out[n,1]=(d[i]+d[j])/2;out[n,2]=g;out[n,3]=(h[i]+h[j])/2;n+=1
    return out[:n]


def _consensus(c,start_s,start_offset):
    """Candidates agreeing on e(s)=e0+a*t+b*t^2 from the last near rail."""
    t=c[:,0]-start_s;e=c[:,1]-start_offset;n=len(c)
    i,j=np.triu_indices(n,1)
    ti,tj,ei,ej=t[i],t[j],e[i],e[j]
    det=ti*tj*tj-tj*ti*ti
    ok=abs(ti-tj)>=2
    a=np.where(ok,(ei*tj*tj-ej*ti*ti)/np.where(ok,det,1.),0.)
    b=np.where(ok,(ti*ej-tj*ei)/np.where(ok,det,1.),0.)
    # A line through the last near rail and one candidate is a hypothesis too.
    a=np.r_[e/t,a];b=np.r_[np.zeros(n),b];ok=np.r_[np.ones(n,bool),ok]
    ok&=(abs(b)<=4e-4)&(abs(a)<=.03)
    if not ok.any():return np.zeros(n,bool)
    a,b=a[ok],b[ok]
    residual=abs(a[:,None]*t+b[:,None]*t*t-e)
    inlier=residual<INLIER_M
    # Distinct ring crossings count, not repeated candidates on one ring.
    stations=np.round(c[:,0])
    score=np.array([len(np.unique(stations[row])) for row in inlier])-.1*np.where(inlier,residual,0.).sum(axis=1)/INLIER_M
    best=inlier[int(np.argmax(score))]
    return best if best.sum()>=2 else np.zeros(n,bool)


def far_rail_centers(local,origin,basis,reference,start,gauge,height_line,extent):
    """Route-local (s, y) centres of the own track beyond the near rails.

    local: (N,3) route-local returns; origin/basis map them back to the sensor
    frame, where the ring is the elevation angle. reference(s) is a lateral
    line to measure offsets from; start is the last near rail centre (s, y);
    height_line(s) is the near rail plane continued forward.
    """
    q=np.asarray(local);s0,y0=float(start[0]),float(start[1])
    m=(q[:,0]>s0)&(q[:,0]<min(extent,MAX_RANGE_M))
    if not m.any():return np.empty((0,2)),0
    q=q[m];offset=q[:,1]-reference(q[:,0]);height=q[:,2]-height_line(q[:,0])
    keep=(abs(offset)<REACH_M)&(abs(height)<1.5)
    q,offset,height=q[keep],offset[keep],height[keep]
    if len(q)<8:return np.empty((0,2)),0
    sensor=q@np.asarray(basis).T+np.asarray(origin)
    ring=np.round(np.degrees(np.arctan2(sensor[:,2],np.hypot(sensor[:,0],sensor[:,1])))/.02).astype(np.int64)
    order=np.lexsort((offset,ring))
    ring=ring[order]
    starts=np.flatnonzero(np.r_[True,ring[1:]!=ring[:-1]]);ends=np.r_[starts[1:],len(ring)]
    c=_pairs(offset[order],height[order],q[order,0],starts,ends,float(gauge))
    c=c[(abs(c[:,1])<MAX_OFFSET_M)&(c[:,0]>s0+.5)] if len(c) else c
    if not len(c):return np.empty((0,2)),0
    _,first=np.unique(np.round(c[:,:2],2),axis=0,return_index=True)
    c=c[np.sort(first)]
    chosen=c[_consensus(c,s0,y0-float(reference(np.array([s0]))[0]))]
    chosen=chosen[np.argsort(chosen[:,0])]
    gaps=np.flatnonzero(np.diff(np.r_[s0,chosen[:,0]])>MAX_GAP_M)
    if len(gaps):chosen=chosen[:gaps[0]]
    return np.column_stack((chosen[:,0],chosen[:,1]+reference(chosen[:,0]))),len(c)


def warmup():
    d=np.linspace(-1.,1.,12);_pairs(d,np.zeros(12),np.full(12,30.),np.array([0]),np.array([12]),1.52)
