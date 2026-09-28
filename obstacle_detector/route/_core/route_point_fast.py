"""Exact occupied-point grouping with bounded temporary storage."""
import numpy as np
from numba import njit


@njit(cache=True)
def grid_components(points,offset):
    """Same radius-3 graph as cKDTree, streamed through a uniform cell grid.

    Only connected-component membership matters, not the order of graph edges.
    Labels retain first-source order. Return -1 for a sparse pathological grid.
    """
    n=len(points)
    if n==0:return 0,np.empty(0,np.int32)
    cells=np.empty((n,2),np.int64)
    for i in range(n):
        for j in range(2):cells[i,j]=int(np.floor(points[i,j]/3.))
    lo=np.empty(2,np.int64);hi=np.empty(2,np.int64)
    for j in range(2):lo[j]=cells[0,j];hi[j]=cells[0,j]
    for i in range(n):
        for j in range(2):lo[j]=min(lo[j],cells[i,j]);hi[j]=max(hi[j],cells[i,j])
    nx=hi[0]-lo[0]+1;ny=hi[1]-lo[1]+1
    if nx>2000000 or ny>2000000 or nx*ny>2000000:
        return -1,np.empty(0,np.int32)
    head=np.full(nx*ny,-1,np.int64);link=np.full(n,-1,np.int64)
    parent=np.arange(n);rank=np.zeros(n,np.int64)
    for i in range(n):
        cx=cells[i,0]-lo[0];cy=cells[i,1]-lo[1]
        for x in range(max(0,cx-1),min(nx,cx+2)):
            for y in range(max(0,cy-1),min(ny,cy+2)):
                j=head[x*ny+y]
                while j>=0:
                    dx=points[i,0]-points[j,0];dy=points[i,1]-points[j,1]
                    if (dx*dx+dy*dy<=9. and
                        (np.sign(offset[i])==np.sign(offset[j]) or
                         (abs(offset[i])<.6 and abs(offset[j])<.6 and abs(dy)<.5))):
                        a=i;b=j
                        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
                        while parent[b]!=b:parent[b]=parent[parent[b]];b=parent[b]
                        if a!=b:
                            if rank[a]<rank[b]:parent[a]=b
                            else:
                                parent[b]=a
                                if rank[a]==rank[b]:rank[a]+=1
                    j=link[j]
        cell=cx*ny+cy;link[i]=head[cell];head[cell]=i
    labels=np.empty(n,np.int32);numbers=np.full(n,-1,np.int32);count=0
    for i in range(n):
        a=i
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        if numbers[a]<0:numbers[a]=count;count+=1
        labels[i]=numbers[a]
    return count,labels


@njit(cache=True)
def height_keys(points,offset):
    out=np.empty((len(points),2),np.int64)
    for i in range(len(points)):
        s=points[i,0]
        sx=s if s<60 else (60+(s-60)/2 if s<120 else 90+(s-120)/3)
        out[i,0]=int(np.floor(sx+offset))
        out[i,1]=int(np.floor(points[i,1]/.35+offset))
    return out


@njit(cache=True)
def voxel_keys(points):
    out=np.empty((len(points),3),np.int64)
    for i in range(len(points)):
        out[i,0]=int(np.floor(points[i,0]/.5))
        out[i,1]=int(np.floor(points[i,1]/.12))
        out[i,2]=int(np.floor(points[i,2]/.3))
    return out


@njit(cache=True)
def column_hypotheses(posts):
    """Interpolate triples only to associate posts; coefficients are not output.

    Near a membership threshold request the original SVD polyfit instead.
    Every retained row is still refitted by the original NumPy operation.
    """
    n=len(posts);count=n*(n-1)*(n-2)//6
    masks=np.full(count,-1,np.int64);triples=np.empty((count,3),np.int64);k=0
    conservative=np.max(np.abs(posts[:,1:]))>10000.
    for a in range(n-2):
        for b in range(a+1,n-1):
            for c in range(b+1,n):
                triples[k,0]=a;triples[k,1]=b;triples[k,2]=c
                x0=posts[a,1];x1=posts[b,1];x2=posts[c,1]
                if min(x1-x0,x2-x1)<5:k+=1;continue
                mask=0;uncertain=conservative
                for j in range(n):
                    x=posts[j,1]
                    prediction=(posts[a,2]*(x-x1)*(x-x2)/((x0-x1)*(x0-x2))+
                                posts[b,2]*(x-x0)*(x-x2)/((x1-x0)*(x1-x2))+
                                posts[c,2]*(x-x0)*(x-x1)/((x2-x0)*(x2-x1)))
                    error=abs(prediction-posts[j,2])
                    if abs(error-.5)<1e-7:uncertain=True
                    if error<.5:mask|=1<<j
                masks[k]=-2 if uncertain else mask;k+=1
    return masks,triples


def warmup():
    p=np.zeros((2,3));grid_components(p,np.zeros(2));height_keys(p,0.);voxel_keys(p)
    column_hypotheses(np.zeros((4,3)))
