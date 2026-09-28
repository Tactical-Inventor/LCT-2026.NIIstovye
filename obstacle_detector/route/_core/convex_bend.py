"""A single connected bend with variable curvature and an exact rail tangent."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from .train_envelope import WIDTH


def fit(bounds,stations,reference,anchors,values,origin,slope,extent,direction,obstacles):
    from .cloud_route import surface_at
    grid=np.linspace(0.,extent,max(4,int(np.ceil(extent/18)))+1)
    linear_knots=np.r_[grid[0],grid,grid[-1]]
    primitive=BSpline(linear_knots,np.eye(len(grid)),1).antiderivative(2)
    scale=primitive(extent)
    def design(s,nu=0):return primitive(s,nu)/scale
    A=direction*design(stations);R=direction*design(anchors)
    O=direction*design(obstacles[:,0])
    line=origin+slope*stations;rail_line=origin+slope*anchors
    magnitude=max(.01,direction*(reference[-1]-line[-1]))
    q=np.full(len(grid),magnitude/len(grid))
    weight=100.*(1+2*(stations/extent)**2)
    D=direction*design(stations,1);OD=direction*design(obstacles[:,0],1)
    signed=obstacles[:,2,None]*O
    from .bend_solver import solve_convex
    contiguous=lambda a:np.ascontiguousarray(a,float)
    A,R,signed=contiguous(A),contiguous(R),contiguous(signed).reshape(-1,len(q))
    for _ in range(4):
        dy=slope+D@q;norm=np.sqrt(1+dy*dy)
        lo,_=surface_at(bounds,stations+WIDTH/2*dy/norm)
        _,hi=surface_at(bounds,stations-WIDTH/2*dy/norm)
        lower=lo[:,0]+WIDTH/2/norm+.02-line
        upper=hi[:,0]-WIDTH/2/norm-.02-line
        od=slope+OD@q
        limit=obstacles[:,2]*(obstacles[:,1]-origin-slope*obstacles[:,0])-(WIDTH/2+.12)*np.sqrt(1+od*od)
        # Rails held within 8 cm, reference, outer bounds and observed
        # obstacles as one-sided penalties, curvature weights kept positive;
        # solved in one compiled call (bend_solver.solve_convex).
        q,success=solve_convex(q,1e-9,2*extent,100,A,R,signed,contiguous(line),contiguous(rail_line),
                               contiguous(values),contiguous(reference),contiguous(weight),
                               contiguous(lower),contiguous(upper),contiguous(limit))
    knots=primitive.t;basis=BSpline(knots,np.eye(len(knots)-4),3)
    coefficients=direction*(primitive.c[:len(knots)-4]/scale)@q
    greville=np.array([np.mean(knots[j+1:j+4]) for j in range(len(knots)-4)])
    coefficients+=origin+slope*greville
    return knots,basis,coefficients,dict(amplitude_m=float(direction*q.sum()),onset_m=0.,
        direction=direction,bend_count=1,optimizer_success=bool(success),
        family='one connected convex bend; strictly one curvature sign',curvature_weights=q.tolist())
