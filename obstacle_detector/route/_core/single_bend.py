"""One C2 straight-to-turn transition; opposite bends are impossible by construction."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from .train_envelope import WIDTH

# The six starts of the compiled bend fit are independent and release the GIL.
_STARTS = ThreadPoolExecutor(max_workers=3, thread_name_prefix='bend_start')


def horizontal(s,origin,slope,amplitude,onset,extent,shape=1.):
    t=np.maximum(0.,(np.asarray(s)-onset)/(extent-onset))
    phi=(1-shape)*t*t+shape*t**3
    derivative=(2*(1-shape)*t+3*shape*t*t)/(extent-onset)
    return origin+slope*np.asarray(s)+amplitude*phi, slope+amplitude*derivative


def fit_horizontal(bounds,stations,reference,anchors,values,origin,slope,extent,direction):
    from .cloud_route import surface_at
    from .distant_wall import wall_at
    # The optimizer evaluates the same observations thousands of times. Prepare
    # arrays once per fit, without changing the public evidence or its precision.
    bounds=dict(bounds)
    for name in ['stations_m','lower_m','upper_m','inferred_axes']:
        if name in bounds:bounds[name]=np.asarray(bounds[name])
    wall=bounds.get('far_wall')
    if wall is not None:
        wall=dict(wall);wall['coefficients']=np.asarray(wall['coefficients'])
        bounds['far_wall']=wall
    weight=100.*(1+2*(stations/extent)**2)
    selected=stations>=wall['observed_range_m'][0] if wall is not None else None
    from .route_surface_fast import prepare,lateral_surface,residual_from_powers
    prepared_surface=prepare(bounds)
    def residual(parameters,global_bend=False):
        amplitude=parameters[0]
        onset=0. if global_bend else parameters[1]
        shape=parameters[1] if global_bend else 1.
        if prepared_surface is not None:
            t=np.maximum(0.,(stations-onset)/(extent-onset))
            at=np.maximum(0.,(anchors-onset)/(extent-onset))
            return residual_from_powers(t,t**3,at,at**3,stations,reference,anchors,values,weight,
                origin,slope,amplitude,onset,shape,extent,direction,*prepared_surface)
        def evaluate(s):
            return horizontal(s,origin,slope,amplitude,onset,extent,shape)
        y,dy=evaluate(stations)
        norm=np.sqrt(1+dy*dy);shift=WIDTH/2*dy/norm;half=WIDTH/2/norm
        left_s=stations+shift;right_s=stations-shift
        if prepared_surface is None:
            low,_=surface_at(bounds,left_s,lateral_only=True);_,high=surface_at(bounds,right_s,lateral_only=True)
            low=low[:,0];high=high[:,0]
        else:
            low,_=lateral_surface(left_s,*prepared_surface)
            _,high=lateral_surface(right_s,*prepared_surface)
        anchor,_=evaluate(anchors)
        # Equal weight per distance station, not per source return: dense near
        # points cannot drown out a sparse but coherent far surface.
        pieces=[100.*(anchor-values),.04*(y-reference),
                weight*np.minimum(y-half-low-.02,0.),
                weight*np.maximum(y+half-high+.02,0.)]
        if wall is not None:
            s=left_s if direction>0 else right_s
            d=y-half if direction>0 else y+half
            measured,_=wall_at(wall,s)
            clearance=direction*(d-measured)
            pieces.append(600.*np.minimum(clearance[selected]-.025,0.))
        return np.concatenate(pieces)
    if direction==0:
        parameters=np.array([0.,0.]);success=True;shape=1.
    else:
        magnitude=max(.1,direction*(reference[-1]-origin-slope*extent))
        if wall is not None:
            d,ds=wall_at(wall,np.array([extent]))
            target=d[0]+direction*(WIDTH/2+.05)*np.sqrt(1+ds[0]**2)
            magnitude=max(magnitude,direction*(target-origin-slope*extent))
        lower=[0.,0.] if direction>0 else [-2*extent,0.]
        upper=[2*extent,.75*extent] if direction>0 else [0.,.75*extent]
        candidates=[]
        onsets=[0.,.25*extent,.5*extent,.7*extent]
        # A bend already present at the sensor can have increasing or decreasing
        # curvature. r in [-.5,1] keeps 2(1-r)+6*r*t nonnegative everywhere.
        shapes=[-.25,.5]
        if prepared_surface is not None:
            # The same residual, bounds, scaling and tolerances in one compiled
            # call per start (bend_solver); 1 ms a start instead of 8.
            from .bend_solver import solve_bend
            data=(np.asarray(stations,float),np.asarray(reference,float),np.asarray(anchors,float),
                  np.asarray(values,float),np.asarray(weight,float),float(origin),float(slope),
                  float(extent),int(direction),*prepared_surface)
            starts=[(np.array([direction*min(1.9*extent,magnitude),onset]),
                     np.asarray(lower,float),np.asarray(upper,float),
                     np.array([max(1.,magnitude),extent]),False) for onset in onsets]
            starts+=[(np.array([direction*min(1.9*extent,magnitude),shape_start]),
                      np.array([lower[0],-.5]),np.array([upper[0],1.]),
                      np.array([max(1.,magnitude),1.]),True) for shape_start in shapes]
            solved=list(_STARTS.map(lambda start:solve_bend(*start,70,1e-8,1e-8,1e-6,*data),starts))
            for x,cost,ok in solved[:len(onsets)]:
                candidates.append((2*cost,x,bool(ok),1.))
            for x,cost,ok in solved[len(onsets):]:
                candidates.append((2*cost,np.array([x[0],0.]),bool(ok),float(x[1])))
        else:
            from .route_difference import forward_difference
            for onset in onsets:
                start=[direction*min(1.9*extent,magnitude),onset]
                fun,jac=forward_difference(residual,lower,upper)
                fit=least_squares(fun,start,jac=jac,bounds=(lower,upper),x_scale=[max(1.,magnitude),extent],
                                  max_nfev=70,ftol=1e-8,xtol=1e-8,gtol=1e-6)
                candidates.append((float(fit.fun@fit.fun),fit.x,bool(fit.success),1.))
            for shape_start in shapes:
                fun,jac=forward_difference(lambda p:residual(p,True),[lower[0],-.5],[upper[0],1.])
                fit=least_squares(fun,[direction*min(1.9*extent,magnitude),shape_start],jac=jac,
                                  bounds=([lower[0],-.5],[upper[0],1.]),x_scale=[max(1.,magnitude),1.],
                                  max_nfev=70,ftol=1e-8,xtol=1e-8,gtol=1e-6)
                candidates.append((float(fit.fun@fit.fun),np.array([fit.x[0],0.]),bool(fit.success),float(fit.x[1])))
        _,parameters,success,shape=min(candidates,key=lambda item:item[0])
    amplitude,onset=parameters
    if onset<1e-7:onset=0.
    knots=np.r_[np.zeros(4),([onset] if onset>0 else []),np.full(4,extent)]
    basis=BSpline(knots,np.eye(len(knots)-4),3)
    greville=np.array([np.mean(knots[j+1:j+4]) for j in range(len(knots)-4)])
    y,_=horizontal(greville,origin,slope,amplitude,onset,extent,shape)
    coefficients=np.linalg.solve(basis(greville),y)
    return knots,basis,coefficients,dict(amplitude_m=float(amplitude),onset_m=float(onset),
        direction=int(np.sign(amplitude)) if abs(amplitude)>.01 else 0,
        bend_count=int(abs(amplitude)>.01),optimizer_success=success,shape=float(shape),
        family='one convex cubic bend, optionally after a straight; curvature never changes sign')


def fit_height(basis,stations,lower,upper,anchors,values,origin,slope,extent):
    """A single quadratic vertical bend with the exact initial rail grade."""
    phi=(stations/extent)**2;line=origin+slope*stations
    anchor_phi=(anchors/extent)**2
    target=float(anchor_phi@(values-origin-slope*anchors)/(anchor_phi@anchor_phi+1e-6))
    positive=phi>1e-9
    lo=float(np.max((lower[positive]-line[positive])/phi[positive]))
    hi=float(np.min((upper[positive]-line[positive])/phi[positive]))
    if lo<=hi:
        amplitude=np.clip(target,lo,hi);success=True
    else:
        def residual(a):
            h=line+a[0]*phi
            return np.r_[100.*(origin+slope*anchors+a[0]*anchor_phi-values),
                         150.*np.minimum(h-lower,0.),150.*np.maximum(h-upper,0.)]
        fit=least_squares(residual,[target],max_nfev=50)
        amplitude=fit.x[0];success=bool(fit.success)
    greville=np.array([np.mean(basis.t[j+1:j+4]) for j in range(len(basis.t)-4)])
    height=origin+slope*greville+amplitude*(greville/extent)**2
    coefficients=np.linalg.solve(basis(greville),height)
    return coefficients,dict(amplitude_m=float(amplitude),optimizer_success=success,
                             family='single quadratic vertical bend')
