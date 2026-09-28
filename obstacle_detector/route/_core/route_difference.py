"""Bounded float64 forward differences for the small route least-squares fits.

Keep SciPy's default 2-point steps, bounds adjustment and array arithmetic;
avoid its generic dtype/worker/sparsity dispatch for every two-column Jacobian.
"""
import numpy as np


def forward_difference(function,lower,upper):
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    cached=[None,None]
    def fun(x):
        value=function(x)
        cached[:]=[x.copy(),value]
        return value
    def jac(x):
        if cached[0] is None or not np.array_equal(cached[0],x):fun(x)
        f0=cached[1]
        sign=(x>=0).astype(float)*2-1
        h=np.finfo(float).eps**.5*sign*np.maximum(1.,np.abs(x))
        lower_dist=x-lower;upper_dist=upper-x
        violated=(x+h<lower)|(x+h>upper)
        fitting=np.abs(h)<=np.maximum(lower_dist,upper_dist)
        h[violated&fitting]*=-1
        forward=(upper_dist>=lower_dist)&~fitting
        backward=(upper_dist<lower_dist)&~fitting
        h[forward]=upper_dist[forward];h[backward]=-lower_dist[backward]
        transposed=np.empty((len(x),len(f0)))
        for i in range(len(x)):
            shifted=x.copy();shifted[i]=x[i]+h[i]
            transposed[i]=(function(shifted)-f0)/((x[i]+h[i])-x[i])
        return transposed.T
    return fun,jac
