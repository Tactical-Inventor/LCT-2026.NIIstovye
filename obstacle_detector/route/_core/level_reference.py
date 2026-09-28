"""Level the output without changing which rail/wall source supports were found."""
import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
from .rail_path_fast import sample_path,_rail_terms,signed_distance
from .train_envelope import build_envelope


@njit(cache=True,parallel=True)
def reproject(local,rotation,translation):
    result=np.empty_like(local)
    for i in prange(len(local)):
        for j in range(3):
            result[i,j]=translation[j]
            for k in range(3):result[i,j]+=local[i,k]*rotation[k,j]
    return result


def _parameters(parameters,length,rotation,translation,fast=False):
    source=sample_path(dict(parameters=parameters),length,max(.5,length/20))['center']
    points=np.column_stack((source,np.zeros(len(source))))@rotation+translation
    tangent=np.array([np.cos(parameters[1]),np.sin(parameters[1]),0.])@rotation
    yaw=np.arctan2(tangent[1],tangent[0])
    offset=points[0,1]-points[0,0]*np.tan(yaw)
    if parameters[2]==0:return [float(offset),float(yaw),0.],0.
    scale=np.linalg.norm(tangent[:2])
    k=parameters[2]*np.linalg.det(rotation[:2,:2])/scale**3
    if fast:return [float(offset),float(yaw),float(k)],None
    targets=np.zeros(len(points));terms=lambda p:_rail_terms(points[:,:2],np.asarray(p),targets)
    fit=least_squares(lambda p:terms(p)[0],[offset,yaw,k],jac=lambda p:terms(p)[1],
                      max_nfev=6,x_scale=[.1,.03,.002])
    return fit.x.tolist(),float(np.max(abs(terms(fit.x)[0])))


def level_path(path,envelope,rail,evidence,fast=False):
    level=build_envelope(rail['pose'],level=True,rail_height=float(np.median(rail['head_centers'][:,2])))
    if path.get('status')!='PATH_ESTIMATED':return level
    rotation=envelope['basis'].T@level['basis'];translation=(envelope['origin']-level['origin'])@level['basis']
    path['parameters'],error=_parameters(path['parameters'],path['length_m'],rotation,translation,fast)
    path['rail_only_parameters'],_= _parameters(path['rail_only_parameters'],path['observed_forward_range_m'][1],rotation,translation,fast)
    heads=np.asarray(path['heads_local'])@rotation+translation
    path['heads_local']=heads.tolist();path['observed_forward_range_m']=[float(heads[:,0].min()),float(heads[:,0].max())]
    for row in path.get('wall_observations',[]):
        p=np.array([row['s'],row['d'],row.get('h',1.2)])@rotation+translation
        row.update(s=float(p[0]),d=float(p[1]),h=float(p[2]))
    rows=path.get('wall_observations',[])
    if rows:
        path['wall_observed_range_m']=[min(r['s'] for r in rows),max(r['s'] for r in rows)]
        path['wall_observed_end_by_side_m']={str(side):max((r['s'] for r in rows if r['side']==side),default=0.) for side in (-1,1)}
    if evidence is not None:
        evidence['local']=reproject(evidence['local'],rotation,translation)
        evidence['grid_reference_origin']=envelope['origin'].tolist()
        evidence['grid_reference_basis']=envelope['basis'].tolist()
        path.setdefault('wall_evidence_grid',{}).update(reference_origin=envelope['origin'].tolist(),reference_basis=envelope['basis'].tolist())
    path['level_projection_max_arc_error_m']=error
    path['height_reference']='constant_sensor_z_at_near_rails'
    return level


@njit(cache=True)
def level_extent(xyz,origin,forward):
    extent=0.
    for p in xyz:
        if not np.isfinite(p[0]+p[1]+p[2]):continue
        s=0.
        for j in range(3):s+=(p[j]-origin[j])*forward[j]
        extent=max(extent,s)
    return extent


def warmup_level():
    reproject(np.zeros((2,3)),np.eye(3),np.zeros(3))
    _parameters([0.,0.,.002],20.,np.eye(3),np.zeros(3))
    for dtype in (np.float32,np.float64):level_extent(np.zeros((2,3),dtype=dtype),np.zeros(3),np.eye(3)[:,0])
