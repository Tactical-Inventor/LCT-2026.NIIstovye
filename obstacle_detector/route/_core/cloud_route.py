"""Rail-anchored spatial spline fitted to outer cloud projections, at full range.

The cloud provides necessary outer bounds, not a collision-free volume. Missing
returns and genuinely incompatible fixed-size sections remain explicit.
"""
import time
import numpy as np
from scipy.interpolate import BSpline
from .train_envelope import WIDTH, HEIGHT


def _interp(x, stations, values):
    """Linear surface continuation only outside the observed station centers."""
    out = np.interp(x, stations, values)
    if len(stations) > 1:
        for a, b, mask in [(-2, -1, x > stations[-1])]:
            slope = (values[b] - values[a]) / (stations[b] - stations[a])
            out[mask] = values[a] + slope * (x[mask] - stations[a])
    return out


def surface_at(bounds, stations, lateral_only=False):
    """Outer bounds at the given stations; one compiled pass (route_surface_fast)."""
    from .route_surface_fast import outer_surface
    x=np.ascontiguousarray(stations,dtype=float)
    s=np.ascontiguousarray(bounds['stations_m'],dtype=float)
    low=np.asarray(bounds['lower_m'],dtype=float);high=np.asarray(bounds['upper_m'],dtype=float)
    if lateral_only:low=low[:,:1];high=high[:,:1]
    wall=bounds.get('far_wall')
    has_ranges='lateral_evidence_ranges' in bounds
    ranges=np.asarray(bounds.get('lateral_evidence_ranges',[]),dtype=float).reshape(-1,3)
    inferred=(np.asarray(bounds['inferred_axes'])[:,0].astype(float) if 'inferred_axes' in bounds
              else np.zeros(len(s)))
    return outer_surface(x,s,np.ascontiguousarray(low),np.ascontiguousarray(high),
                         np.ascontiguousarray(inferred),bool(bounds.get('preserve_tail_width')),
                         np.asarray(wall['coefficients'],dtype=float) if wall is not None else np.zeros(3),
                         float(wall['station_origin_m']) if wall is not None else 0.,
                         float(wall['station_scale_m']) if wall is not None else 1.,
                         float(wall['observed_range_m'][0]) if wall is not None else 0.,
                         int(wall['direction']) if wall is not None else 0,
                         float(wall.get('opposite_wall_reference_width_m',0.)) if wall is not None else 0.,
                         wall is not None,ranges,float(bounds.get('lateral_search_allowance_m',WIDTH)),has_ranges)


def _surface_at_arrays(bounds, stations, lateral_only=False):
    """The array version of surface_at, kept as its reference."""
    x = np.asarray(stations)
    s = np.asarray(bounds['stations_m'])
    low = np.asarray(bounds['lower_m']); high = np.asarray(bounds['upper_m'])
    if lateral_only:low=low[:,:1];high=high[:,:1]
    lo=np.column_stack([_interp(x, s, low[:,a]) for a in range(low.shape[1])])
    hi=np.column_stack([_interp(x, s, high[:,a]) for a in range(high.shape[1])])
    if bounds.get('preserve_tail_width'):
        tail=x>s[-1]
        middle=(lo[tail]+hi[tail])/2
        lo[tail]=middle-(high[-1]-low[-1])/2
        hi[tail]=middle+(high[-1]-low[-1])/2
    wall=bounds.get('far_wall')
    if wall is not None:
        from .distant_wall import wall_at
        d,slope=wall_at(wall,x);factor=np.sqrt(1+slope*slope)
        active=x>=wall['observed_range_m'][0]
        missing=active&(np.interp(x,s,np.asarray(bounds['inferred_axes'])[:,0].astype(float))>0)
        if wall['direction']>0:
            lo[active,0]=np.maximum(lo[active,0],d[active])
            hi[missing,0]=np.maximum(hi[missing,0],d[missing]+wall['opposite_wall_reference_width_m']*factor[missing])
        else:
            hi[active,0]=np.minimum(hi[active,0],d[active])
            lo[missing,0]=np.minimum(lo[missing,0],d[missing]-wall['opposite_wall_reference_width_m']*factor[missing])
    if 'lateral_evidence_ranges' in bounds:
        observed=lateral_observed(bounds,x)
        # Floor/roof extrema and an inferred opposite wall cannot certify a
        # lateral boundary. Keep raw projections separately for diagnostics.
        # Missing evidence is not a wall, but also not permission to send the
        # route arbitrarily far away. Limit the search to one vehicle width
        # beyond the nominal reconstructed corridor; it stays unconfirmed.
        allowance=bounds.get('lateral_search_allowance_m',WIDTH)
        lo[~observed[:,0],0]-=allowance
        hi[~observed[:,1],0]+=allowance
    return lo,hi


def lateral_observed(bounds,x):
    observed=np.zeros((len(x),2),bool)
    for begin,end,side in bounds.get('lateral_evidence_ranges',[]):
        observed[:,0 if side<0 else 1]|=(x>=begin-2)&(x<=end+2)
    wall=bounds.get('far_wall')
    if wall is not None:
        observed[:,0 if wall['direction']>0 else 1]|=x>=wall['observed_range_m'][0]
    return observed


def reconstruct_bounds(stations, lower, upper):
    """Bridge incomplete surface observations, explicitly retaining raw bounds.

    A window thinner than the vehicle plus a small evidence band does not
    identify two outer surfaces. Reconstruct its surfaces from adjacent wider
    observations; never manufacture a successful *raw* containment result.
    """
    low=lower.copy();high=upper.copy();inferred=np.zeros_like(low,dtype=bool)
    for axis,minimum in [(0,WIDTH+.4),(1,HEIGHT+.6)]:
        good=upper[:,axis]-lower[:,axis]>=minimum
        if good.sum()<2:continue
        missing=~good;inferred[:,axis]=missing
        lo=_interp(stations,stations[good],lower[good,axis])
        hi=_interp(stations,stations[good],upper[good,axis])
        width=np.interp(stations,stations[good],upper[good,axis]-lower[good,axis])
        if axis==0:
            # The center of one surviving wall must not become the track axis.
            middle=_interp(stations,stations[good],(lower[good,axis]+upper[good,axis])/2)
            tail=stations>stations[good][-1]
            middle[tail]=(lower[tail,axis]+upper[tail,axis])/2
            lo=middle-width/2;hi=middle+width/2
        else:
            # Missing distant floor returns otherwise look like a sudden ramp
            # towards the ceiling. Continue the measured roof and last depth.
            lo=hi-width
        low[missing,axis]=lo[missing];high[missing,axis]=hi[missing]
    return low,high,inferred


def route_excess(curves, bounds):
    """Check entire cross-sections at their own forward stations, in both views."""
    n = len(curves['center']); excess = np.zeros((n, 4))
    for t in np.linspace(0., 1., 9):
        edge = curves['left'] * (1-t) + curves['right'] * t
        low, high = surface_at(bounds, edge[:,0])
        h = curves['bottom']
        excess = np.maximum(excess, np.column_stack((low[:,0]-edge[:,1], edge[:,1]-high[:,0],
                                                      low[:,1]-h, h+HEIGHT-high[:,1])))
    return excess


def fit_cloud_route(path, local, envelope=None):
    if path.get('status') != 'PATH_ESTIMATED': return
    tick = time.perf_counter()
    extent = float(path['measured_forward_extent_m'])
    from .distant_wall import distance_edges,collect,turn_direction,fit_wall,apply_wall
    from .single_bend import fit_horizontal,fit_height
    heads=np.asarray(path['heads_local'])
    centers=(heads[::2]+heads[1::2])/2
    near=centers[centers[:,0]<=min(18.,centers[:,0].max())]
    if len(near)<2:near=centers
    dy,y0=np.polyfit(near[:,0],near[:,1],1)
    dz,z0=np.polyfit(near[:,0],near[:,2],1)
    from .rail_path_fast import sample_path
    prior=sample_path(dict(parameters=path['parameters']),path['length_m'],.25)
    # The near tracker stops at 25-31 m; the own rails stay visible on single
    # ring crossings to 50-60 m. They steer the lateral route only: heights and
    # the rail-support verdict keep the near heads.
    far=np.empty((0,2));far_candidates=0
    if envelope is not None:
        from .far_rails import far_rail_centers
        last=centers[np.argmax(centers[:,0])]
        far,far_candidates=far_rail_centers(local,envelope['origin'],envelope['basis'],
            lambda s:np.interp(s,prior['center'][:,0],prior['center'][:,1]),last[:2],
            path['gauge_m'],lambda s:z0+dz*s,extent)
    rail_s=np.r_[centers[:,0],far[:,0]];rail_y=np.r_[centers[:,1],far[:,1]]
    edges=distance_edges(extent)
    low,high,smin,smax,walls,ids,count,height_bits=collect(np.asarray(local),edges,z0,dz)
    required_span=np.maximum(.25,1.-.004*edges[:-1])
    observed=np.isfinite(low).all(axis=1)&(smax-smin>=np.minimum(required_span,extent/4))
    if np.count_nonzero(observed)<3:
        path['route_fit']=dict(status='INSUFFICIENT_EVIDENCE',reason='too_few_longitudinal_observations')
        return
    stations=(smin[observed]+smax[observed])/2
    raw_low,raw_high=low[observed],high[observed]
    direction=turn_direction(stations,raw_low,raw_high,dy)
    low,high,inferred=reconstruct_bounds(stations,raw_low,raw_high)
    if direction==0:
        # A shallow turn may not move both raw edges enough for the coherent
        # wall detector, yet a straight full-width vehicle can already cross
        # a boundary. Use that geometric conflict to choose one bend direction.
        tail=stations>=.4*extent
        line=y0+dy*stations
        # A distant isolated return can extend the cloud horizon beyond all
        # supported sections, especially behind an occluding obstacle. An
        # unobserved tail supplies no evidence for a bend.
        left=float(np.max(low[tail,0]-line[tail]+WIDTH/2, initial=0.))
        right=float(np.max(line[tail]+WIDTH/2-high[tail,0], initial=0.))
        if max(left,right)>.05:direction=1 if left>right else -1
    # Rails leaving the near line are a measured bend. They outrank the outer
    # extremes of the cloud: in a gate or at a platform those can turn the
    # other way, and a one-direction bend then cannot pass the own rails.
    deviation=(rail_y-(y0+dy*rail_s))[rail_s>18.]
    if len(deviation) and np.max(abs(deviation))>(.05 if direction==0 else .15):
        direction=1 if deviation[np.argmax(abs(deviation))]>0 else -1
    wall=fit_wall(edges,walls,ids,count,height_bits,direction,extent)
    raw_bounds=dict(stations_m=stations.tolist(),lower_m=raw_low.tolist(),upper_m=raw_high.tolist())
    bounds=dict(stations_m=stations.tolist(),lower_m=low.tolist(),upper_m=high.tolist(),
                inferred_axes=inferred.tolist(),raw=raw_bounds,preserve_tail_width=True,
                window_m=float(np.median(np.diff(edges))),window_widths_m=np.diff(edges)[observed].tolist(),
                window_edges_m=edges.tolist(),measured_forward_extent_m=extent,
                missing_intervals=int(np.count_nonzero(~observed)),
                semantics='range-adaptive outer bounds; observed turn wall retained as a SIDE')
    apply_wall(bounds,raw_low,raw_high,wall,np.asarray(local),edges,z0,dz)
    inferred=np.asarray(bounds['inferred_axes'])
    x=np.unique(np.r_[np.linspace(0.,extent,max(3,int(np.ceil(extent/.5))+1)),stations])
    anchor_s=centers[:,0]
    reference=np.interp(x,prior['center'][:,0],prior['center'][:,1])
    knots,basis,cy,turn=fit_horizontal(bounds,x,reference,rail_s,rail_y,y0,dy,extent,direction)
    slopes=basis(x,1)@cy
    sin=slopes/np.sqrt(1+slopes*slopes);cos=1/np.sqrt(1+slopes*slopes)
    left_low,left_high=surface_at(bounds,x+WIDTH/2*sin)
    right_low,right_high=surface_at(bounds,x-WIDTH/2*sin)
    lower_y=left_low[:,0]+WIDTH/2*cos;upper_y=right_high[:,0]-WIDTH/2*cos
    lower_z=np.maximum(left_low[:,1],right_low[:,1])
    upper_z=np.minimum(left_high[:,1],right_high[:,1])-HEIGHT
    cz,vertical=fit_height(basis,x,lower_z+.015,upper_z-.015,anchor_s,centers[:,2],z0,dz,extent)
    from .route_obstacles import extract,clearance,constraints,anchor_column_rows,_forward_indices,evidence_array
    prepared_points=_forward_indices(np.asarray(local),extent)
    evidence=extract(local,BSpline(knots,cy,3),BSpline(knots,cz,3),extent,prepared=prepared_points)
    # Held as an array for the clearance checks: the evidence keeps plain lists.
    evidence_points=np.asarray(evidence_array(evidence,'points'),float).reshape(-1,3)
    row_cache={}
    column_rows=anchor_column_rows(evidence,BSpline(knots,cy,3),cache=row_cache)
    gap,_,active=clearance(BSpline(knots,cy,3),BSpline(knots,cz,3),evidence_points,extent)
    row_conflict=False
    if column_rows:
        points=evidence_points;labels=evidence_array(evidence,'components')
        occupied=np.concatenate([np.column_stack((points[np.isin(labels,row['components']),:2],
                                  np.full(np.isin(labels,row['components']).sum(),row['side'])))
                                  for row in column_rows])
        lat=BSpline(knots,cy,3)
        row_conflict=bool(np.any(occupied[:,2]*(lat(occupied[:,0])-occupied[:,1])+
                               (WIDTH/2+.05)*np.sqrt(1+lat(occupied[:,0],1)**2)>0))
    # A single cubic bend that misses the measured far rails is refitted with
    # the variable-curvature bend, which holds every rail centre within 8 cm.
    far_miss=bool(len(far)) and float(np.max(abs(basis(far[:,0])@cy-far[:,1])))>.08
    if np.any(active&(gap<.05)) or row_conflict or far_miss:
        bounds['lateral_evidence_ranges']=evidence.get('side_ranges',[])
        bounds['unobserved_lateral_policy']='uncertain search band around nominal reconstruction; not measured free space'
        bounds['lateral_search_allowance_m']=WIDTH
        candidates=[];candidate_checks=[]
        fit_cache={}
        row_rejected=False
        for lock_rows in ([True,False] if column_rows else [False]):
            working_evidence=evidence if lock_rows else {k:v for k,v in evidence.items() if k!='column_rows'}
            candidates=[]
            for policy in [0,1,-1]:
                obstacles=constraints(working_evidence,policy)
                directions=[direction] if direction else [-1,1]
                for candidate_direction in directions:
                    from .convex_bend import fit as fit_convex
                    fit_key=(candidate_direction,obstacles.shape,obstacles.tobytes())
                    if fit_key not in fit_cache:
                        fit_cache[fit_key]=fit_convex(bounds,x,basis(x)@cy,rail_s,rail_y,y0,dy,extent,
                                                     candidate_direction,obstacles)
                    k,b,y,t=fit_cache[fit_key]
                    slope=b(x,1)@y;norm=np.sqrt(1+slope*slope)
                    ll,lh=surface_at(bounds,x+WIDTH/2*slope/norm)
                    rl,rh=surface_at(bounds,x-WIDTH/2*slope/norm)
                    z,v=fit_height(b,x,np.maximum(ll[:,1],rl[:,1])+.015,
                                   np.minimum(lh[:,1],rh[:,1])-HEIGHT-.015,anchor_s,centers[:,2],z0,dz,extent)
                    gap,_,active=clearance(BSpline(k,y,3),BSpline(k,z,3),evidence_points,extent)
                    penetration=float(np.max(np.maximum(-gap[active],0))) if active.any() else 0.
                    surface=max(float(np.max(ll[:,0]+WIDTH/2/norm-b(x)@y)),
                                float(np.max(b(x)@y+WIDTH/2/norm-rh[:,0])),0.)
                    rail=float(np.max(abs(b(rail_s)@y-rail_y)))
                    score=max(penetration,surface,max(0.,rail-.1))
                    deviation=float(np.mean((b(x)@y-basis(x)@cy)**2))
                    candidates.append((score,deviation,k,b,y,z,t,v,policy,rail))
                    candidate_checks.append(dict(column_side=policy,direction=candidate_direction,
                                                 max_penetration_m=penetration,max_surface_excess_m=surface,
                                                 max_rail_error_m=rail, column_rows_enforced=lock_rows))
                # A feasible nearby solution needs no alternate obstacle side.
                if min(a[0] for a in candidates)<.025:break
            best=min(candidates,key=lambda a:(round(a[0],3),a[1]))
            if best[0]<=.05 or not lock_rows:break
            # A coherent-looking row is still a hypothesis. If it conflicts
            # with other measured surfaces, retain the independently checked
            # obstacle solution and report the association as unresolved.
            row_rejected=True
        # An unsuccessful avoidance attempt is not a new track hypothesis.
        # Keep the smooth rail-anchored proposal when every refit still violates
        # the geometric tolerance, rather than displaying an aggressive detour
        # that did not actually resolve the contradiction.
        refit_rejected=best[0]>.05
        # That proposal is no better a hypothesis when it misses the measured
        # rails themselves. A refit through the rails is kept instead, and its
        # remaining collision stays in the verdict.
        initial_miss=float(np.max(abs(basis(rail_s)@cy-rail_y)))
        through_rails=[a for a in candidates if a[-1]<=min(.15,initial_miss/2)]
        rails_outrank=bool(refit_rejected and through_rails and initial_miss>.2)
        if rails_outrank:
            best=min(through_rails,key=lambda a:(round(a[0],3),a[1]));refit_rejected=False
        if not refit_rejected:
            _,_,knots,basis,cy,cz,turn,vertical,policy,_=best
        else:policy=0
        # Validate the final height/route against newly extracted evidence too;
        # a lateral/vertical change must not hide a newly encountered surface.
        evidence=extract(local,BSpline(knots,cy,3),BSpline(knots,cz,3),extent,prepared=prepared_points)
        evidence_points=np.asarray(evidence_array(evidence,'points'),float).reshape(-1,3)
        anchor_column_rows(evidence,BSpline(knots,cy,3),cache=row_cache)
        if row_rejected:
            evidence['column_row_rejected']='no_feasible_route_with_locked_column_row'
            for row in evidence.get('column_rows',[]):row['enforced']=False
        evidence['column_side_policy']=policy
        evidence['refit_candidates']=candidate_checks
        if rails_outrank:
            evidence['refit_kept_through_rails']=dict(reason='initial_bend_misses_rails',
                remaining_violation_m=float(best[0]))
        if refit_rejected:
            evidence['refit_rejected']=dict(reason='no_feasible_obstacle_refit',
                best_violation_m=float(best[0]),retained='initial_smooth_rail_anchored_bend')
    successes=[turn['optimizer_success'],vertical['optimizer_success']]
    dense = np.linspace(0.,extent,max(2,int(np.ceil(extent/.2))+1))
    dy = basis(dense,1)@cy; dz = basis(dense,1)@cz
    speed = np.sqrt(1+dy*dy+dz*dz)
    distance = np.r_[0.,np.cumsum(np.diff(dense)*(speed[:-1]+speed[1:])/2)]
    path['arc_model'] = path['model']
    path['arc_length_m'] = path['length_m']
    path['arc_validation'] = path.get('validation',{})
    path['model']='spatial_spline'
    path['parameters_role']='initial_arc_only'
    path['spatial_route']=dict(knots=knots.tolist(), lateral_coefficients=cy.tolist(),
                               height_coefficients=cz.tolist(), lookup_station_m=dense.tolist(),
                               lookup_distance_m=distance.tolist(),bounds=bounds,one_bend=turn,vertical_bend=vertical,
                               obstacles=evidence)
    path['length_m']=float(distance[-1])
    path['horizon_policy']='full measured forward extent; one-direction bend; no clipping'
    path['height_reference']='varying_profile_anchored_to_near_rail_heads'
    from .route_obstacles import audit
    obstacle_check=audit(path['spatial_route'],extent,evidence_points)
    curves=_sample_route(path,path['length_m'],.25,obstacle_check)
    excess=curves['projection_excess']
    raw_excess=curves['raw_projection_excess']
    # Recompute diagnostic bounds for the final (possibly obstacle-refined) bend.
    final_slope=basis(x,1)@cy;final_norm=np.sqrt(1+final_slope*final_slope)
    ll,lh=surface_at(bounds,x+WIDTH/2*final_slope/final_norm)
    rl,rh=surface_at(bounds,x-WIDTH/2*final_slope/final_norm)
    lower_y=ll[:,0]+WIDTH/2/final_norm;upper_y=rh[:,0]-WIDTH/2/final_norm
    lower_z=np.maximum(ll[:,1],rl[:,1]);upper_z=np.minimum(lh[:,1],rh[:,1])-HEIGHT
    # A narrow observed strip may be occlusion, not a physically narrow tunnel.
    # Report it explicitly; never shrink the cross-section or claim it fits.
    narrow=(upper_y<lower_y)|(upper_z<lower_z)
    max_excess=float(excess.max())
    conflict=excess>.05
    anchor_error=(basis(anchor_s)@cy-centers[:,1])
    status='NO_DETECTED_CONTRADICTION' if max_excess<=.05 and max(abs(anchor_error))<=.10 else 'REJECTED'
    reconstructed=bool(inferred.any() or raw_excess.max()>.05 or bounds['missing_intervals'])
    if status!='REJECTED' and (reconstructed or not all(successes)):
        status='INSUFFICIENT_EVIDENCE'
    path['route_fit']=dict(status=status, elapsed_ms=1000*(time.perf_counter()-tick),
                          optimizer_success=all(successes), max_projection_excess_m=max_excess,
                          bend_count=turn['bend_count'],turn_direction=turn['direction'],
                          bend_onset_m=turn['onset_m'],far_wall=wall,
                          max_raw_projection_excess_m=float(raw_excess.max()),
                          reconstructed_windows=int(inferred.any(axis=1).sum()),
                          max_rail_center_error_m=float(max(abs(anchor_error))),
                          incompatible_section_stations_m=x[narrow].tolist(),
                          conflicting_station_range_m=None if not conflict.any() else
                              [float(curves['center'][conflict,0].min()),float(curves['center'][conflict,0].max())],
                          start_height_m=float(z0),end_height_m=float(curves['bottom'][-1]))
    path['route_fit']['obstacle_check']=obstacle_check
    # Reported, not yet a verdict: the single-bend family may still miss them.
    path['route_fit']['far_rails']=dict(centers_m=far.tolist(),candidates=int(far_candidates),
        observed_end_m=float(far[-1,0]) if len(far) else float(centers[:,0].max()),
        max_center_error_m=float(np.max(abs(basis(far[:,0])@cy-far[:,1]))) if len(far) else None,
        curvature_per_m=rails_curvature(rail_s,rail_y),
        semantics='own-track centres from rail-head pairs on single lidar rings; lateral route only')
    all_conflicts=np.r_[curves['center'][conflict,0],path['route_fit']['obstacle_check']['conflicting_stations_m']]
    path['route_fit']['conflicting_station_range_m']=([float(all_conflicts.min()),float(all_conflicts.max())]
                                                     if len(all_conflicts) else None)
    path['validation']=_validate_route(path,curves,obstacle_check)
    status=path['validation']['status'];path['route_fit']['status']=status
    path['candidate_rejected']=status=='REJECTED';path['model_fit_consistent']=status!='REJECTED'
    path['full_path_validated']=False


def rails_curvature(stations,centres):
    """Mean curvature of the measured rail centres: one parabola through them all.

    The route spline's own curvature at the end of the rails wobbles by
    1-2e-4 1/m on a straight standing track; the rails themselves say whether
    the track turns there.
    """
    stations=np.asarray(stations,float);centres=np.asarray(centres,float)
    if len(stations)<4 or np.ptp(stations)<10:return 0.
    return float(2*np.polyfit(stations,centres,2)[0])


def sample_route(path, length, step=.25):
    return _sample_route(path,length,step)


def _sample_route(path, length, step=.25, obstacle_check=None):
    route=path['spatial_route']
    length=min(float(length),path['length_m'])
    u=np.linspace(0.,length,max(1,int(np.ceil(length/step)))+1)
    s=np.interp(u,route['lookup_distance_m'],route['lookup_station_m'])
    lateral=BSpline(route['knots'],route['lateral_coefficients'],3)
    height=BSpline(route['knots'],route['height_coefficients'],3)
    center=np.column_stack((s,lateral(s)))
    slope=lateral(s,1);normal=np.column_stack((-slope,np.ones(len(s))))/np.sqrt(1+slope*slope)[:,None]
    curves=dict(distance=u,center=center,left=center-WIDTH/2*normal,right=center+WIDTH/2*normal,bottom=height(s))
    curves['projection_excess']=route_excess(curves,route['bounds']).max(axis=1)
    curves['raw_projection_excess']=route_excess(curves,route['bounds']['raw']).max(axis=1)
    bounds=route['bounds']
    inferred=np.asarray(bounds['inferred_axes']).any(axis=1).astype(float)
    curves['surface_inferred']=np.interp(s,bounds['stations_m'],inferred)>0
    if 'lateral_evidence_ranges' in bounds:
        curves['surface_inferred']|=~lateral_observed(bounds,s).all(axis=1)
    from .route_obstacles import audit
    check=audit(route,path['measured_forward_extent_m']) if obstacle_check is None else obstacle_check
    bad=np.asarray(check['conflicting_stations_m'])
    curves['obstacle_collision']=np.zeros(len(s),bool)
    if len(bad):
        # Mark each local conflict, not the entire interval between two posts.
        ordered=np.sort(bad);j=np.searchsorted(ordered,s)
        distance=np.minimum(abs(s-ordered[np.clip(j,0,len(bad)-1)]),abs(s-ordered[np.clip(j-1,0,len(bad)-1)]))
        curves['obstacle_collision']=distance<.75
    return curves


def warmup_cloud_route():
    from .route_point_fast import warmup as warmup_points
    warmup_points()
    from .route_surface_fast import warmup
    warmup()
    from .distant_wall import collect,distance_edges,opposite_support
    local=np.zeros((2,3));edges=distance_edges(10.)
    collect(local,edges,0.,0.)
    opposite_support(local,edges,np.zeros(3),0.,1.,1,0.,0.)
    from .route_obstacles import _dense_groups,_components,_limits,_height_supported,_forward_indices,_closest_per_group
    _dense_groups(np.zeros((1,2),np.int64),np.zeros(2,np.int64),np.ones(2,np.int64),1)
    _limits(np.zeros((1,2),np.int64));_limits(np.zeros((1,2)))
    _height_supported(np.zeros(1,np.int64),np.zeros(1),1)
    _forward_indices(local,10.)
    _closest_per_group(np.zeros(2,np.int64),np.ones(2),local,1)
    _components(np.zeros((1,3)),np.zeros(1),np.empty((0,2),np.int64))
    from .spline_fast import warmup as warmup_splines
    warmup_splines()
    from .route_obstacles import _raised,_clearance
    knots=np.r_[np.zeros(4),np.full(4,10.)];c=np.zeros(4)
    unbounded=np.array([np.inf])
    _raised(local,np.arange(2),knots,c,c,3,2.3,-unbounded,unbounded,-unbounded,unbounded)
    _clearance(local,knots,c,c,3,10.,1.15,2.35)
    from .route_surface_fast import outer_surface
    s2=np.array([0.,5.]);b2=np.zeros((2,2))
    outer_surface(s2,s2,b2,b2+1,np.zeros(2),True,np.zeros(3),0.,1.,0.,1,1.,True,np.zeros((1,3)),2.3,True)
    from .bend_solver import solve_bend
    s=np.array([0.,5.,10.]);z=np.zeros(3)
    for global_bend in (False,True):
        solve_bend(np.array([1.,0.]),np.array([0.,-.5]),np.array([20.,1.]),np.ones(2),global_bend,3,1e-8,1e-8,1e-6,
                   s,z,s[:2],z[:2],np.ones(3),0.,0.,10.,1,s,z-2,z+2,z,True,np.empty(0))


def validate_route(path):
    """Public revalidation uses the spline, never its obsolete arc parameters."""
    c=sample_route(path,path['length_m'],.25)
    return _validate_route(path,c)


def _validate_route(path,c,obstacle_check=None):
    bounds=path['spatial_route']['bounds']
    fitted=float(c['projection_excess'].max());raw=float(c['raw_projection_excess'].max())
    r=path['spatial_route'];heads=np.asarray(path['heads_local']).reshape(-1,2,3).mean(axis=1)
    lateral=BSpline(r['knots'],r['lateral_coefficients'],3)
    rail_error=float(np.max(abs(lateral(heads[:,0])-heads[:,1])))
    reconstructed=bool(np.asarray(bounds['inferred_axes']).any() or raw>.05 or bounds['missing_intervals']
                       or 'lateral_evidence_ranges' in bounds)
    reasons=[]
    # For a cubic spline the second derivative is linear per knot interval;
    # its extrema are at the knots, so this checks the whole forward domain.
    curvature=lateral(np.unique(r['knots']),2)
    one_direction=not (np.any(curvature>1e-9) and np.any(curvature< -1e-9))
    if not one_direction:reasons.append('multiple_bend_directions')
    if fitted>.05:reasons.append('spatial_route_projection_conflict')
    if rail_error>.10:reasons.append('incompatible_with_rail_supports')
    from .route_obstacles import audit
    if obstacle_check is None:obstacle_check=audit(r,path['measured_forward_extent_m'])
    if obstacle_check['intersecting_return_count']:reasons.append('observed_obstacle_collision')
    if reasons:status='REJECTED'
    elif reconstructed or not path.get('route_fit',{}).get('optimizer_success',True):
        status='INSUFFICIENT_EVIDENCE'
        reasons.append('reconstructed_surface_intervals' if reconstructed else 'optimizer_incomplete')
    else:status='NO_DETECTED_CONTRADICTION'
    return dict(status=status,reasons=reasons,obstacle_check=obstacle_check,
                one_bend_direction_verified=one_direction,max_projection_excess_m=fitted,
                max_raw_projection_excess_m=raw,max_rail_center_error_m=rail_error,
                semantics='outer bounds plus observed occupied surfaces; unobserved volume is not certified')
