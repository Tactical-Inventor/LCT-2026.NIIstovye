"""Measured obstacle returns and predicted boxes."""
from types import SimpleNamespace
import numpy as np


def gauge_display(result, xyz, box_history, box_rotation):
    """Draw nominal detector returns in red/amber; forecasts use corners."""
    region=result.region
    history={} if box_history is None else box_history
    if box_rotation is not None:
        for track_id in history:
            history[track_id]=history[track_id]@box_rotation
    points=[np.asarray(region.points)];cursor=len(region.points);obstacles=[]
    mask=np.zeros(len(xyz),bool);warning_mask=np.zeros(len(xyz),bool)
    for track in result.tracks:
        if not track.exists or track.outside_confirmed or track.decision=='CLEAR':
            continue
        centre=np.asarray(track.centre)
        observed=track.observed_points
        if track.seen and len(observed):
            offsets=observed-centre
            low,high=offsets.min(axis=0),offsets.max(axis=0)
            history[track.id]=np.array([[a,b,c] for a in (low[0],high[0])
                                       for b in (low[1],high[1]) for c in (low[2],high[2])])
            points.append(observed)
            ids=np.arange(cursor,cursor+len(observed));cursor+=len(observed)
            (mask if track.decision=='BLOCKED' else warning_mask)[track.point_indices]=True
            obstacles.append(SimpleNamespace(indices=ids,range_m=track.range_m,
                track_id=track.display_id,internal_id=track.id,decision=track.decision,predicted=False,
                stop_station_m=float(track.observed_local[:,0].min())))
        else:
            offsets=history.get(track.id)
            if offsets is None:
                size=np.maximum(np.asarray(track.high)-np.asarray(track.low),.02)
                basis=np.eye(3) if result.geometry is None else result.geometry.basis
                offsets=(np.array([[a,b,c] for a in (-.5,.5) for b in (-.5,.5)
                                   for c in (-.5,.5)])*size)@np.asarray(basis).T
                history[track.id]=offsets
            points.append(offsets+centre)
            obstacles.append(SimpleNamespace(indices=np.arange(cursor,cursor+8),range_m=track.range_m,
                track_id=track.display_id,internal_id=track.id,decision=track.decision,predicted=True))
            cursor+=8
    active={t.id for t in result.tracks}
    for track_id in list(history):
        if track_id not in active:del history[track_id]
    # Blockers first: the subtitle/distance describe the same nearest alert.
    obstacles.sort(key=lambda o:(o.decision!='BLOCKED',o.range_m))
    shown_region=SimpleNamespace(points=np.vstack(points),indices=region.indices,
        mask=region.mask,local=region.local,geometry=result.geometry,status=region.status)
    assert sum(o.decision=='BLOCKED' for o in obstacles)==len(result.blocking)
    return SimpleNamespace(region=shown_region,obstacles=tuple(obstacles),obstacle_mask=mask,
                           warning_mask=warning_mask,gauge_config=result.gauge['config'])
