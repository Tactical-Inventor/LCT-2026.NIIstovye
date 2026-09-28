"""Assess only objects from the nominal detector and tracker, without cloud search."""
from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BSpline

from .decision import InnerConfig, InnerState, classify
from .geometry import path_distance


@dataclass(frozen=True)
class GaugeConfig(InnerConfig):
    inner_maximum_m: float = .10
    # Five consecutive measured overlaps, about 0.4 s from first to fifth at 10 Hz.
    promote_hits: int = 5
    promote_window: int = 5
    # Read observed returns across the track where the detector's own
    # cross-section found it, not across the route centre, which drifts off
    # the rails with range (0.3-0.6 m at 45-90 m on the roundT recordings).
    track_relative: bool = True
    # Detector readings that may count towards the consecutive entry hits.
    # Measured over the eight recordings: every true obstacle entered on a run
    # of 'object' readings; the false ones entered on readings the detector
    # itself set aside - a post cut off by the gauge ceiling and possibly
    # beside the track (new_data 4993-5008), a bracket below the rail heads
    # (new_data 4135-4142). Such readings still hold an object already
    # reported: a low obstacle close up often reads as level with the rails.
    promote_kinds: tuple = ('object', 'transient')
    # Detector notes that keep a reading from counting towards entry even when
    # its kind would: a broad face far past the rails stays an 'object' for the
    # detector, yet the track may pass beside it there.
    entry_vetoes: tuple = ('may_lie_beside_the_track_past_the_rails',)
    # A vehicle standing still sees the same forecast from the same place on
    # every frame, so consecutive hits there are one observation repeated.
    # Past the rails, where the track may lie more than standing_forecast_m
    # beside the route (above the uncertainty along the rails), entry waits
    # until the vehicle moves. Measured on new_data 7600-7860: standing, a post
    # 88 m ahead and 25-37 m past the rails (uncertainty 0.4-0.9 m) was read
    # alternately on the track and beside it for 250 frames.
    standing_m: float = .05
    standing_forecast_m: float = .25
    # Finite lifetime of predictions. Public BLOCKED history is independent of
    # the nominal tracker's much earlier existence confirmation.
    candidate_missing_frames: int = 8
    blocked_missing_frames: int = 20
    predicted_outside_frames: int = 3
    prediction_margin_m: float = .10
    prediction_margin_per_frame_m: float = .05

    def __post_init__(self):
        super().__post_init__()
        counts = (self.candidate_missing_frames, self.blocked_missing_frames,
                  self.predicted_outside_frames)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in counts):
            raise ValueError('Prediction lifetime thresholds must be positive integers')
        if self.blocked_missing_frames < max(self.candidate_missing_frames, self.missing_frames):
            raise ValueError('BLOCKED retention must cover candidate and decision retention')
        if any(not np.isfinite(v) or v < 0 for v in
               (self.prediction_margin_m, self.prediction_margin_per_frame_m)):
            raise ValueError('Prediction margins must be finite and nonnegative')

    def inner_width(self, distance):
        return np.full_like(np.asarray(distance, float), self.inner_maximum_m)


@dataclass
class Assessment:
    evidence: InnerState = field(default_factory=InnerState)
    offsets: np.ndarray | None = None
    missing: int = 0
    was_blocked: bool = False
    ever_blocked: bool = False
    predicted_outside_streak: int = 0
    retired_reason: str | None = None


def project_known_points(points, geometry):
    """Project a known object's predicted shape without selecting any new points."""
    g=geometry
    q=(np.asarray(points)-g.origin)@g.basis
    lat=BSpline(g.knots,g.lateral_coefficients,g.degree)
    bottom=BSpline(g.knots,g.height_coefficients,g.degree)
    lo=np.zeros(len(q));hi=np.full(len(q),g.extent_m)
    station=np.clip(q[:,0],lo,hi)
    for _ in range(48):
        centre=lat(station);slope=lat(station,1)
        residual=(station-q[:,0])+(centre-q[:,1])*slope
        done=np.abs(residual)<=1e-10
        if done.all():break
        hi=np.where((residual>0)&~done,station,hi)
        lo=np.where((residual<=0)&~done,station,lo)
        derivative=1+slope*slope+(centre-q[:,1])*lat(station,2)
        proposal=station-residual/np.maximum(derivative,1e-12)
        proposal=np.where((derivative>1e-12)&(proposal>lo)&(proposal<hi),proposal,(lo+hi)/2)
        station=np.where(done,station,proposal)
    slope=lat(station,1)
    lateral=((q[:,1]-lat(station))-(q[:,0]-station)*slope)/np.sqrt(1+slope*slope)
    return np.column_stack((station,lateral,q[:,2]-bottom(station)))


def track_frame(local, indices, detection, config):
    """Observed returns read across the track where the detector found it."""
    shift = getattr(detection, 'shift_m', None)
    if not config.track_relative or shift is None:
        return local
    local = local.copy()
    local[:, 1] -= np.nan_to_num(np.asarray(shift)[indices])
    return local


def forecast_spread(local, geometry):
    """How much further the track may lie beside the route at a group than along the rails."""
    uncertainty = getattr(geometry, 'lateral_uncertainty_m', None)
    if uncertainty is None or uncertainty(0.) is None:
        return 0.
    return max(0., float(uncertainty(float(np.median(local[:, 0])))) - float(uncertainty(0.)))


def matched_group(track, original, used):
    """Recover the original tracker's observation without regrouping points."""
    if not track.seen or original.detection is None:return None
    groups=original.detection.obstacles
    for index,obj in enumerate(groups):
        if index not in used and np.allclose(original.region.points[obj.indices].mean(axis=0),
                                            track.centre,atol=1e-6,rtol=0):return index
    # Close objects can retain a predicted centroid; their observed bounds
    # still identify the actual group used by the original tracker.
    for index,obj in enumerate(groups):
        if index in used:continue
        local=original.region.local[obj.indices]
        if (len(local) and np.allclose(local.min(axis=0),track.low,atol=1e-6,rtol=0) and
                np.allclose(local.max(axis=0),track.high,atol=1e-6,rtol=0)):return index
    return None


class GaugeProcessor:
    """Assess source tracks and retire predictions that are no longer relevant."""
    def __init__(self, config=GaugeConfig()):
        self.config=config
        self.reset()

    def reset(self):self.states={}

    def process(self, original, motion=None):
        cfg=self.config;g=original.geometry
        rows=[];observed={};used=set();retired={}
        # Pure unplaced/structure groups cannot initiate new gauge objects.
        sources=[t for t in original.tracks if t.object_hits>0 or t.confirmed]
        live={t.id for t in sources}
        self.states={tid:s for tid,s in self.states.items() if tid in live}
        nominal_blocking={t.id for t in original.blocking}
        standing=motion is not None and abs(float(motion.travel_m))<cfg.standing_m
        for track in sources:
            memory=self.states.setdefault(track.id,Assessment());state=memory.evidence
            if memory.retired_reason is not None:
                if not track.seen:
                    retired[track.id]=memory.retired_reason
                    continue
                # Standalone gauge users may still pass a retired nominal ID.
                # A new observation starts fresh evidence, never revives BLOCKED.
                memory=self.states[track.id]=Assessment();state=memory.evidence
            if memory.offsets is not None and motion is not None:
                memory.offsets=memory.offsets@motion.rotation
            index=matched_group(track,original,used)
            local=None;origin='unavailable';relation='unknown';deep=0;supported=None
            promotable=False
            if index is not None:
                used.add(index);obj=original.detection.obstacles[index]
                points=original.region.points[obj.indices];local=original.region.local[obj.indices]
                memory.offsets=points-np.asarray(track.centre)
                memory.missing=0;origin='observed'
                observed[track.id]=dict(point_indices=original.region.indices[obj.indices],
                    observed_points=points,observed_local=local)
                local=track_frame(local,obj.indices,original.detection,cfg)
                promotable=(obj.kind in cfg.promote_kinds
                            and not set(getattr(obj,'reasons',())).intersection(cfg.entry_vetoes)
                            and not (standing and g is not None
                                     and forecast_spread(local,g)>cfg.standing_forecast_m))
            else:
                memory.missing+=1
                if memory.offsets is not None and g is not None:
                    local=project_known_points(memory.offsets+np.asarray(track.centre),g)
                    origin='predicted'
            if local is not None and len(local) and g is not None:
                width=float(cfg.width(path_distance(g,local[:,0])).max(initial=0.))
                margin=max(state.retained_margin_m,width) if cfg.continuity else width
                relation,deep,supported=classify(local,g.half_width_m,g.gauge_height_m,
                                                 cfg.inner_maximum_m,margin,cfg.min_unique)
            else:margin=state.retained_margin_m
            outside_prediction=False
            if (origin=='predicted' and relation=='outside' and motion is not None
                    and motion.locked):
                # Use the whole shape projected onto the curved route. Beyond
                # measured rails include route uncertainty, plus growing drift
                # of the stale shape. A centroid or the straight sensor axis
                # cannot establish that the object is safely beside a curve.
                uncertainty=g.lateral_uncertainty_m(local[:,0])
                spread=(0. if uncertainty is None else
                        max(0.,float(np.max(uncertainty))-float(g.lateral_uncertainty_m(0.))))
                extra=cfg.prediction_margin_m + memory.missing*cfg.prediction_margin_per_frame_m
                outside_prediction=classify(local,g.half_width_m,g.gauge_height_m,
                    cfg.inner_maximum_m,margin+spread+extra,cfg.min_unique)[0]=='outside'
            memory.predicted_outside_streak=(memory.predicted_outside_streak+1
                                             if outside_prediction else 0)
            if origin=='observed':
                state.observe(SimpleNamespace(relation=relation,margin_m=margin,promotable=promotable,
                    source_ids=(track.id,),source_confirmed=track.confirmed),cfg)
            else:
                # A predicted shape supplies neither a deep hit nor an observed exit.
                state.observe(None,cfg);state.exists |= track.confirmed
            lifetime=cfg.blocked_missing_frames if memory.ever_blocked else cfg.candidate_missing_frames
            reason=None
            if memory.missing >= lifetime:
                reason='prediction_expired'
            elif memory.predicted_outside_streak >= cfg.predicted_outside_frames:
                reason='prediction_outside_corridor'
            elif state.outside_confirmed:
                reason='observed_outside_corridor'
            if reason is not None:
                memory.retired_reason=reason
                retired[track.id]=reason
                observed.pop(track.id,None)
                continue
            verdict=state.decision
            if verdict=='BLOCKED' and (track.id not in nominal_blocking or
                    memory.missing>cfg.missing_frames or
                    (origin=='predicted' and relation!='inside')):verdict='CAUTION'
            if origin!='observed' and verdict=='CLEAR':verdict='CAUTION'
            if verdict!='BLOCKED':
                # A downgraded track must earn entry again. A predicted return
                # into the core cannot revive an earlier BLOCKED decision.
                state.held_blocked=False
                if memory.was_blocked:state.deep_recent.clear()
            memory.was_blocked=verdict=='BLOCKED'
            memory.ever_blocked |= memory.was_blocked
            rows.append(dict(id=track.id,reference_ids=[track.id],source_ids=[track.id],
                decision=verdict,relation=relation,relation_source=origin,
                nominal_confirmed=track.id in nominal_blocking,
                source_kind=track.kind,detection_index=index,seen=origin=='observed',
                exists=state.exists,held_blocked=verdict=='BLOCKED',
                outside_confirmed=verdict=='CLEAR' and state.outside_confirmed,
                outside_streak=state.outside_streak,edge_streak=state.edge_streak,
                deep_hits=sum(state.deep_recent),deep_unique_points=deep,supported_gap_m=supported,
                unseen_frames=memory.missing,inner_margin_m=cfg.inner_maximum_m,margin_m=margin,
                predicted_outside_streak=memory.predicted_outside_streak,
                range_m=track.range_m,centre=list(track.centre),low=list(track.low),high=list(track.high)))
        reasons=[]
        if any(r['decision']=='BLOCKED' for r in rows):
            status='BLOCKED';reasons.append('nominal_obstacle_with_confirmed_inner_overlap')
        else:
            if any(r['decision']=='CAUTION' for r in rows):reasons.append('boundary_or_pending_nominal_object')
            if any(r['relation_source']!='observed' for r in rows):reasons.append('nominal_object_position_predicted')
            # CLEAR means no detected/tracked object in an available corridor.
            # A temporal route alone must not keep an empty scene in CAUTION.
            if g is None or original.detection is None:reasons.append('corridor_unavailable')
            status='CAUTION' if reasons else 'CLEAR'
            if status=='CLEAR':reasons.append('no_active_nominal_objects')
        return dict(status=status,path_clear={'CLEAR':True,'BLOCKED':False,'CAUTION':None}[status],
            source_status=original.status,reasons=reasons,tracks=rows,observed=observed,
            candidate_source='nominal_detector',tracking_policy='nominal_ids_with_gauge_retirement',
            clear_semantics='no_active_nominal_objects_in_available_corridor',
            new_shell_points=0,follow_shell_points=0,unassessed_blocking_ids=[],
            lost_track_ids=list(retired),retired_tracks=retired)
