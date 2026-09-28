"""Causal route estimation, motion compensation and obstacle tracking."""
from dataclasses import dataclass, replace
import time

import numpy as np
from numba import config as numba_config, get_num_threads, set_num_threads

from .stabilization.route_filter import FilterConfig
from .stabilization.early_direction_filter import EarlyDirectionRouteFilter
from .pipeline import Config, Analysis, _decision
from .route.corridor import extract_corridor, select_with_geometry
from .obstacles import detect
from .result import StreamResult
from .tracking.ego_motion import EgoMotion
from .tracking.obstacle_tracker import ObstacleTracker, observations, CEILING

SELECTED_VARIANT = 'obstacle_detector'
SELECTED_ROUTE_CONFIG = FilterConfig(reach_quantile=1.)


class NominalStreamProcessor:
    """Process each cloud using only current and previous measurements.

    An optional raw_region supplies the current frame's route measurement.
    Otherwise process() runs every stage from XYZ. Motion is estimated using
    the previous raw route; detection and tracking use the current smoothed route.
    """
    def __init__(self, config=None):
        self.config = Config() if config is None else config
        self.route = EarlyDirectionRouteFilter(SELECTED_ROUTE_CONFIG)
        self.ego = EgoMotion()
        self.objects = ObstacleTracker()
        self.reset()

    def reset(self):
        self.route.reset()
        self.ego.reset()
        self.objects.reset()
        self.previous_raw_geometry = None
        self.confirmed_ids = set()
        self.diagnostics = {}
        self.last_motion = None

    def process(self, xyz, ring=None, timestamp_ns=None, *, raw_region=None):
        start = time.perf_counter()
        previous = get_num_threads()
        set_num_threads(max(1, min(self.config.threads, numba_config.NUMBA_NUM_THREADS)))
        try:
            stage = self.route_stage(xyz, ring, raw_region=raw_region)
            return self.after_route(stage, start=start)
        finally:
            set_num_threads(previous)

    def route_stage(self, xyz, ring=None, *, raw_region=None):
        """Route, ego motion and corridor selection: everything the detector needs.

        The stage holds no reference to the cloud beyond the selected region,
        so it can be stored and replayed through after_route() unchanged.
        """
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype not in (np.float32,np.float64):
            raise ValueError('xyz must be floating-point (N,3) coordinates')
        if ring is not None:
            ring = np.asarray(ring)
            if ring.shape != (len(xyz),) or not np.issubdtype(ring.dtype,np.integer):
                raise ValueError('ring must be an integer array of length N')
        shared = raw_region is not None
        # The raw route's own corridor is needed only when no smoothed route
        # exists, which happens only when the raw route itself is missing.
        raw = (extract_corridor(xyz,return_details=True,return_geometry=True,select=False)
               if raw_region is None else raw_region)
        if len(raw.mask) != len(xyz):
            raise ValueError('Shared raw region belongs to a different cloud')
        mark = time.perf_counter()
        motion = self.ego.update(xyz,self.previous_raw_geometry)
        ego_ms = 1000*(time.perf_counter()-mark)
        self.previous_raw_geometry = raw.geometry
        mark = time.perf_counter()
        geometry = self.route.update(raw.geometry,motion)
        filter_ms = 1000*(time.perf_counter()-mark)
        mark = time.perf_counter()
        if geometry is None:
            if raw.geometry is not None and raw_region is None:
                raw = extract_corridor(xyz,return_details=True,return_geometry=True)
            region = raw
        else:
            region = select_with_geometry(xyz,geometry)
            # Raw-path validation is not proof about a different smoothed path.
            # Preserve the existing decision contract: unvalidated free space
            # stays UNKNOWN; positive detections still produce BLOCKED.
            region = replace(region,status='TEMPORALLY_ESTIMATED',
                             reasons=('smoothed_route_not_independently_revalidated',))
        select_ms = 1000*(time.perf_counter()-mark)
        diagnostics = dict(self.route.diagnostics,raw_route_status=raw.status,
                           raw_route_available=raw.geometry is not None)
        return RouteStage(len(xyz),None if ring is None else ring[region.indices],region,geometry,motion,
                          raw.elapsed_ms,ego_ms,filter_ms,select_ms,shared,diagnostics)

    def after_route(self, stage, *, start=None):
        """Detector, tracker and decision on a route stage of the next frame."""
        start = time.perf_counter() if start is None else start
        region,geometry,motion = stage.region,stage.geometry,stage.motion
        self.last_motion = motion
        detection = None if region.geometry is None else detect(
            region,ring=stage.ring,rise_m=self.config.rise_m,
            min_unique=self.config.min_unique,self_return_m=self.config.self_return_m)
        status_frame,clear,accepted,reasons = _decision(region,detection,self.config)
        mask = np.zeros(stage.points,bool)
        for obstacle in accepted:
            mask[region.indices[obstacle.indices]] = True
        nearest = accepted[0] if accepted else None
        route_ms = stage.raw_ms + stage.filter_ms + stage.select_ms
        frame_ms = route_ms + (0. if detection is None else detection.elapsed_ms)
        analysis = Analysis(status_frame,clear,bool(accepted),
                            None if nearest is None else nearest.range_m,
                            None if nearest is None else nearest.confidence,
                            accepted,reasons,region,detection,mask,frame_ms,self.config)
        mark = time.perf_counter()
        tracks = tuple(self.objects.update(observations(detection,region),geometry,motion))
        blocking = tuple(self.objects.blocking(tracks))
        pending = any(not t.confirmed and t.object_hits>0 for t in tracks) and not blocking
        status = 'BLOCKED' if blocking else ('UNKNOWN' if status_frame=='BLOCKED' else status_frame)
        ids = {t.id for t in blocking}
        event = ('confirmed' if ids-self.confirmed_ids else
                 'released' if self.confirmed_ids-ids else '')
        self.confirmed_ids = ids
        track_ms = 1000*(time.perf_counter()-mark)
        total_ms = 1000*(time.perf_counter()-start) + (stage.raw_ms if stage.shared else 0.)
        timing = dict(total=total_ms,frame=frame_ms,route=route_ms,
                      raw_route=stage.raw_ms,route_filter=stage.filter_ms,select=stage.select_ms,
                      detect=0. if detection is None else detection.elapsed_ms,
                      ego=stage.ego_ms,obstacle_tracking=track_ms)
        self.diagnostics = stage.diagnostics
        return StreamResult(
            status,{'BLOCKED':False,'CLEAR':True,'UNKNOWN':None}[status],
            blocking[0].range_m if blocking else None,
            min(1.,blocking[0].evidence/CEILING) if blocking else None,
            pending,event,status_frame,analysis.distance_m,analysis.confidence,
            0 if detection is None else len(detection.obstacles),tracks,blocking,
            float(motion.travel_m),bool(motion.locked),region.status,analysis,timing)


@dataclass(frozen=True)
class RouteStage:
    """Output of the route stage of one frame; see NominalStreamProcessor.route_stage."""
    points: int                     # returns in the whole cloud
    ring: np.ndarray | None         # laser index per region row
    region: object
    geometry: object
    motion: object
    raw_ms: float
    ego_ms: float
    filter_ms: float
    select_ms: float
    shared: bool
    diagnostics: dict


class StreamProcessor(NominalStreamProcessor):
    """Nominal corridor detector/tracker followed by three-state gauge assessment."""
    def __init__(self, config=None, gauge_config=None):
        from .gauge import GaugeConfig, GaugeDecision
        config = Config() if config is None else config
        gauge_config = GaugeConfig(rise_m=config.rise_m,min_unique=config.min_unique,
                                  self_return_m=config.self_return_m) if gauge_config is None else gauge_config
        self.gauge_decision = GaugeDecision(gauge_config)
        super().__init__(config)

    def reset(self):
        super().reset()
        self.gauge_decision.reset()

    def after_route(self, stage, *, start=None):
        nominal = super().after_route(stage,start=start)
        result = self.gauge_decision.apply(nominal,None,self.last_motion)
        retired = set(result.gauge['lost_track_ids'])
        if retired:
            # Stop association and occlusion checks too, not just display.
            self.objects.tracks = [t for t in self.objects.tracks if t.id not in retired]
            self.confirmed_ids.difference_update(retired)
        return result
