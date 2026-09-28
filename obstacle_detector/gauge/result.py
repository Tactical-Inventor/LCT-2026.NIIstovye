"""Apply gauge decisions to the public stream result and its actual returns."""
from dataclasses import asdict, dataclass, field, replace
import time
import numpy as np

from ..tracking.obstacle_tracker import TrackReport, CEILING
from .processor import GaugeConfig, GaugeProcessor


@dataclass(frozen=True)
class GaugeTrackReport(TrackReport):
    decision: str = 'CAUTION'
    relation: str = 'unknown'
    exists: bool = False
    outside_confirmed: bool = False
    inner_margin_m: float = .10
    outer_margin_m: float = 0.
    nominal_ids: tuple = ()
    point_indices: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64), repr=False)
    observed_points: np.ndarray = field(default_factory=lambda: np.empty((0,3)), repr=False)
    observed_local: np.ndarray = field(default_factory=lambda: np.empty((0,3)), repr=False)

    @property
    def display_id(self):
        return '/'.join(map(str,self.nominal_ids[:2])) if self.nominal_ids else f'M{self.id}'


class GaugeDecision:
    """Shared integration for raw-cloud streaming and verified nominal replay."""
    def __init__(self, config=GaugeConfig()):
        self.config = config
        self.processor = GaugeProcessor(config)
        self.reset()

    def reset(self):
        self.processor.reset()
        self.confirmed_ids = set()
        self.previous_status = None

    def apply(self, nominal, xyz, motion):
        if getattr(nominal,'nominal',None) is not None:
            raise ValueError('GaugeDecision requires a nominal result, not an already updated result')
        tick = time.perf_counter()
        # xyz intentionally is not searched: every observation comes from nominal.
        decision = self.processor.process(nominal,motion)
        active = {t.id:t for t in nominal.tracks}
        tracks = []
        for row in decision['tracks']:
            base = active[row['id']]
            values = asdict(base)
            values.update(confirmed=row['decision']=='BLOCKED',seen=row['seen'],
                          kind='object' if row['decision']=='BLOCKED' else 'unplaced')
            extra = decision['observed'].get(row['id'],{})
            tracks.append(GaugeTrackReport(**values,decision=row['decision'],relation=row['relation'],
                exists=row['exists'],outside_confirmed=row['outside_confirmed'],
                inner_margin_m=self.config.inner_maximum_m,outer_margin_m=row['margin_m'],
                nominal_ids=tuple(row['reference_ids']),**extra))
        tracks=tuple(tracks)
        blocking=tuple(sorted((t for t in tracks if t.decision=='BLOCKED'),key=lambda t:t.range_m))
        warnings=tuple(sorted((t for t in tracks if t.decision=='CAUTION' and t.exists),key=lambda t:t.range_m))
        status=decision['status']
        assert bool(blocking)==(status=='BLOCKED')
        relevant=blocking if blocking else warnings
        nearest=relevant[0] if relevant else None
        ids={t.id for t in blocking}
        event=('confirmed' if ids-self.confirmed_ids else
               'released' if self.confirmed_ids-ids else
               'warning' if status=='CAUTION' and status!=self.previous_status else '')
        self.confirmed_ids=ids; self.previous_status=status
        elapsed=1000*(time.perf_counter()-tick)
        timing=dict(nominal.timing_ms,gauge=elapsed,
                    total=nominal.timing_ms.get('total',0.)+elapsed)
        diagnostics={k:v for k,v in decision.items() if k!='observed'}
        diagnostics['config']=asdict(self.config)
        return replace(nominal,status=status,path_clear=decision['path_clear'],
            distance_m=None if nearest is None else nearest.range_m,
            confidence=None if nearest is None else float(np.clip(nearest.evidence/CEILING,0.,1.)),
            pending=not blocking and any(t.decision=='CAUTION' for t in tracks),event=event,
            tracks=tracks,blocking=blocking,timing_ms=timing,nominal=nominal,gauge=diagnostics)
