"""Result of one sequential pipeline update."""
from dataclasses import dataclass, field
from .pipeline import Analysis

@dataclass(frozen=True)
class StreamResult:
    """Decision of the stream for one frame, with what it was based on."""
    status: str                     # BLOCKED / CAUTION / CLEAR (nominal diagnostic may be UNKNOWN)
    path_clear: bool | None
    distance_m: float | None        # nearest blocking object, otherwise nearest warning object
    confidence: float | None
    pending: bool                   # an object is being followed but not yet confirmed
    event: str
    status_frame: str               # the frame's own decision, as analyze_frame gives it
    distance_frame_m: float | None
    confidence_frame: float | None
    groups_frame: int
    tracks: tuple = field(repr=False)
    blocking: tuple = field(repr=False)
    travel_m: float = 0.0
    ego_locked: bool = False
    route_status: str = ''
    analysis: Analysis | None = field(default=None, repr=False)
    timing_ms: dict = field(default_factory=dict)
    nominal: object | None = field(default=None, repr=False)
    gauge: dict | None = field(default=None, repr=False)

    @property
    def region(self):
        return self.analysis.region

    @property
    def detection(self):
        return self.analysis.detection

    @property
    def geometry(self):
        """The stabilized analysis route; None where it is unavailable."""
        return self.analysis.region.geometry

    @property
    def route_measured(self):
        """The route is temporally estimated, not independently validated."""
        return False

