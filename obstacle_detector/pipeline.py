"""Compose route estimation and detection with an explicit decision contract.

No file IO, rendering, previous frames or mutable global state are required.
Confidence is an evidence score, never a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import time
import numpy as np

from .route import CorridorResult, extract_corridor
from .obstacles import Obstacle, ObstacleResult, detect


@dataclass(frozen=True)
class Config:
    min_confidence: float = 0.0
    rise_m: float = 0.10
    min_unique: int = 3
    self_return_m: float = 2.5
    # Numba threads while a frame is processed. The compiled kernels split
    # their work into fixed blocks, so the result does not depend on this
    # (checked bit for bit with 1, 3 and 12 threads); only the time does. On a
    # 6-core Ryzen 5 3600 the small per-frame kernels ran fastest with 2-3
    # threads, 12 cost about 25 ms per frame in thread wake-ups.
    threads: int = 3

    def __post_init__(self):
        if not np.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be finite and in [0, 1]")
        if not np.isfinite(self.rise_m) or self.rise_m <= 0:
            raise ValueError("rise_m must be finite and positive")
        if isinstance(self.min_unique, bool) or not isinstance(self.min_unique, int) or self.min_unique < 3:
            raise ValueError("min_unique must be an integer >= 3")
        if not np.isfinite(self.self_return_m) or self.self_return_m < 0:
            raise ValueError("self_return_m must be finite and nonnegative")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError("threads must be a positive integer")


@dataclass(frozen=True)
class Analysis:
    status: str
    path_clear: bool | None
    obstacles_detected: bool
    distance_m: float | None
    confidence: float | None
    obstacles: tuple[Obstacle, ...]
    reasons: tuple[str, ...]
    region: CorridorResult = field(repr=False)
    detection: ObstacleResult | None = field(repr=False)
    obstacle_mask: np.ndarray = field(repr=False)
    elapsed_ms: float
    config: Config

    def to_dict(self, *, include_indices=False):
        """JSON-ready summary; dense point arrays stay out of the JSON."""
        def group(obj):
            out = {f.name: getattr(obj, f.name) for f in fields(obj) if f.name != "indices"}
            out["reasons"] = list(out["reasons"])
            if include_indices:
                out["frame_indices"] = self.region.indices[obj.indices].tolist()
            return out
        return {
            "schema_version": "1.0", "mode": "single_frame",
            "status": self.status, "path_clear": self.path_clear,
            "obstacles_detected": self.obstacles_detected,
            "distance_m": self.distance_m, "confidence": self.confidence,
            "distance_definition": "minimum Euclidean distance from LiDAR to an accepted object return",
            "confidence_definition": "heuristic evidence score, not probability",
            "reasons": list(self.reasons),
            "route": {"status": self.region.status, "reasons": list(self.region.reasons),
                      "extent_m": self.region.extent_m,
                      "rails_end_m": getattr(self.region.geometry, "rails_end_m", None),
                      "heads_end_m": getattr(self.region.geometry, "heads_end_m", None),
                      "input_points": len(self.region.mask), "roi_points": len(self.region.points)},
            "obstacles": [group(obj) for obj in self.obstacles],
            "all_groups": [] if self.detection is None else [group(obj) for obj in self.detection.obstacles],
            "timing_ms": {"total": self.elapsed_ms, "route_and_roi": self.region.elapsed_ms,
                          "detector": 0.0 if self.detection is None else self.detection.elapsed_ms},
            "config": {f.name: getattr(self.config, f.name) for f in fields(self.config)},
        }


def _decision(region, detection, config):
    """Do not equate absence of detections with confirmed free space."""
    groups = () if detection is None else detection.obstacles
    accepted = tuple(sorted((o for o in groups if o.kind == "object"
                             and o.confidence >= config.min_confidence), key=lambda o: o.range_m))
    if accepted:
        return "BLOCKED", False, accepted, ("object_in_candidate_corridor",)
    reasons = []
    if region.geometry is None:
        reasons.append("route_unavailable")
    if region.status != "NO_DETECTED_CONTRADICTION":
        reasons.append("route_not_fully_supported:" + region.status)
    if detection is None:
        reasons.append("detector_not_run")
    else:
        reasons.extend(detection.reasons)
        if not len(detection.surface_m) or not np.isfinite(detection.surface_m).any():
            reasons.append("surface_unavailable")
        if any(o.kind in {"object", "unplaced", "transient", "structure"} for o in groups):
            # Structure is a geometric class, not proof that a tall object is harmless.
            reasons.append("unresolved_groups_in_corridor")
    if reasons:
        return "UNKNOWN", None, accepted, tuple(dict.fromkeys(reasons))
    return "CLEAR", True, accepted, ("no_obstacle_detected_in_observed_roi",)


def analyze_frame(xyz, *, ring=None, config=None):
    """Analyze one (N,3) float32/float64 cloud in metres; forward -Y, up +Z.

    CLEAR means no object detected within the modelled ROI under this algorithm.
    UNKNOWN means evidence cannot establish that result. BLOCKED reports even
    objects on a REJECTED candidate route and retains its diagnostic status.
    """
    config = Config() if config is None else config
    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if xyz.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("xyz must be float32/float64 coordinates in metres")
    if ring is not None:
        ring = np.asarray(ring)
        if ring.shape != (len(xyz),) or not np.issubdtype(ring.dtype, np.integer):
            raise ValueError("ring must be an integer array of length N")
    tick = time.perf_counter()
    region = extract_corridor(xyz, return_details=True, return_geometry=True)
    detection = None
    if region.geometry is not None:
        detection = detect(region, ring=ring, rise_m=config.rise_m,
                           min_unique=config.min_unique, self_return_m=config.self_return_m)
    status, clear, objects, reasons = _decision(region, detection, config)
    mask = np.zeros(len(xyz), dtype=bool)
    for obj in objects:
        mask[region.indices[obj.indices]] = True
    nearest = objects[0] if objects else None
    return Analysis(status, clear, bool(objects), None if nearest is None else nearest.range_m,
                    None if nearest is None else nearest.confidence, objects, reasons,
                    region, detection, mask, 1000 * (time.perf_counter() - tick), config)
