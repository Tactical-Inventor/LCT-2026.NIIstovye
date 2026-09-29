"""Transport-independent validation and JSON conversion."""
import json
import math
import numpy as np
from obstacle_detector.io import decode_pointcloud_message


def header_dict(header):
    return dict(frame_id=header.frame_id,
                timestamp_ns=int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec))


def cloud_arrays(message):
    """Keep finite XYZ and the corresponding integer rings in original order."""
    frame = decode_pointcloud_message(message)
    xyz, ring = np.asarray(frame.xyz, dtype=np.float64), frame.ring
    if ring is not None and (ring.shape != (len(xyz),) or not np.issubdtype(ring.dtype, np.integer)):
        raise ValueError('ring must be a scalar integer field')
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], None if ring is None else ring[finite], int((~finite).sum())


def packed_cloud(xyz, ring=None, *, float64=False, allow_nonfinite=False):
    """Little-endian XYZ and optional UINT16 ring; preserve NPY float64 on request."""
    xyz = np.asarray(xyz)
    dtype = np.dtype('<f8' if float64 else '<f4')
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Expected XYZ with shape (N, 3)')
    finite = np.isfinite(xyz)
    if (not allow_nonfinite and not finite.all()) or np.any(np.abs(xyz[finite]) > np.finfo(dtype).max):
        raise ValueError('Output XYZ must be finite and representable in the output dtype')
    if ring is None:
        return np.ascontiguousarray(xyz, dtype=dtype).tobytes(), 3*dtype.itemsize
    ring = np.asarray(ring)
    if (ring.shape != (len(xyz),) or not np.issubdtype(ring.dtype, np.integer)
            or np.any(ring < 0) or np.any(ring > 65535)):
        raise ValueError('ring must contain N integers in [0, 65535]')
    records = np.empty(len(xyz), dtype=[('xyz', dtype, (3,)), ('ring', '<u2')])
    records['xyz'], records['ring'] = xyz, ring
    return records.tobytes(), 3*dtype.itemsize + 2


def result_dict(result, header, index, *, removed_points=0, reset_reason=None):
    tracks = [dict(id=int(t.id), decision=t.decision, confirmed=bool(t.confirmed),
                   seen=bool(t.seen), exists=bool(t.exists), kind=t.kind,
                   range_m=float(t.range_m), centre_xyz=list(t.centre),
                   station_m=float(t.station_m), offset_m=float(t.offset_m),
                   speed_m_per_frame=float(t.speed_m_per_frame)) for t in result.tracks]
    return dict(schema_version='1.0', header=header_dict(header), frame_index=index,
                processing_ok=True, status=result.status, path_clear=result.path_clear,
                distance_m=result.distance_m, confidence=result.confidence,
                route_available=result.geometry is not None, route_status=result.route_status,
                travel_m=result.travel_m, blocking_ids=[int(t.id) for t in result.blocking],
                tracks=tracks, timing_ms=result.timing_ms, removed_points=removed_points,
                reset_reason=reset_reason)


def json_text(value):
    """Emit strict JSON; numerical unknowns become null instead of NaN."""
    def clean(item):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list, np.ndarray)):
            return [clean(v) for v in item]
        return item
    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False, separators=(',', ':'))


class SequenceGuard:
    """Reset temporal evidence when a different sensor sequence arrives."""
    def __init__(self, max_gap_sec=0.25):
        if not math.isfinite(max_gap_sec) or max_gap_sec < 0:
            raise ValueError('max_gap_sec must be finite and nonnegative')
        self.max_gap_ns = round(max_gap_sec * 1e9)
        self.previous = None

    def update(self, header):
        current = header_dict(header)
        reason = None
        if self.previous is not None:
            delta = current['timestamp_ns'] - self.previous['timestamp_ns']
            if current['frame_id'] != self.previous['frame_id']:
                reason = 'frame_id_changed'
            elif delta < 0:
                reason = 'timestamp_moved_backwards'
            elif self.max_gap_ns and delta > self.max_gap_ns:
                reason = 'timestamp_gap'
        self.previous = current
        return reason
