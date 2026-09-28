"""Positive evidence for a broad upright face across the candidate corridor.

Ceiling clipping and a contaminated rail-crest estimate are not evidence that
such a face is harmless. Require several adjacent vertical columns, a shallow
longitudinal extent, and returns in the inner part of the gauge. Side walls,
narrow posts, overhead beams and distant unsupported route forecasts do not
qualify merely because they touch a boundary.
"""
import numpy as np

MIN_WIDTH_M = 0.6
MIN_HEIGHT_M = 0.6
COLUMN_M = 0.2
COLUMN_HEIGHT_M = 0.4
FRONT_TOLERANCE_M = 0.10


def transverse_face(local, members, usable, half_width, uncertainty=0.0):
    """Return supporting region indices, or an empty array.

    The seed comes from a connected raised group. Other layers at the same front
    are read directly from the cloud, because a broad obstacle can itself corrupt
    the estimated surface and hide its lower layers from that group. This remains
    evidence in a *candidate* corridor, not a certification of route geometry.
    """
    empty = np.empty(0, dtype=np.int64)
    if len(members) < 8 or uncertainty > 2 * half_width:
        return empty
    seed = local[members]
    low, high = seed.min(axis=0), seed.max(axis=0)
    depth, width, _ = high - low
    if width < MIN_WIDTH_M or depth > 0.25 * width + 0.05:
        return empty
    q = local[usable]
    keep = ((q[:, 0] >= low[0] - FRONT_TOLERANCE_M)
            & (q[:, 0] <= high[0] + FRONT_TOLERANCE_M)
            & (q[:, 2] >= 0.15))
    indices = usable[keep]
    face = local[indices]
    if len(face) < 12 or len(np.unique(face, axis=0)) < 12:
        return empty
    low, high = face.min(axis=0), face.max(axis=0)
    if high[1] - low[1] < MIN_WIDTH_M or high[2] - low[2] < MIN_HEIGHT_M:
        return empty
    columns = np.floor(face[:, 1] / COLUMN_M).astype(np.int64)
    tall = {int(c) for c in np.unique(columns)
            if np.ptp(face[columns == c, 2]) >= COLUMN_HEIGHT_M}
    # Three adjacent columns, with their middle inside the rail span. Two
    # separated uprights of an arch must not masquerade as a solid front.
    if not any(c + 1 in tall and c + 2 in tall and abs((c + 1.5) * COLUMN_M) <= 0.7
               for c in tall):
        return empty
    return indices
