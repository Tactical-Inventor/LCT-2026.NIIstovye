"""Find objects standing on the track inside an already selected corridor region.

The corridor package hands over the original returns inside the route together
with their route-local coordinates. This package decides which of them belong
to something standing on the track rather than to the track itself, and says
how sure it is. Nothing here selects points, estimates a route or judges
whether the line ahead is clear.

A candidate is a group of returns that rises above the surface its own part of
the cross-section has at its own station. The threshold is a height above that
local surface, not above the route profile, because the profile drifts with
range while the cross-section does not.

Range is not cut off. Far away the returns thin out - on doubleT_obstacle a
low object 56 m ahead leaves about a dozen of them, spread over two or three
rings - so a far candidate is reported with the low confidence its evidence
deserves, instead of being dropped or promoted to a certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from . import _groups
from ._surface import estimate
from ._faces import transverse_face

RISE_M = 0.10          # metres above the local surface a return must reach
MIN_UNIQUE = 3         # distinct returns a group needs to be reported at all
FULL_EVIDENCE = 5      # distinct returns at which a group is no longer thin
SELF_RETURN_M = 2.5    # returns closer than this are the carrier and its halo
LAYER_M = 0.05         # height resolution the layer count is measured at
MATCH_STATION_M = 1.0  # how far a group may move between frames and stay itself
MATCH_OFFSET_M = 0.5   # across the route, at short range
MATCH_ANGLE = 0.01     # ... growing by this times range, for the turn between frames
MATCH_HEIGHT_M = 0.4
MATCH_SLACK_M = 0.5    # how far a thing may seem to recede between frames
APPROACH_M = 3.5       # furthest a thing may come closer in one frame, 126 km/h at 10 Hz
CEILING_M = 0.25       # how close to the gauge ceiling counts as cut off by it
SIDE_M = 0.10          # how close to the gauge side counts as cut off by it
SIDE_TOP_M = 0.50      # above this a thing at the gauge side is not track edge
RAIL_REACH_M = 0.95    # beyond this offset a group no longer touches the track
ALONG_M = 3.0          # stations a narrow group must span to read as a wall
NARROW_M = 0.6
RAIL_MARGIN_M = 0.05   # how far above the rail heads a top must reach to clear them
FLAT_LENGTH_M = 0.3    # a group this long or longer may read as a patch of surface
FLAT_RATIO = 8.0       # ... when it is this many times longer than it is tall
FLAT_TOP_M = 0.25      # ... and no taller than this
LINE_TOP_M = 0.6       # a group lower than this may be one piece of a line along the track
LINE_REACH_M = 8.0     # stations either side the rest of such a line is looked for
LINE_GAP_M = 0.5       # ... leaving out the group's own neighbourhood
LINE_SLOPE = 0.1       # the line may cross the route at most this steeply
LINE_TOL_M = 0.08      # raised returns this close across to the line belong to it
LINE_BIN_M = 1.0       # the line must be raised in this many separate metres of station
LINE_BINS = 3
LINE_SPAN_M = 5.0      # ... and run at least this far, the group included


@dataclass(frozen=True)
class Obstacle:
    """One group of returns that stands above the track surface around it.

    Positions are route-local: station along the route, offset across it, both
    from the geometry the region was selected with. track_offset_m is the same
    offset measured from the rails where the frame shows them, which can be
    tenths of a metre away from the route centre far along a curve. height_m is measured from
    the local surface under the group, not from the route profile, so it is
    comparable between near and far groups. clearance_m is the same top read
    against the track on either side of the group instead: it is negative for
    something that rises out of the drainage channel without reaching the
    walkway beside it. confidence is this frame's own evidence, in 0..1; it is
    not a probability and carries no guarantee.

    kind sorts the group by what its shape says about it, not by what it is:

    * 'object' stands on the track, or presents a broad upright face across
      the candidate corridor even when clipped by its ceiling;
    * 'structure' is cut off by an edge of that gauge - the ceiling above or a
      side beyond the rails - or runs along the route while staying narrow: a
      post, a wall, an arch, a platform edge. What it really says is that the
      thing continues outside the corridor, so the corridor only holds a slice
      of it. A broad transverse face is checked separately before this rule;
    * 'transient' would be an object, but the frame before - when given - had
      nothing raised where it would have been had it stood still;
    * 'unplaced' would be an object, but it stands where the route is a
      forecast rather than a measurement - past the end of the rails the route
      was fitted to - and close enough to the side that the track itself may
      pass beside it: its innermost return plus position_uncertainty_m lies
      beyond the gauge. It may be on the track; the frame cannot say;
    * 'track_furniture' belongs to the track rather than stands on it: it does
      not reach the level of the track beside it, does not clear the rail
      heads, or is a patch of surface many times longer than it is tall.

    This is geometry, not recognition. Read it as a sorting of the evidence,
    and keep the groups it sets aside rather than discarding them.

    position_uncertainty_m is how far the track may lie beside the route
    centre at the group's station (RouteGeometry.lateral_uncertainty_m), 0
    when the geometry does not say.
    """
    indices: np.ndarray = field(repr=False)
    kind: str
    station_m: float
    offset_m: float
    track_offset_m: float
    range_m: float
    height_m: float
    clearance_m: float
    top_above_profile_m: float
    base_height_m: float
    length_m: float
    width_m: float
    returns: int
    unique_returns: int
    rings: int
    layers: int
    surface_support: int
    confidence: float
    confirmed: bool
    reasons: tuple[str, ...]
    position_uncertainty_m: float = 0.0


@dataclass(frozen=True)
class DetectorState:
    """What the next frame needs to recognise the same group again."""
    stations: np.ndarray = field(repr=False)
    offsets: np.ndarray = field(repr=False)
    origin: np.ndarray | None = field(default=None, repr=False)
    basis: np.ndarray | None = field(default=None, repr=False)
    # Returns above the surface, in the sensor's own coordinates: the next
    # frame looks for its groups among them, whatever route it is read under.
    raised: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ObstacleResult:
    """Per-return findings for one frame, in the order of region.points."""
    mask: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    obstacles: tuple[Obstacle, ...]
    surface_m: np.ndarray = field(repr=False)
    excess_m: np.ndarray = field(repr=False)
    state: DetectorState = field(repr=False)
    elapsed_ms: float
    reasons: tuple[str, ...] = ()
    # How far the track lies beside the route centre at each return, as the
    # cross-section itself shows it (NaN where the return was not read).
    shift_m: np.ndarray | None = field(default=None, repr=False)

    @property
    def objects(self):
        """The groups that stand on the track and fit under the gauge.

        The rest are still in obstacles: a group set aside as structure was
        sorted by its shape, not identified, and the shape of a vehicle
        standing on the track is the shape of something tall.
        """
        return tuple(found for found in self.obstacles if found.kind == 'object')

    @property
    def structures(self):
        """The groups the gauge ceiling cut off, or that run along the route."""
        return tuple(found for found in self.obstacles if found.kind == 'structure')


def _checked(region):
    for name in ('points', 'local', 'geometry', 'indices'):
        if not hasattr(region, name):
            raise TypeError('region must be a corridor result carrying local coordinates; '
                            'call extract_corridor(..., return_details=True, return_geometry=True)')
    if region.local is None or region.geometry is None:
        raise ValueError('region has no local coordinates; pass return_geometry=True')
    if len(region.local) != len(region.points):
        raise ValueError('region.local and region.points disagree in length')
    return region


def _aligned(values, region, name):
    """Accept a per-region array, or a whole-frame one to index with the region."""
    if values is None:
        return None
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(name + ' must be one-dimensional')
    if len(values) == len(region.points):
        return values
    if len(values) > len(region.points) and int(region.indices.max(initial=-1)) < len(values):
        return values[region.indices]
    raise ValueError(name + ' matches neither the region nor the frame it came from')


def _seen_before(before, now, distance, closer):
    """Whether any return of a group had a raised return near it a frame ago.

    Both are given along, across and up the current route. A thing standing
    still comes closer by the distance the vehicle travelled, so it was
    further along then by an amount within `closer`; across and in height it
    stays put, give or take the turn of the vehicle between the two frames.
    """
    if not len(before) or not len(now):
        return False
    # Only the previous returns inside the group's own window can answer, and
    # a large region a frame ago holds thousands of them.
    widest = MATCH_OFFSET_M + MATCH_ANGLE * float(distance.max())
    low, high = now.min(axis=0), now.max(axis=0)
    near = before[(before[:, 0] >= low[0] + closer[0]) & (before[:, 0] <= high[0] + closer[1])
                  & (before[:, 1] >= low[1] - widest) & (before[:, 1] <= high[1] + widest)
                  & (before[:, 2] >= low[2] - MATCH_HEIGHT_M)
                  & (before[:, 2] <= high[2] + MATCH_HEIGHT_M)]
    if not len(near):
        return False
    for start in range(0, len(now), 256):
        part = now[start:start + 256]
        ahead = near[None, :, 0] - part[:, None, 0]
        aside = np.abs(near[None, :, 1] - part[:, None, 1])
        above = np.abs(near[None, :, 2] - part[:, None, 2])
        reach = (MATCH_OFFSET_M + MATCH_ANGLE * distance[start:start + 256])[:, None]
        if np.any((ahead >= closer[0]) & (ahead <= closer[1])
                  & (aside <= reach) & (above <= MATCH_HEIGHT_M)):
            return True
    return False


def _on_a_line(members, rows, station, across, excess):
    """Whether a low group is one piece of a raised line running along the track.

    Far along the route a rail, a kerb or a cable trough that the forecast lets
    into the corridor comes back as a row of short raised pieces, one per laser
    ring, that the grouping cannot join across the metres between rings. Each
    piece alone reads as a small object. Measured on roundT_pressureGate_roundT
    frames 115-141: ten such pieces, 0.2-0.4 m high, on one straight line from
    0.9 m to 0.3 m across over 45-68 m, followed as a dozen overlapping tracks.
    A thing standing on the track is compact and has no such line through it.
    """
    s0 = float(np.median(station[members]))
    y0 = float(np.median(across[members]))
    others = rows[~np.isin(rows, members)]
    s = station[others] - s0
    y = across[others] - y0
    near = ((np.abs(s) <= LINE_REACH_M) & (np.abs(s) >= LINE_GAP_M)
            & (np.abs(y) <= LINE_SLOPE * np.abs(s) + LINE_TOL_M)
            & (excess[others] < LINE_TOP_M))
    if np.count_nonzero(near) < LINE_BINS:
        return False
    s, y = s[near], y[near]
    for slope in np.unique(np.round(y / s, 3)):
        inline = np.abs(y - slope * s) <= LINE_TOL_M
        if np.count_nonzero(inline) < LINE_BINS:
            continue
        on = s[inline]
        if (len(np.unique(np.floor(on / LINE_BIN_M))) >= LINE_BINS
                and max(on.max(), 0.) - min(on.min(), 0.) >= LINE_SPAN_M):
            return True
    return False


def _confidence(excess, clearance, unique, layers, support, rise):
    """Weigh the evidence a single frame offers, without pretending it is more.

    Each term saturates where more of it stops telling us anything: a return
    three thresholds above the surface is as convincing as one ten thresholds
    above, and a dozen returns settle the matter as well as a hundred. The
    terms are added, not multiplied, so one thin term lowers the confidence
    instead of erasing the candidate.
    """
    strength = min(1.0, max(0.0, (excess - rise) / (2 * rise)))
    standing = min(1.0, max(0.0, clearance / rise + 0.5))
    sampling = min(1.0, max(0.0, (unique - MIN_UNIQUE) / 8.0))
    stack = min(1.0, max(0.0, (layers - 1) / 2.0))
    ground = min(1.0, support / 12.0)
    weighed = (0.30 * strength + 0.15 * standing + 0.25 * sampling
               + 0.18 * stack + 0.12 * ground)
    # A handful of returns cannot be convincing however well they score: three
    # of them say almost nothing about extent, and the other terms would
    # otherwise carry such a group to the middle of the scale on their own.
    # Measured on doubleT_obstacle, the groups that turned out to be noise
    # were the ones with two or three returns.
    return weighed * min(1.0, (unique - 1) / (FULL_EVIDENCE - 1))


def detect(region, *, ring=None, rise_m=RISE_M, min_unique=MIN_UNIQUE,
           self_return_m=SELF_RETURN_M, previous=None, station_shift_m=None,
           approach_m=APPROACH_M, shape_quantile=None):
    """Report the groups of returns that stand above the track surface.

    region is the result of extract_corridor(..., return_details=True,
    return_geometry=True), or of select_with_geometry; both carry the local
    coordinates this works in. ring is optional: the sensor's laser index per
    return, either one value per selected row or one per row of the frame the
    region came from. It is not required - a height-layer count stands in for
    it - but it separates a vertical face from a slope more reliably, because
    two rings can land at one height while one ring cannot land at two.

    previous is the state of the frame before. With it, a group is confirmed
    when the frame before had a raised return where the group would have been
    if it stands still: at most approach_m further along the route, which is
    how far the vehicle can travel in one frame, and at the same offset and
    height. The comparison is made in the sensor's coordinates, so it holds
    whether the route was fitted afresh or held. An object not confirmed this
    way is set aside as 'transient'. station_shift_m, when the caller knows
    how far the vehicle travelled, narrows the window to a metre around it.
    Without previous nothing is confirmed and nothing is set aside for it.
    """
    region = _checked(region)
    tick = time.perf_counter()
    count = len(region.points)
    mask = np.zeros(count, bool)
    labels = np.full(count, -1, np.int32)
    surface = np.full(count, np.nan)
    excess = np.full(count, np.nan)
    empty = DetectorState(np.empty(0), np.empty(0), raised=np.empty((0, 3)))
    if not count:
        return ObstacleResult(mask, labels, (), surface, excess, empty,
                              1000 * (time.perf_counter() - tick), ('empty_region',))

    ring = _aligned(ring, region, 'ring')
    points = np.asarray(region.points, float)
    station, offset, height = np.asarray(region.local, float).T
    distance = np.linalg.norm(points, axis=1)
    # Returns from within arm's length of the sensor are its own mounting, not
    # track: they carry no intensity and sit above the rails at station zero.
    usable = np.flatnonzero(distance >= self_return_m)
    if len(usable) < 3:
        return ObstacleResult(mask, labels, (), surface, excess, empty,
                              1000 * (time.perf_counter() - tick), ('too_few_returns',))

    half_width = float(region.geometry.half_width_m)
    ceiling = float(getattr(region.geometry, 'gauge_height_m', None) or 2.4)
    track = estimate(station[usable], offset[usable], height[usable], half_width,
                     **({} if shape_quantile is None
                        else dict(shape_quantile=shape_quantile)))
    surface[usable] = track.base
    excess[usable] = height[usable] - track.base
    beside = np.full(count, np.nan)
    beside[usable] = track.shoulder
    rails = np.full(count, np.nan)
    rails[usable] = track.crest
    near = np.zeros(count, int)
    near[usable] = track.support
    # Offsets from the track itself rather than from the route centre: the
    # centre is a fit and wanders off the rails with range on a curve.
    beside_track = offset.copy()
    beside_track[usable] = offset[usable] - track.shift
    shift = np.full(count, np.nan)
    shift[usable] = track.shift
    rows = usable[excess[usable] > rise_m]
    heads_end = getattr(region.geometry, 'heads_end_m', None)
    uncertainty = getattr(region.geometry, 'lateral_uncertainty_m', None)
    if uncertainty is not None and uncertainty(0.0) is None:
        uncertainty = None

    found = []
    for part in _groups.group(station[rows], offset[rows], height[rows], distance[rows]):
        members = rows[part]
        unique = int(len(np.unique(points[members], axis=0)))
        if unique < min_unique:
            continue
        mask[members] = True
        top = int(members[np.argmax(excess[members])])
        levels = int(len(np.unique(np.round(height[members] / LAYER_M))))
        # Per return, so the answer is whether any part of the group rises
        # above the track beside that part, not above some other part's.
        clearance = float(np.nanmax(height[members] - beside[members]))
        crown = float(height[members].max())
        length = float(np.ptp(station[members]))
        width = float(np.ptp(offset[members]))
        note = []
        # The corridor stops at the clearance gauge, so anything continuing
        # past it comes back cut off flat at the ceiling. Measured on
        # roundT_doubleT: posts, arches and walls crowd into the top 0.2 m,
        # while the person on doubleT_obstacle tops out at 1.38 m and the
        # object left on the rails at 0.11 m.
        clipped = crown >= ceiling - CEILING_M
        # The same thing sideways: a group pressed against the side of the
        # corridor that never reaches as far in as the rails came into the
        # gauge from outside it. Measured on roundT_doubleT: 33 of the 34
        # groups lying wholly beyond the rails also touch that side.
        #
        # It must also be low. What lies along the side of a track at the edge
        # of the gauge is its shoulder, a trough lip, a platform edge; someone
        # standing out there is not, and would otherwise be set aside for
        # standing too far from the rails. Anything taller than half a metre
        # therefore stays an object unless the ceiling takes it.
        # Touching the side is about the corridor, which the route centre
        # drew; how far in the group reaches is about the rails, measured
        # where they are.
        across = np.abs(beside_track[members])
        edged = (crown < SIDE_TOP_M and float(across.min()) > RAIL_REACH_M
                 and float(np.abs(offset[members]).max()) >= half_width - SIDE_M)
        # The corridor follows the route centre, so where that centre has
        # wandered off the track the corridor takes in a strip beyond the
        # gauge. Measured on roundT_pressureGate_roundT: at 45-65 m the rails
        # lie 0.3-0.4 m to one side of the centre, and the frame of the gate
        # and the tunnel wall fill that strip. A group wholly beyond the gauge
        # measured from the rails cannot touch a train on them.
        outside = float(across.min()) >= half_width
        # Something that runs along the route for metres while staying narrow
        # is how a wall, a platform edge or a neighbouring rail looks from
        # inside the corridor.
        along = length > ALONG_M and width < NARROW_M
        if clipped:
            note.append('cut_off_by_the_gauge_ceiling')
        if edged:
            note.append('cut_off_by_the_gauge_side')
        if along:
            note.append('runs_along_the_route')
        if outside:
            note.append('outside_the_gauge_of_the_rails')
        # Nothing on a railway stands higher than the rail heads at the same
        # station, so a group whose own top stays under them sits inside the
        # track structure: a bracket in the drainage channel, a fastening, a
        # sleeper end. Measured on roundT_doubleT, this is what the returns
        # tracked from 27 m in to 3.6 m as the train approached turned out to
        # be - a real thing, compact and repeatable, and below the rails.
        sunken = crown < float(np.nanmin(rails[members]))
        if sunken:
            note.append('below_the_rail_heads')
        # Level with them is the same thing seen through an estimate: the crest
        # is read off the near cross-section, and a rail where that section has
        # none - a switch blade, a check rail, a diverging track, a rail the
        # route centre has slid across - rises above the surface under it and
        # tops out right at the rail heads. Measured over the three recordings,
        # half the false objects topped them by less than 0.11 m and a quarter
        # by less than 0.03 m, while the object left on the rails cleared them
        # by 0.14 m at the least and the person by 0.15 m.
        # Near the sensor the route's own profile is the rail-head level, as
        # measured by the near rail tracker, and needs no estimate of the
        # crest. Measured on doubleT_platform frames 140-141: where the surface
        # estimate sagged, the rails themselves and a 10 m stretch of the whole
        # track bed stood 0.2 m above it with their top 0.01 m over the profile.
        on_profile = (heads_end is not None and float(np.median(station[members])) <= heads_end
                      and crown < RAIL_MARGIN_M)
        level = (not sunken and (on_profile or float(np.nanmax(height[members] - rails[members]))
                                 < RAIL_MARGIN_M))
        if level:
            note.append('level_with_the_rail_heads')
        # A thing standing on the track shows a face, so its height span is
        # comparable to its length: at most 1.75 times shorter for every group
        # of the person and the object. A group many times longer than it is
        # tall is a patch of the track surface itself standing higher than the
        # near cross-section says - a platform, the plates of a switch, a
        # crossing - and on squareT_platform_squareT_switch that is what the
        # 25 m long, 0.2 m tall groups in frames 700-800 are.
        rise = float(np.ptp(height[members]))
        flat = length > FLAT_LENGTH_M and rise < min(FLAT_TOP_M, length / FLAT_RATIO)
        if flat:
            note.append('flat_along_the_route')
        if clearance < 0:
            note.append('below_the_track_beside_it')
        if unique < 5:
            note.append('few_returns')
        if ring is None:
            note.append('height_layers_instead_of_rings')
        # Past the end of the rails the route was fitted to, the route is a
        # forecast, and the track may run beside it by more than the distance
        # from this group to the side of the gauge. Measured against the rails
        # later frames saw: 0.3-0.5 m wrong 20-30 m past the rails, over a
        # metre 40 m past them on a curve. A wall or a gate frame that the
        # forecast put inside the corridor is then indistinguishable from a
        # thing on the track - of the 46 objects reported on the five clean
        # recordings 36 stood 5-150 m past the rails. What can be said is only
        # that the track may miss it.
        # The cross-section's own shift (section 5a) reaches a few metres past
        # the rails too, but it is no better there than the route: measured
        # against the same truth, 0.16-0.54 m against 0.06-0.18 m at the 90th
        # percentile within 5 m past the rails. The forecast starts at the rails.
        spread = float(uncertainty(float(np.median(station[members])))) if uncertainty else 0.0
        unplaced = uncertainty is not None and float(across.min()) + spread > half_width
        if unplaced:
            note.append('may_lie_beside_the_track_past_the_rails')
        # A dense frontal obstacle can reach the ceiling and can raise the
        # estimated crest to its own top. Neither heuristic makes it a wall or
        # a rail fastening. Check direct, multi-column vertical evidence before
        # those exclusions; keep ordinary objects and their scores unchanged.
        face = (clearance > 0.2 and (clipped or edged or along or outside
                or sunken or level or flat or unplaced)
                and len(transverse_face(np.asarray(region.local), members, usable,
                                        half_width, spread)) > 0)
        if face:
            note.append('broad_transverse_face_in_candidate_corridor')
        found.append(dict(
            indices=members,
            kind=('object' if face
                  else 'structure' if clipped or edged or along or outside
                  else 'track_furniture' if sunken or level or flat or clearance < 0
                  else 'unplaced' if unplaced
                  else 'object'),
            position_uncertainty_m=spread,
            station_m=float(np.median(station[members])),
            offset_m=float(np.median(offset[members])),
            track_offset_m=float(np.median(beside_track[members])),
            range_m=float(distance[members].min()),
            height_m=float(excess[top]), clearance_m=clearance,
            top_above_profile_m=crown, base_height_m=float(surface[top]),
            length_m=length, width_m=width,
            returns=int(len(members)), unique_returns=unique,
            rings=int(len(np.unique(ring[members]))) if ring is not None else levels,
            layers=levels, surface_support=int(np.median(near[members])),
            reasons=tuple(note)))

    for entry in found:
        if (entry['kind'] in ('object', 'unplaced') and entry['height_m'] < LINE_TOP_M
                and 'broad_transverse_face_in_candidate_corridor' not in entry['reasons']
                and _on_a_line(entry['indices'], rows, station, beside_track, excess)):
            entry['kind'] = 'track_furniture'
            entry['reasons'] = entry['reasons'] + ('piece_of_a_line_along_the_track',)

    reasons = []
    axes = np.asarray(region.geometry.basis, float)
    before = None
    if previous is not None:
        if getattr(previous, 'raised', None) is None:
            reasons.append('previous_frame_has_no_raised_returns_recorded')
        else:
            # Both frames in the sensor's own coordinates, read along, across
            # and up the current route. A fresh route per frame changes the
            # route frame but not the sensor's, so this holds either way.
            before = np.asarray(previous.raised, float) @ axes
    if station_shift_m is None:
        closer = (-MATCH_SLACK_M, approach_m + MATCH_SLACK_M)
    else:
        closer = (station_shift_m - MATCH_STATION_M, station_shift_m + MATCH_STATION_M)

    obstacles = []
    for entry in found:
        confidence = _confidence(entry['height_m'], entry['clearance_m'],
                                 entry['unique_returns'], entry['layers'],
                                 entry['surface_support'], rise_m)
        confirmed = False
        if before is not None:
            confirmed = _seen_before(before, points[entry['indices']] @ axes,
                                     distance[entry['indices']], closer)
            # A thing lying on the track is there again next rotation; ballast
            # resampled a little differently is not. Confirmation therefore
            # raises confidence and its absence lowers it.
            confidence = min(1.0, confidence * 1.3) if confirmed else confidence * 0.7
            # Measured with a route fitted on every frame: of the objects
            # reported where there is nothing, 34 of 43 on roundT_doubleT and
            # 31 of 38 on squareT_platform_squareT_switch had no raised return
            # anywhere near in the frame before, while the person and the
            # object were found there in 202 of 204 frames. What costs this is
            # one frame of delay for something that has just appeared.
            if entry['kind'] == 'object' and not confirmed:
                entry = dict(entry, kind='transient',
                             reasons=entry['reasons'] + ('not_seen_in_the_previous_frame',))
        obstacles.append(Obstacle(confidence=float(confidence), confirmed=confirmed, **entry))

    # Objects first, so the list reads as what to look at before what to
    # account for; within a kind, the strongest evidence leads.
    order = {'object': 0, 'transient': 1, 'unplaced': 2, 'track_furniture': 3, 'structure': 4}
    obstacles.sort(key=lambda found: (order[found.kind], -found.confidence, found.station_m))
    for number, obstacle in enumerate(obstacles):
        labels[obstacle.indices] = number
    state = DetectorState(np.array([o.station_m for o in obstacles]),
                          np.array([o.offset_m for o in obstacles]),
                          np.asarray(region.geometry.origin),
                          np.asarray(region.geometry.basis),
                          points[rows])
    return ObstacleResult(mask, labels, tuple(obstacles), surface, excess, state,
                          1000 * (time.perf_counter() - tick), tuple(reasons), shift)


def warmup():
    """Compile the grouping kernel once before timing a stream of frames."""
    _groups.warmup()
