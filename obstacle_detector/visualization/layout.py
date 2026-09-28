"""Render a processed LiDAR frame with route, tracks and status.

Perspective camera is exactly at the LiDAR origin: forward -Y, up +Z.
No corridor is drawn in the perspective panel. BEV has explicitly enlarged
lateral scale for readability. This script does not encode or create a video.
"""
from pathlib import Path
import os
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import BSpline

HERE = Path(__file__).resolve().parent
S = 1
W, H = 1920, 1080
BG = '#080e18'
PANEL = '#0c1522'
TEXT = '#eaf1f8'
MUTED = '#8394a9'
CYAN = '#53d4de'
FORECAST = '#f2b15c'
RED = '#ff655d'
FONT = None
BOLD = None


def configure_fonts(regular=None, bold=None):
    """Use Segoe UI on Windows or DejaVu Sans when available."""
    global FONT, BOLD
    if (regular is None) != (bold is None):
        raise ValueError('Specify both --font and --font-bold')
    windows = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    candidates = [(regular, bold)] if regular is not None else [
        (windows / 'segoeui.ttf', windows / 'segoeuib.ttf'),
        (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
         Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')),
    ]
    for normal_path, bold_path in candidates:
        if Path(normal_path).is_file() and Path(bold_path).is_file():
            FONT, BOLD = Path(normal_path), Path(bold_path)
            return
    raise FileNotFoundError('No Cyrillic fonts found. Specify --font and --font-bold (TTF).')


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), round(size * S))


def rgb(color):
    return np.array(tuple(bytes.fromhex(color.lstrip('#'))), dtype=float)


def text(draw, xy, label, size=22, color=TEXT, bold=False, anchor=None):
    draw.text(tuple(round(v * S) for v in xy), label, font=font(size, bold),
              fill=color, anchor=anchor)


def line(draw, points, color, width=1):
    draw.line([(round(x*S), round(y*S)) for x, y in points], fill=color,
              width=max(1, round(width*S)), joint='curve')


def rounded(draw, box, fill, radius=12, outline=None, width=1):
    draw.rounded_rectangle(tuple(round(v*S) for v in box), radius=round(radius*S),
                           fill=fill, outline=outline, width=round(width*S))


def dashed_box(draw, box, color=RED, width=2, dash=7):
    x0,y0,x1,y1 = box
    for a,b in [((x0,y0),(x1,y0)),((x1,y0),(x1,y1)),
                ((x1,y1),(x0,y1)),((x0,y1),(x0,y0))]:
        a,b = np.asarray(a,float),np.asarray(b,float)
        length = float(np.linalg.norm(b-a))
        if length <= 0:
            continue
        direction = (b-a)/length
        for start in np.arange(0.,length,2*dash):
            line(draw,[a+direction*start,a+direction*min(start+dash,length)],color,width)


def as_sensor(geometry, stations, offsets):
    s, d = np.broadcast_arrays(stations, offsets)
    lateral = BSpline(geometry.knots, geometry.lateral_coefficients, geometry.degree)
    height = BSpline(geometry.knots, geometry.height_coefficients, geometry.degree)
    slope = lateral(s, 1)
    norm = np.sqrt(1. + slope*slope)
    local = np.column_stack((s-slope*d/norm, lateral(s)+d/norm, height(s)))
    return local @ geometry.basis.T + geometry.origin


def mmss(seconds):
    return f'{int(seconds)//60:02d}:{seconds%60:04.1f}'.replace('.', ',')


def bev_axis(xyz, forward_limit=None):
    """Fit the forward axis to all finite forward returns, not a percentile."""
    points = np.asarray(xyz)
    points = points[np.isfinite(points).all(axis=1)]
    forward = -points[:, 1]
    valid = (forward > 0.) & (np.linalg.norm(points, axis=1) >= 2.5)
    cloud_end = float(np.max(forward[valid], initial=0.))
    requested = cloud_end if forward_limit is None else float(forward_limit)
    if not np.isfinite(requested) or requested < 0.:
        raise ValueError('BEV forward limit must be finite and nonnegative')
    target = max(10., requested)
    rough_step = target / 5.
    magnitude = 10. ** np.floor(np.log10(rough_step))
    step = next(v*magnitude for v in (1., 2., 2.5, 5., 10.) if v*magnitude >= rough_step)
    extent = float(np.ceil(target/step)*step)
    return extent, np.arange(0., extent+step*.5, step), cloud_end


def corridor_stop_station(analysis, status):
    """End the displayed corridor at the near face of a confirmed obstacle.

    Use distance along the route, not the radial LiDAR range. This is only
    a drawing limit: neither the route geometry nor detection is modified.
    """
    geometry = analysis.region.geometry
    if status != 'BLOCKED' or geometry is None:
        return None
    stops = []
    for obstacle in analysis.obstacles:
        if getattr(obstacle,'decision','BLOCKED') != 'BLOCKED':
            continue
        station = getattr(obstacle, 'stop_station_m', None)
        if station is None:
            indices = np.asarray(obstacle.indices)
            local = analysis.region.local
            if local is not None and len(indices) and np.all(indices < len(local)):
                stations = local[indices, 0]
            else:
                points = analysis.region.points[indices]
                stations = ((points - geometry.origin) @ geometry.basis)[:, 0]
            stations = stations[np.isfinite(stations)]
            station = float(stations.min()) if len(stations) else None
        if station is not None and np.isfinite(station):
            stops.append(max(0., float(station)))
    return min(stops) if stops else None


def perspective(xyz, obstacle_mask, width, height):
    """Point splats with a z-buffer; no surfaces, invented points, or corridor."""
    width, height = round(width*S), round(height*S)
    f = width / (2*np.tan(np.deg2rad(80.)/2))
    depth = -xyz[:, 1]
    good = np.isfinite(xyz).all(axis=1) & (depth > 1.8) & (depth < 160.)
    good &= np.linalg.norm(xyz, axis=1) >= 2.5
    ids = np.flatnonzero(good)
    points = xyz[ids]
    z = depth[ids]
    u = width*.5 + f*points[:, 0]/z
    v = height*.48 - f*points[:, 2]/z
    keep = (u >= 2) & (u < width-2) & (v >= 2) & (v < height-2)
    ids, points, z = ids[keep], points[keep], z[keep]
    u, v = np.rint(u[keep]).astype(int), np.rint(v[keep]).astype(int)
    base = np.tile(rgb('#78b4cb'), (len(points), 1))
    floor = points[:, 2] < -.7
    base[floor] = rgb('#aab8c8')
    fade = .38 + .62*np.exp(-z/75.)
    base = rgb('#0a1420') + (base-rgb('#0a1420'))*fade[:, None]
    base[obstacle_mask[ids]] = rgb(RED)
    base = np.clip(base, 0, 255).astype(np.uint8)
    image = np.empty((height, width, 3), np.uint8)
    image[:] = rgb('#0a121d').astype(np.uint8)
    buffer = np.full(width*height, np.inf, dtype=np.float32)
    # 3x3 high-resolution splats become ~1.5 px after antialiasing.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            flat = (v+dy)*width+u+dx
            np.minimum.at(buffer, flat, z)
    flat_image = image.reshape(-1, 3)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            flat = (v+dy)*width+u+dx
            visible = z <= buffer[flat] + .001
            flat_image[flat[visible]] = base[visible]

    def project(points):
        points = np.asarray(points)
        d = -points[:, 1]
        return np.column_stack((width*.5+f*points[:, 0]/d,
                                height*.48-f*points[:, 2]/d))/S

    return Image.fromarray(image), project, len(ids)


def main(payload=None, renderer=None, output=None, *, left_title='Вид с лидара',
         left_subtitle='По направлению движения', camera_metadata=None, bev_forward_limit=None,
         badge_label='obstacle_detector', annotate_perspective=True, save=True, quiet=False,
         algorithm_name='obstacle_detector', status_subtitle=None):
    if payload is None:
        raise ValueError('A frame payload is required')
    xyz, analysis = payload['xyz'], payload['analysis']
    decision = payload['decision']
    status = decision['status_out']
    assert status in {'BLOCKED', 'CAUTION', 'CLEAR', 'UNKNOWN'}
    display_status = 'CAUTION' if status == 'UNKNOWN' else status
    image = Image.new('RGB', (W*S, H*S), BG)
    draw = ImageDraw.Draw(image)
    # Header and recording position.
    rounded(draw, (32, 34, 40, 76), CYAN, radius=3)
    text(draw, (56, 29), 'ОБНАРУЖЕНИЕ ПРЕПЯТСТВИЙ', 30, bold=True)
    text(draw, (57, 71), f"LiDAR  /  {payload.get('recording_label', 'recording')}  /  {algorithm_name}", 19, MUTED)
    rounded(draw, (1660, 32, 1888, 70), '#122333', radius=7, outline='#294559')
    text(draw, (1774, 49), badge_label, 17, CYAN, bold=True, anchor='mm')
    text(draw, (1888, 86), f"Кадр {payload['frame_idx']:04d} / {payload['frames_total']-1:04d}    •    {mmss(payload['time_s'])}",
         20, MUTED, anchor='ra')

    left = (32, 122, 1316, 866)
    right = (1340, 122, 1888, 866)
    for box in (left, right):
        rounded(draw, box, PANEL, outline='#213043')
    text(draw, (54, 139), left_title, 24, bold=True)
    text(draw, (1294, 145), left_subtitle, 18, MUTED, anchor='ra')
    text(draw, (1362, 139), 'BEV · вид сверху', 24, bold=True)

    px, py, pw, ph = 34, 182, 1280, 682
    cloud, project, visible_points = (renderer or perspective)(xyz, analysis.obstacle_mask, pw, ph)
    image.paste(cloud, (px*S, py*S))
    draw = ImageDraw.Draw(image)

    for number, obstacle in enumerate(analysis.obstacles if annotate_perspective else (), 1):
        points = analysis.region.points[obstacle.indices]
        points = points[-points[:, 1] > 1.8]
        if not len(points):
            continue
        projected = project(points)
        low, high = projected.min(axis=0), projected.max(axis=0)
        x0, y0 = np.maximum(low-9, [6, 6]) + [px, py]
        x1, y1 = np.minimum(high+9, [pw-6, ph-6]) + [px, py]
        if x1 <= x0 or y1 <= y0:
            continue
        # A screen-space frame around exactly the returns accepted by the detector.
        rounded(draw, (x0, y0, x1, y1), None, radius=3, outline=RED, width=2)
        corner = min(19, (x1-x0)/3, (y1-y0)/3)
        for a, b, sx, sy in [(x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)]:
            line(draw, [(a+sx*corner,b),(a,b),(a,b+sy*corner)], '#ffb1a8', 3)
        label = f"{number:02d}   •   {obstacle.range_m:.1f} м".replace('.', ',')
        label_width = draw.textlength(label, font=font(19, True))/S+24
        label_y = max(py+6, y0-43)
        label_x = min(max(px+6, x0), px+pw-label_width-6)
        rounded(draw, (label_x,label_y,label_x+label_width,label_y+33), '#432029', radius=5, outline=RED)
        text(draw, (label_x+12,label_y+4), label, 19, '#ffb4ad', bold=True)

    # Top-down view of the exact current corridor model, in sensor X / forward -Y.
    plot = (1398, 212, 1848, 778)
    x_min, x_max = -4., 4.
    y_max, bev_ticks, cloud_end = bev_axis(xyz, bev_forward_limit)

    def bev_x(x):
        # Match the forward-facing camera: positive sensor X is screen-left.
        return plot[0]+(x_max-x)/(x_max-x_min)*(plot[2]-plot[0])

    def bev(points):
        q = np.asarray(points)
        return np.column_stack((bev_x(q[:, 0]),
                                plot[3]+q[:, 1]/y_max*(plot[3]-plot[1])))

    def top_poly(points):
        return [(round(x*S), round(y*S)) for x,y in bev(points)]

    text(draw, (1362, 176), f'Габарит 2,3 м · облако до {cloud_end:.1f} м'.replace('.', ','), 18, MUTED)
    for metre in bev_ticks:
        y = plot[3] - metre/y_max*(plot[3]-plot[1])
        line(draw, [(plot[0],y),(plot[2],y)], '#233044', 1)
        text(draw, (plot[0]-14,y), f'{metre:g}'.replace('.', ','), 16, MUTED, anchor='rm')
    for x in (-4., -2., 0., 2., 4.):
        u = bev_x(x)
        line(draw, [(u,plot[1]),(u,plot[3])], '#192638', 1)
    text(draw, (plot[0]+8, plot[1]+8), 'м', 15, MUTED)

    g = analysis.region.geometry
    model_end = min(g.extent_m, y_max) if g is not None else 0.
    original_model_end = model_end
    stop_station = corridor_stop_station(analysis, status)
    if stop_station is not None:
        model_end = min(model_end, stop_station)
    rails_end = g.rails_end_m if g is not None else None
    reused_geometry = analysis.region.status == 'REUSED_GEOMETRY'
    has_rail_support = (not reused_geometry and rails_end is not None
                        and np.isfinite(rails_end) and rails_end > 0.)
    support_end = float(np.clip(rails_end, 0., model_end)) if has_rail_support else 0.
    # Include the exact support boundary so the two fills meet without a gap.
    s = np.unique(np.r_[np.linspace(0, model_end, 300), support_end])
    centre = as_sensor(g, s, 0.) if g is not None else np.empty((0,3))
    sides = ([as_sensor(g, s, offset) for offset in (-g.half_width_m,g.half_width_m)]
             if g is not None else [])
    # Clip the overlay to the BEV viewport, including any curve at the edges.
    overlay = Image.new('RGBA', image.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    for section, fill in ((s <= support_end, (46,158,173,40)),
                          (s >= support_end, (232,160,70,35))):
        if sides and np.count_nonzero(section) >= 2:
            od.polygon(top_poly(np.vstack((sides[0][section],sides[1][section][::-1]))), fill=fill)
    crop = overlay.crop(tuple(round(v*S) for v in plot))
    image.paste(crop, (plot[0]*S,plot[1]*S), crop)
    draw = ImageDraw.Draw(image)

    selected = np.isfinite(xyz).all(axis=1) & (xyz[:,0]>x_min) & (xyz[:,0]<x_max)
    selected &= (-xyz[:,1]>2.5) & (-xyz[:,1]<y_max) & (xyz[:,2]>-3.) & (xyz[:,2]<2.5)
    ids = np.flatnonzero(selected)
    coords = bev(xyz[ids])
    cloud_layer = np.array(image)
    u, v = np.rint(coords[:,0]*S).astype(int), np.rint(coords[:,1]*S).astype(int)
    cloud_layer[v,u] = np.array([64,85,105],np.uint8)
    stations = np.full(len(xyz), np.nan)
    if analysis.region.local is not None:
        stations[analysis.region.indices] = analysis.region.local[:,0]
    # Returns beyond the stop remain visible, but lose corridor coloring.
    roi = analysis.region.mask[ids] & (stations[ids] <= model_end)
    supported = roi & has_rail_support & (stations[ids] <= support_end)
    predicted = roi & ~supported
    cloud_layer[v[supported],u[supported]] = np.array([75,156,167],np.uint8)
    cloud_layer[v[predicted],u[predicted]] = np.array([176,130,74],np.uint8)
    object_ids = analysis.obstacle_mask[ids]
    warning_ids = getattr(analysis,'warning_mask',np.zeros(len(xyz),bool))[ids]
    cloud_layer[v[warning_ids],u[warning_ids]] = rgb(FORECAST).astype(np.uint8)
    for dy in (-1,0,1):
        cloud_layer[v[object_ids]+dy,u[object_ids]] = rgb(RED).astype(np.uint8)
    image = Image.fromarray(cloud_layer)
    draw = ImageDraw.Draw(image)

    # Cyan solid boundaries use observed rails; orange dashes show the forecast.
    for boundary in sides:
        xy = bev(boundary)
        for i in range(len(xy)-1):
            measured = has_rail_support and s[i+1] <= support_end
            if not measured and i % 8 >= 4:
                continue
            if all(plot[0] <= p[0] <= plot[2] and plot[1] <= p[1] <= plot[3] for p in xy[i:i+2]):
                line(draw, xy[i:i+2], CYAN if measured else FORECAST, 1.8)
    xy = bev(centre)
    for i in range(0, len(xy)-1, 7):
        q = xy[i:min(i+3,len(xy))]
        if all(plot[0] <= p[0] <= plot[2] and plot[1] <= p[1] <= plot[3] for p in q):
            for j in range(len(q)-1):
                measured = has_rail_support and s[i+j+1] <= support_end
                line(draw, q[j:j+2], '#447c8f' if measured else '#997344', 1)

    support_forward_m = None
    if has_rail_support:
        support_centre = as_sensor(g, [rails_end], 0.)
        support_forward_m = float(-support_centre[0,1])
    forecast_label_box=None
    if 0. < support_end < model_end:
        boundary = bev(as_sensor(g, support_end, [-g.half_width_m,g.half_width_m]))
        if all(plot[0] <= p[0] <= plot[2] and plot[1] <= p[1] <= plot[3] for p in boundary):
            line(draw, boundary, FORECAST, 1.5)
            marker_x, marker_y = boundary[np.argmax(boundary[:,0])]
            label_x = min(marker_x+18, plot[2]-135)
            label_y = float(np.clip(marker_y-23, plot[1]+5, plot[3]-49))
            line(draw, [(marker_x,marker_y),(label_x,marker_y)], FORECAST, 1)
            rounded(draw, (label_x,label_y,label_x+135,label_y+45), PANEL, radius=4)
            forecast_label_box=(label_x,label_y,label_x+135,label_y+45)
            text(draw, (label_x+7,label_y+2), 'Начало прогноза', 14, FORECAST)
            text(draw, (label_x+7,label_y+20), f'{support_forward_m:.1f} м'.replace('.',','), 18, FORECAST, True)
    if g is None:
        text(draw, ((plot[0]+plot[2])/2,plot[1]+30), 'Коридор не определён', 18, MUTED, anchor='mm')
    elif reused_geometry:
        text(draw, ((plot[0]+plot[2])/2,plot[1]+30), 'Коридор из предыдущих кадров', 17, FORECAST, anchor='mm')

    free_tag_rows=list(range(plot[1]+12,plot[3]-10,24))
    for number, obj in enumerate(analysis.obstacles, 1):
        color=FORECAST if getattr(obj,'decision','BLOCKED')=='CAUTION' else RED
        q = bev(analysis.region.points[obj.indices])
        lo, hi = q.min(axis=0), q.max(axis=0)
        x0, y0 = np.maximum(lo-[5,6], plot[:2])
        x1, y1 = np.minimum(hi+[5,6], plot[2:])
        if x1 <= x0 or y1 <= y0:
            continue
        if getattr(obj,'predicted',False):
            dashed_box(draw,(x0,y0,x1,y1),color,2.5)
        else:
            rounded(draw,(x0,y0,x1,y1),None,radius=3,outline=color,width=2.5)
        label=str(getattr(obj,'track_id',number))
        label_width=draw.textlength(label,font=font(18,True))/S
        tag_x = min(x1+19,plot[2]-label_width-4)
        object_y = (y0+y1)/2
        available=[row for row in free_tag_rows if forecast_label_box is None or not (
            tag_x < forecast_label_box[2]+4 and tag_x+label_width+4 > forecast_label_box[0] and
            row-12 < forecast_label_box[3]+4 and row+12 > forecast_label_box[1]-4)]
        tag_y = min(available or free_tag_rows,key=lambda row:abs(row-object_y)) if free_tag_rows else object_y
        if tag_y in free_tag_rows:free_tag_rows.remove(tag_y)
        line(draw, [(x1,object_y),(tag_x-5,tag_y)], color, 1)
        text(draw,(tag_x,tag_y),label,18,color,True,'lm')

    lidar_x, lidar_y = bev(np.array([[0.,0.,0.]]))[0]
    draw.polygon([(round(lidar_x*S),round((lidar_y-12)*S)),
                  (round((lidar_x-8)*S),round((lidar_y+5)*S)),
                  (round((lidar_x+8)*S),round((lidar_y+5)*S))], fill=TEXT)
    text(draw,(lidar_x,lidar_y+16),'LiDAR',16,TEXT,anchor='ma')
    for x in (-4,4):
        u=bev_x(x)
        text(draw,(u,plot[3]+15),f'{x:+d} м',15,MUTED,anchor='ma')
    line(draw, [(1362,830),(1390,830)], CYAN, 2)
    text(draw, (1400,818), 'По рельсам', 17, CYAN)
    for x in (1605,1622):
        line(draw, [(x,830),(x+10,830)], FORECAST, 2)
    text(draw, (1642,818), 'Прогноз', 17, FORECAST)
    text(draw,(1362,844),'Поперечный масштаб увеличен',14,MUTED)

    # Playback position on the output video's timeline.
    text(draw,(32,881),mmss(payload['time_s']),17,TEXT)
    text(draw,(1888,881),mmss(payload['duration_s']),17,MUTED,anchor='ra')
    line(draw,[(110,894),(1798,894)],'#223348',3)
    progress=110+(1798-110)*payload['time_s']/payload['duration_s']
    line(draw,[(110,894),(progress,894)],CYAN,3)
    draw.ellipse(((progress-5)*S,889*S,(progress+5)*S,899*S),fill=CYAN)

    palettes = {
        'BLOCKED': ('#26181f','#70404a',RED,'#c5989f','ПРЕПЯТСТВИЕ ЕСТЬ','Обнаружение подтверждено'),
        'CLEAR': ('#102620','#285a4d','#63d6ac','#9fbdaf','ПУТЬ СВОБОДЕН','В анализируемом габарите'),
        'CAUTION': ('#2b2418','#75603a',FORECAST,'#d8bf91','ПРЕДУПРЕЖДЕНИЕ','Пересечение габарита не подтверждено'),
    }
    fill,border,accent,secondary,headline,subtitle = palettes[display_status]
    if status == 'UNKNOWN':
        subtitle = 'Свободный путь не подтверждён'
    subtitle = status_subtitle if status_subtitle is not None else subtitle
    rounded(draw,(32,924,1888,1050),fill,radius=12,outline=border)
    rounded(draw,(32,924,39,1050),accent,radius=3)
    if display_status == 'CLEAR':
        draw.ellipse((61*S,958*S,109*S,1006*S),outline=accent,width=2*S)
        line(draw,[(72,982),(82,992),(100,972)],accent,3)
    else:
        draw.polygon([(85*S,957*S),(62*S,1002*S),(108*S,1002*S)],fill=accent)
        text(draw,(85,985),'!',27,fill,bold=True,anchor='mm')
    text(draw,(135,947),headline,34,TEXT,bold=True)
    text(draw,(137,997),subtitle,22,secondary)
    line(draw,[(1388,943),(1388,1030)],border,1)
    distance_title = 'До объекта' if display_status=='CAUTION' else 'До препятствия'
    if decision.get('distance_estimated',False): distance_title += ' (прогноз)'
    text(draw,(1420,948),distance_title,20,secondary)
    value=decision.get('distance_temporal_input_m')
    distance=float(value) if status in {'BLOCKED','CAUTION'} and value is not None else None
    distance_label=f'{distance:.1f} м'.replace('.',',') if distance is not None else '—'
    text(draw,(1855,1035),distance_label,67,TEXT,bold=True,anchor='rs')

    image=image.resize((W,H),Image.Resampling.LANCZOS)
    output=Path(output) if output is not None else Path.cwd() / f"frame_{payload['frame_idx']:04d}.png"
    if save:
        output.parent.mkdir(parents=True,exist_ok=True)
        image.save(output)
    meta=dict(algorithm_name=algorithm_name,frame_idx=payload['frame_idx'],time_s=payload['time_s'],status_out=decision['status_out'],
              display_status=display_status,
              distance_m=distance,perspective_camera=camera_metadata or dict(origin=[0,0,0],forward=[0,-1,0],up=[0,0,1],hfov_deg=80),
              corridor_in_perspective=False,bev_extent=dict(x_m=[x_min,x_max],forward_m=[0,y_max]),
              bev_cloud_forward_end_m=cloud_end,bev_forward_ticks_m=bev_ticks.tolist(),
              bev_rail_support_station_m=float(rails_end) if has_rail_support else None,
              bev_rail_support_forward_m=support_forward_m,
              bev_corridor_stop_station_m=stop_station,
              bev_corridor_model_end_station_m=original_model_end,
              bev_corridor_display_end_station_m=model_end,
              bev_corridor_clipped_at_obstacle=model_end < original_model_end,
              bev_corridor_colors=dict(rail_supported=CYAN,forecast=FORECAST),
              bev_lateral_scale_enlarged=True,bev_horizontal_mirrored=True,visible_perspective_returns=visible_points,
              output=str(output),resolution=[W,H])
    if not quiet:
        print(json.dumps(meta,ensure_ascii=False))
    return image, meta
