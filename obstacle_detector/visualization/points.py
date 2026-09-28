"""Unchanged point projection and boxes from render_b_details.py."""
import numpy as np
from PIL import Image, ImageDraw
from . import layout as ui

WIDTH, HEIGHT = 1280, 682
S = ui.S
STYLE = dict(id='B1', title='Тонкие точки', near_px=1.25, far_px=1.25,
             contrast='natural', voxel_m=0.)
CAMERA = dict(id='2', title='Вдоль пути · наклон 4,9°', eye=[0.,9.,1.8],
              target=[0.,-20.,-.7], fov=50)


def render(xyz, obstacle_mask, style, camera=None, warning_mask=None):
    camera = CAMERA if camera is None else camera
    eye = np.array(camera['eye'],float)
    forward = np.array(camera['target'],float)-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0.,0.,1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward,right)
    rotation = np.stack([right,down,forward])
    w,h = WIDTH*S,HEIGHT*S
    focal = w/(2*np.tan(np.deg2rad(camera['fov'])/2))

    def project(points):
        q=(np.asarray(points)-eye)@rotation.T
        return np.column_stack((w*.5+focal*q[:,0]/q[:,2],h*.5+focal*q[:,1]/q[:,2]))/S

    good=np.isfinite(xyz).all(axis=1)
    good &= (np.linalg.norm(xyz,axis=1)>=2.5)&(-xyz[:,1]>.1)&(-xyz[:,1]<160)
    ids=np.flatnonzero(good)
    count_before=len(ids)
    if style['voxel_m']:
        near=(-xyz[ids,1]<25.)&(~obstacle_mask[ids])
        candidates=ids[near]
        cells=np.floor(xyz[candidates]/style['voxel_m']).astype(np.int32)
        _,representatives=np.unique(cells,axis=0,return_index=True)
        ids=np.sort(np.r_[ids[~near],candidates[representatives]])
    p=xyz[ids]
    q=(p-eye)@rotation.T
    keep=q[:,2]>.1
    p,q,ids=p[keep],q[keep],ids[keep]
    u=np.rint(w*.5+focal*q[:,0]/q[:,2]).astype(int)
    v=np.rint(h*.5+focal*q[:,1]/q[:,2]).astype(int)
    keep=(u>4)&(u<w-5)&(v>4)&(v<h-5)
    u,v,p,q,ids=u[keep],v[keep],p[keep],q[keep],ids[keep]
    distance=np.maximum(-p[:,1],0)
    far=np.clip((distance-12.)/45.,0,1)
    radius=(style['near_px']+(style['far_px']-style['near_px'])*far)*S/2
    # Keep object returns small as well, so they do not turn into an opaque tile.
    radius[obstacle_mask[ids]]=1.25*S/2
    high=np.clip((p[:,2]+.8)/3.5,0,1)
    base=ui.rgb('#becad7')[None,:]*(1-high[:,None])+ui.rgb('#739eb5')[None,:]*high[:,None]
    if style['contrast']=='distance':
        # Distant floor/wall samples become brighter, near dense samples softer.
        strength=1.0+.65*far
        base=base*strength[:,None]
    else:
        strength=.70+.55*np.exp(-distance/100.)
        base=base*strength[:,None]
    if warning_mask is not None:
        base[warning_mask[ids]]=ui.rgb(ui.FORECAST)
    base[obstacle_mask[ids]]=ui.rgb(ui.RED)
    base=np.clip(base,0,255)
    background=ui.rgb('#0a121d')
    arr=np.empty((h,w,3),np.uint8)
    arr[:]=background.astype(np.uint8)
    depth=q[:,2].astype(np.float32)
    zbuffer=np.full(w*h,np.inf,np.float32)
    shifts=[]
    limit=int(np.ceil(radius.max(initial=0.)+.5))
    for dy in range(-limit,limit+1):
        for dx in range(-limit,limit+1):
            coverage=np.clip(radius+.5-np.hypot(dx,dy),0,1)
            active=coverage>.02
            if not active.any():
                continue
            flat=(v[active]+dy)*w+u[active]+dx
            np.minimum.at(zbuffer,flat,depth[active])
            shifts.append((dx,dy,active,coverage))
    rgb=arr.reshape(-1,3)
    for dx,dy,active,coverage in shifts:
        indices=np.flatnonzero(active)
        flat=(v[indices]+dy)*w+u[indices]+dx
        visible=depth[indices]<=zbuffer[flat]+.001
        ids_visible=indices[visible]
        col=background+(base[ids_visible]-background)*coverage[ids_visible,None]
        rgb[flat[visible]]=np.clip(col,0,255).astype(np.uint8)
    meta=dict(input_visible_returns=count_before,drawn_returns=len(ids),
              near_point_diameter_px=style['near_px'],far_point_diameter_px=style['far_px'],
              near_voxel_m=style['voxel_m'],all_obstacle_returns_preserved=True)
    return Image.fromarray(arr),project,meta


def annotate(image,project,result):
    draw=ImageDraw.Draw(image)
    labels=[]
    for number,obj in enumerate(result.obstacles,1):
        warning=getattr(obj,'decision','BLOCKED')=='CAUTION'
        color=ui.FORECAST if warning else ui.RED
        points=project(result.region.points[obj.indices])
        points=points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        lo,hi=points.min(axis=0),points.max(axis=0)
        x0,y0=np.maximum(lo-8,[6,6])
        x1,y1=np.minimum(hi+8,[WIDTH-6,HEIGHT-6])
        if x1 <= x0 or y1 <= y0:
            continue
        if getattr(obj,'predicted',False):
            ui.dashed_box(draw,(x0,y0,x1,y1),color,1.5)
        else:
            ui.rounded(draw,(x0,y0,x1,y1),None,radius=3,outline=color,width=1.5)
        # Move label beside the object: the old label hid the distant track.
        label=f'{getattr(obj,"track_id",number)}   •   {obj.range_m:.1f} м'.replace('.',',')
        if getattr(obj,'predicted',False):
            label += ' · прогноз'
        label_width=draw.textlength(label,font=ui.font(19,True))/S+24
        lx=min(x1+48,WIDTH-label_width-12)
        ly=min(max(12,y0+3),HEIGHT-45)
        # Keep every object's ID and distance readable when projections are close.
        candidate_rows=sorted(range(12,HEIGHT-44,39),key=lambda row:abs(row-ly))
        for row in candidate_rows:
            if not 12 <= row <= HEIGHT-45:
                continue
            box=(lx,row,lx+label_width,row+33)
            if not any(box[0] < b[2]+4 and box[2]+4 > b[0] and
                       box[1] < b[3]+4 and box[3]+4 > b[1] for b in labels):
                ly=row
                break
        labels.append((lx,ly,lx+label_width,ly+33))
        ui.line(draw,[(x1,(y0+y1)/2),(lx-10,ly+17),(lx,ly+17)],color,1.)
        ui.rounded(draw,(lx,ly,lx+label_width,ly+33),'#403325' if warning else '#432029',radius=5,outline=color)
        ui.text(draw,(lx+12,ly+4),label,19,color,True)
    return image

