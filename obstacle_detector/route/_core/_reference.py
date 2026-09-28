"""Rail-reference utilities used by the route estimator."""
import json
from pathlib import Path
import numpy as np

def unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x)

def frame_basis(forward, up):
    up = unit(up)
    forward = unit(np.asarray(forward)-up*np.dot(forward, up))
    return np.column_stack((forward, unit(np.cross(up, forward)), up))

def pairs_from_peaks(peaks, gauge, tolerance, cfg):
    pairs=[]
    for i,left in enumerate(peaks):
        for right in peaks[i+1:]:
            width=float(right-left)
            if cfg['gauge_limits'][0]<=width<=cfg['gauge_limits'][1]:
                if gauge is None or abs(width-gauge)<=tolerance:
                    pairs.append((float((left+right)/2),width,float(left),float(right)))
    return pairs

def default_config():
    return json.loads((Path(__file__).with_name("rail_config.json")).read_text())
