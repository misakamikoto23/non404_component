"""Shadow pin-tip proposals from visible color transitions; no pin identities."""
from dataclasses import dataclass, asdict
from collections import Counter
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class TransitionConfig:
    max_steps: int = 96
    seed_depth: int = 6
    min_seed_support: int = 5
    max_seed_width: int = 9
    min_plateau: int = 4
    chroma_delta: float = 20.
    plateau_spread: float = 12.
    max_lateral_drift: int = 2


def _xy(side,box,p,k):
    x1,y1,x2,y2=box
    if side=='left':return (x1-k,p)
    if side=='right':return (x2+k,p)
    if side=='top':return (p,y1-k)
    return (p,y2+k)


def _runs(values):
    result=[]
    for v in sorted(values):
        if not result or v>result[-1][-1]+1:result.append([v])
        else:result[-1].append(v)
    return result


def _valid(image,x,y):return 0<=y<image.shape[0] and 0<=x<image.shape[1]


def _seeds(foreground,box,side,cfg):
    x1,y1,x2,y2=box
    positions=range(y1+1,y2) if side in {'left','right'} else range(x1+1,x2)
    good=[]
    for p in positions:
        support=0;first=False
        for k in range(1,cfg.seed_depth+1):
            choices=[]
            for delta in [-1,0,1]:
                x,y=_xy(side,box,p+delta,k)
                choices.append(_valid(foreground,x,y) and bool(foreground[y,x]))
            hit=any(choices)
            if k==1:first=hit
            support+=hit
        if first and support>=cfg.min_seed_support:good.append(p)
    return [r for r in _runs(good) if len(r)<=cfg.max_seed_width]


def _cross_width(mask,x,y,side):
    # Width in the direction tangent to the component side.
    total=1
    for sign in [-1,1]:
        for d in range(1,7):
            nx,ny=(x,y+sign*d) if side in {'left','right'} else (x+sign*d,y)
            if not _valid(mask,nx,ny) or not mask[ny,nx]:break
            total+=1
    return total


def trace_corridor(foreground,lab,box,side,p,cfg):
    points=[];colors=[];perp=p;initial_width=None;gap_used=False;reason='search_limit'
    for k in range(1,cfg.max_steps+1):
        x,y=_xy(side,box,perp,k)
        if not _valid(foreground,x,y):reason='image_boundary';break
        options=[]
        for delta in [0,-1,1]:
            if abs(perp+delta-p)>cfg.max_lateral_drift:continue
            nx,ny=_xy(side,box,perp+delta,k)
            if _valid(foreground,nx,ny) and foreground[ny,nx]:options.append((delta,nx,ny))
        if not options:
            nx,ny=_xy(side,box,perp,k+1)
            if not gap_used and _valid(foreground,nx,ny) and foreground[ny,nx]:
                gap_used=True;continue
            reason='foreground_end';break
        straight=next((o for o in options if o[0]==0),None)
        if straight is None and len(options)>1:reason='multiple_paths';break
        delta,x,y=straight or options[0];perp+=delta
        width=_cross_width(foreground,x,y,side)
        if initial_width is None:initial_width=width
        if width>=max(7,initial_width+4):reason='branch_or_wide_stroke';break
        points.append([float(x),float(y)]);colors.append(lab[y,x].astype(float).tolist())
    return points,colors,reason


def stable_color_change(colors,cfg=TransitionConfig()):
    array=np.asarray(colors,dtype=float)
    n=cfg.min_plateau
    if len(array)<2*n:return None,'insufficient_trace'
    initial=np.median(array[:n,1:3],axis=0)
    if np.max(np.linalg.norm(array[:n,1:3]-initial,axis=1))>cfg.plateau_spread:
        return None,'unstable_initial_color'
    hits=[]
    for i in range(n,len(array)-n+1):
        before=array[i-n:i,1:3];after=array[i:i+n,1:3]
        pre,post=np.median(before,axis=0),np.median(after,axis=0)
        if np.linalg.norm(pre-initial)>cfg.plateau_spread:continue
        if np.max(np.linalg.norm(before-pre,axis=1))>cfg.plateau_spread:continue
        if np.max(np.linalg.norm(after-post,axis=1))>cfg.plateau_spread:continue
        if np.linalg.norm(post-pre)>=cfg.chroma_delta:hits.append(i)
    if not hits:return None,'no_observable_color_boundary'
    first=_runs(hits)[0]
    if first[-1]-first[0]>4:return None,'wide_color_transition'
    return int(round(float(np.median(first)))),'stable_chromatic_transition'


def infer_color_transitions(image_bgr,boxes,cfg=TransitionConfig()):
    """Only image pixels, anonymous owner IDs and boxes enter inference.

    No hardcoded red/green palette. In achromatic continuous strokes the semantic
    pin/wire boundary is unobserved and is explicitly left unresolved.
    """
    if image_bgr.ndim!=3 or image_bgr.shape[2]!=3:raise ValueError('BGR image required')
    # Prefer line cores over pale page/symbol fill. Palette independent, but
    # deliberately conservative on very light strokes.
    foreground=(np.min(image_bgr,axis=2)<220)&(np.mean(image_bgr,axis=2)<210)
    lab=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2LAB)
    accepted=[];unresolved=[]
    h,w=image_bgr.shape[:2]
    for owner in boxes:
        owner_seed_count=0
        x1,y1,x2,y2=map(lambda v:int(round(v)),owner['bbox'])
        box=[max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        for side in ['left','right','top','bottom']:
            for run in _seeds(foreground,box,side,cfg):
                owner_seed_count+=1
                p=int(round(float(np.median(run))))
                points,colors,stop=trace_corridor(foreground,lab,box,side,p,cfg)
                index,reason=stable_color_change(colors,cfg)
                row={'owner_id':owner['id'],'side':side,'seed_position':p,'seed_width':len(run),
                     'trace_stop':stop,'trace_length':len(points)}
                if index is None:
                    unresolved.append({**row,'reason':reason});continue
                a,b=points[index-1],points[index]
                if math.dist(a,b)>1.5:
                    unresolved.append({**row,'reason':'gap_at_color_boundary'});continue
                point=[(a[0]+b[0])/2,(a[1]+b[1])/2]
                row.update(candidate_id=f'color_tip_{len(accepted):05d}',point={'x':point[0],'y':h-point[1]},
                           point_top_left=point,evidence=reason,trace=points,
                           transition_interval_top_left=[a,b],gt_pin_used=False)
                accepted.append(row)
        if owner_seed_count==0:
            unresolved.append({'owner_id':owner['id'],'reason':'no_boundary_seed'})
    return {'candidates':accepted,'unresolved':unresolved,'config':asdict(cfg),
            'reason_counts':dict(Counter(x['reason'] for x in unresolved)),
            'coordinate_system':'bottom-left-original-pixels','gt_used_for_inference':False}
