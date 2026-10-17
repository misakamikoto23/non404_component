"""Experimental reference-local empty-value fill, with immutable ownership."""
from collections import Counter
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contest_components import TYPE_MAP,raw_field


def _center(box):return ((box[0]+box[2])/2,(box[1]+box[3])/2)


def _angle(row):return float(row.get('orientation',0))%180


def _bounds(box,angle):
    theta=math.radians(angle);u=np.array([math.cos(theta),math.sin(theta)]);v=np.array([-u[1],u[0]])
    x1,y1,x2,y2=box;points=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    a,b=points@u,points@v
    return [float(a.min()),float(b.min()),float(a.max()),float(b.max())]


def pair_cost(reference,value,all_references):
    diff=abs(_angle(reference)-_angle(value));diff=min(diff,180-diff)
    if diff>15:return None
    a,b=_bounds(reference['bbox'],_angle(reference)),_bounds(value['bbox'],_angle(reference))
    h=max(3,a[3]-a[1],b[3]-b[1]);ac,bc=_center(a),_center(b)
    inline=min(abs(a[0]-b[0]),abs(ac[0]-bc[0]))/h
    cross=min(abs(a[1]-b[1]),abs(ac[1]-bc[1]))/h
    vertical_gap=max(0,a[1]-b[3],b[1]-a[3])/h
    horizontal_gap=max(0,a[0]-b[2],b[0]-a[2])/h
    choices=[]
    if inline<=.5 and vertical_gap<=1.5:choices.append((vertical_gap+.5*inline,'stacked'))
    if cross<=.5 and horizontal_gap<=1.5:choices.append((horizontal_gap+.5*cross,'same_row'))
    if not choices:return None
    start=np.array(_center(reference['bbox']));end=np.array(_center(value['bbox']));vector=end-start
    denom=float(vector@vector)
    for other in all_references:
        if other['id']==reference['id'] or denom<=0:continue
        pos=np.array(_center(other['bbox']));t=float((pos-start)@vector/denom)
        if .1<t<.9 and float(np.linalg.norm(pos-(start+t*vector)))<=.5*h:return None
    cost,layout=min(choices)
    return {'cost':cost,'layout':layout,'orientation_difference':diff,'text_height':h}


def _span_overlap(a,b):
    sa,sb=a.get('span_provenance'),b.get('span_provenance')
    if sa and sb and sa['parent_id']==sb['parent_id']:
        return max(sa['start'],sb['start'])<min(sa['end'],sb['end'])
    return False


def fill_local_values(association):
    out=deepcopy(association);comps=out['components'];texts=out['texts'];byid={x['id']:x for x in texts}
    used={e['text_id'] for c in comps for e in c.get('association_evidence',{}).values() if e.get('text_id')}
    occupied=[byid[i] for i in used if i in byid]
    references=[];owners={};refcounts=Counter(raw_field(c,'reference') for c in comps)
    targets=[]
    for c in comps:
        if TYPE_MAP.get(c.get('type'),c.get('type')) not in {'r','c'}:continue
        raw=raw_field(c,'reference');rid=c.get('association_evidence',{}).get('reference',{}).get('text_id')
        if not raw or refcounts[raw]!=1 or rid not in byid:continue
        references.append(byid[rid]);owners[rid]=c
        if raw_field(c,'value') is None:targets.append(c)
    candidates=[]
    for t in texts:
        if t['id'] in used or any(_span_overlap(t,x) for x in occupied):continue
        roles=[x for x in t.get('role_candidates',[]) if x['role']=='value' and not x.get('requires_review')]
        if roles:candidates.append(t)
    if not targets or not candidates:
        out['local_value_diagnostics']={'filled':0,'targets':len(targets),'candidates':len(candidates)};return out
    cost=np.full((len(targets),len(candidates)+len(targets)),1.75);evidence={}
    eligible={c['id']:i for i,c in enumerate(targets)}
    for j,t in enumerate(candidates):
        ranked=[]
        for r in references:
            item=pair_cost(r,t,references)
            if item and item['cost']<1.75:ranked.append((item['cost'],r,item))
        ranked.sort(key=lambda x:(x[0],x[1]['id']))
        if not ranked or len(ranked)>1 and ranked[1][0]-ranked[0][0]<.15:continue
        _,r,item=ranked[0];owner=owners[r['id']]
        if owner['id'] not in eligible:continue
        i=eligible[owner['id']];cost[i,j]=item['cost'];evidence[(i,j)]=item
    for i in range(len(targets)):
        ranked=sorted(cost[i,:len(candidates)])
        if len(ranked)>1 and ranked[1]<1.75 and ranked[1]-ranked[0]<.15:cost[i,:len(candidates)]=2.75
    left,right=linear_sum_assignment(cost);filled=0
    for i,j in zip(left,right):
        if j>=len(candidates) or cost[i,j]>=1.75:continue
        c,t=targets[i],candidates[j];raw=t.get('raw',t.get('text',''))
        c['value']=raw;c.setdefault('association_evidence',{})['value']={
            'text_id':t['id'],'raw':raw,'role':'value','reason':'mutual_reference_local_empty_slot',
            'uses_ground_truth':False,**evidence[(i,j)]}
        filled+=1
    out['local_value_diagnostics']={'filled':filled,'targets':len(targets),'candidates':len(candidates),
                                    'experimental':True,'overwrites_existing_values':False,'uses_ground_truth':False}
    return out
