"""Experimental box display-name slot, separate from part-number recognition."""
from copy import deepcopy
import math
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

from .contest_components import TYPE_MAP, raw_field


def _center(box):return ((box[0]+box[2])/2,(box[1]+box[3])/2)
def _gap(a,b):return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))


def assign_display_names(association):
    """Fill a dedicated Name candidate from local unclaimed description text.

    Only box-like objects without a part_number are eligible. No reference is
    copied as Name; no source text is normalized or rewritten. This is an
    experimental geometry rule, not a verified label or an API decision.
    """
    out=deepcopy(association)
    components=out['components'];texts=out['texts']
    used={e['text_id'] for row in components for e in row.get('association_evidence',{}).values() if e.get('text_id')}
    tokens=[]
    for text in texts:
        raw=text.get('raw',text.get('text',''))
        if text['id'] in used or not 3<=len(raw)<=96 or not re.search(r'[A-Za-z\u3400-\u9fff]',raw):continue
        if '\t' in raw or '\n' in raw:continue
        roles={c['role'] for c in text.get('role_candidates',[])}
        if roles!={'description'}:continue
        tokens.append(text)
    eligible=[row for row in components if TYPE_MAP.get(row.get('type'),row.get('type'))=='box'
              and raw_field(row,'part_number') is None and raw_field(row,'reference') is not None]
    if not eligible or not tokens:
        out['display_name_diagnostics']={'assigned':0,'eligible_boxes':len(eligible),'description_candidates':len(tokens)}
        return out
    costs=np.full((len(eligible),len(tokens)+len(eligible)),1.75)
    details={}
    byid={t['id']:t for t in texts}
    for i,row in enumerate(eligible):
        box=row['bbox'];cx,cy=_center(box)
        ref=byid.get(row.get('association_evidence',{}).get('reference',{}).get('text_id'))
        for j,text in enumerate(tokens):
            b=text['bbox'];tx,ty=_center(b);h=max(3,min(b[2]-b[0],b[3]-b[1]))
            gap=_gap(box,b)/h
            if gap>3:continue
            inside_header=(box[0]+.2*(box[2]-box[0])<=tx<=box[2]-.2*(box[2]-box[0])
                           and box[1]<=ty<=box[1]+.3*(box[3]-box[1]))
            near_ref=False;refgap=None
            if ref:
                rb=ref['bbox'];rx,ry=_center(rb);refgap=_gap(rb,b)/h
                aligned=(abs(tx-rx)<=max(b[2]-b[0],rb[2]-rb[0],box[2]-box[0])*.5 or abs(ty-ry)<=1.5*h)
                near_ref=refgap<=2.5 and aligned
            if not (inside_header or near_ref):continue
            cost=.25*gap+.02*math.hypot(tx-cx,ty-cy)/h+(0 if near_ref else .2)
            if cost>=1.75:continue
            costs[i,j]=cost
            details[(i,j)]={'gap_text_heights':gap,'near_reference':near_ref,'inside_header':inside_header,
                            'reference_gap_text_heights':refgap,'cost':cost}
        ranked=sorted(costs[i,:len(tokens)])
        if len(ranked)>1 and ranked[1]<1.75 and ranked[1]-ranked[0]<.1:
            costs[i,:len(tokens)]=2.75
    assigned=0
    left,right=linear_sum_assignment(costs)
    for i,j in zip(left,right):
        if j>=len(tokens) or costs[i,j]>=1.75:continue
        row,text=eligible[i],tokens[j];raw=text.get('raw',text.get('text',''))
        evidence={'text_id':text['id'],'raw':raw,'role':'display_name',
                  'reason':'reference_anchored_description_literal','uses_ground_truth':False,**details[(i,j)]}
        row.setdefault('association_evidence',{})['display_name']=evidence
        row['display_name']=raw;assigned+=1
    out['display_name_diagnostics']={'assigned':assigned,'eligible_boxes':len(eligible),'description_candidates':len(tokens),
                                     'experimental':True,'uses_ground_truth':False}
    return out
