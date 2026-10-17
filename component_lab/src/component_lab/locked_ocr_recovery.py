"""Fallback OCR ownership locks and additive identity recovery; no inference/GT."""
from copy import deepcopy
import hashlib
import json
import math

from .fullpage_identity import FIELDS, parse_readings, apply_readings


def area(box):return max(0,box[2]-box[0])*max(0,box[3]-box[1])
def overlap(a,b):
    hit=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    return hit/max(1e-9,min(area(a),area(b)))


def named_hash(baseline,keys=None):
    selected=keys if keys is not None else [k for k in baseline['components'] if not k.startswith('__unresolved')]
    value={k:baseline['components'][k] for k in selected}
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def lock_tokens(tokens,bindings):
    """Reserve primary spatial instances before fallback assignment.

    Same strings at distinct positions remain distinct. Scores do not override
    primary ownership. A new provider ID cannot bypass a spatial lock.
    """
    kept=[];blocked=[]
    for original in tokens:
        token=deepcopy(original);box=token['bbox']
        if len(box)!=4 or not all(math.isfinite(float(x)) for x in box) or area(box)<=0:raise ValueError('Invalid OCR bbox')
        owners=[]
        for owner in token['owner_ids']:
            conflicts=[b for b in bindings if b['owner']!=owner and overlap(box,b['bbox'])>.6]
            if conflicts:blocked.append({'id':token['id'],'owner':owner,'reason':'primary_spatial_owner_locked'})
            else:owners.append(owner)
        if not owners:continue
        token['owner_ids']=owners;duplicate=None
        for old in kept:
            if token['raw']==old['raw'] and overlap(box,old['bbox'])>.5:
                duplicate=old;break
        if duplicate is None:kept.append(token)
        else:
            duplicate['owner_ids']=sorted(set(duplicate['owner_ids'])|set(owners))
            duplicate.setdefault('duplicate_observations',[]).append(token['id'])
    return kept,blocked


def remove_anchor_duplicates(tokens,anchors):
    """One spatial reference instance must not create two assignment slots."""
    kept=[];blocked=[]
    for original in tokens:
        token=deepcopy(original)
        for anchor in anchors:
            if token['raw']==anchor['raw'] and overlap(token['bbox'],anchor['bbox'])>.5:
                removed=set(token['owner_ids'])&set(anchor['owner_ids'])
                if removed:
                    token['owner_ids']=[x for x in token['owner_ids'] if x not in removed]
                    blocked.append({'id':token['id'],'reason':'same_primary_reference_instance_already_anchored'})
        if token['owner_ids']:kept.append(token)
    return kept,blocked


def merge_locked(baseline,targets,proposals,tokens,mode='safe_unresolved'):
    if mode not in {'safe_unresolved','empty_fields_shadow'}:raise ValueError('Unknown merge mode')
    byid={t['id']:t for t in targets};ocr={t['id']:t for t in tokens}
    if len(byid)!=len(targets) or len(ocr)!=len(tokens):raise ValueError('Duplicate IDs')
    if len({p['id'] for p in proposals})!=len(proposals):raise ValueError('Duplicate proposals')
    items={t['id']:{'id':t['id'],**{f:None for f in FIELDS},'ocr_ids':{f:None for f in FIELDS},'uncertain_fields':list(FIELDS)} for t in targets}
    blocked=[]
    for proposal in proposals:
        ident=proposal['id']
        if ident not in byid:raise ValueError('Unknown target')
        target=byid[ident];old=target['baseline_record'];unresolved=target['baseline_key'].startswith('__unresolved')
        for field,reading in proposal['fields'].items():
            if field not in FIELDS:raise ValueError('Unknown field')
            reason=None;token=ocr.get(reading.get('ocr_id'))
            if mode=='safe_unresolved' and not unresolved:reason='named_record_locked'
            elif field=='reference' and not unresolved:reason='existing_reference_locked'
            elif field!='reference' and old.get(field) is not None:reason='existing_nonempty_field_locked'
            elif token is None or token.get('source_tier')!='fallback_v6':reason='fallback_evidence_required'
            elif ident not in token.get('owner_ids',[]):reason='outside_original_roi_owner'
            elif reading.get('text')!=token['raw']:reason='raw_literal_changed'
            if reason:blocked.append({'id':ident,'field':field,'reason':reason});continue
            items[ident][field]=token['raw'];items[ident]['ocr_ids'][field]=token['id'];items[ident]['uncertain_fields'].remove(field)
    readings,rejected=parse_readings(json.dumps({'items':list(items.values())},ensure_ascii=False),targets,tokens)
    out,audit=apply_readings(baseline,targets,readings,'fill_empty')
    old_named=[k for k in baseline['components'] if not k.startswith('__unresolved')]
    if mode=='safe_unresolved':assert named_hash(out,old_named)==named_hash(baseline,old_named)
    for t in targets:
        old=t['baseline_record']
        if not t['baseline_key'].startswith('__unresolved'):
            new=out['components'][t['baseline_key']]
            assert new['bbox']==old['bbox'] and new['type']==old['type']
            for field in ['Name','value']:
                if old[field] is not None:assert new[field]==old[field]
    return out,{'merge':audit,'blocked':blocked+rejected,'named_hash_before':named_hash(baseline,old_named),'named_hash_after':named_hash(out,old_named)}
