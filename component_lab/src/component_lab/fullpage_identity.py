"""Full-page text ownership with immutable component geometry and raw types.

Machine readings are evidence, never human labels. This module has no network,
detector, target-annotation or evaluator dependency.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json

FIELDS=('reference','Name','value')
MODES=('full_replace','keys_only','fill_empty','unresolved_keys')


def _overlap(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    smallest=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
    return area/max(smallest,1e-9)>.7


def parse_readings(content,targets,ocr):
    """Validate page IDs strictly, quarantine invalid/conflicting text fields."""
    value=content.strip()
    if value.startswith('```'):
        lines=value.splitlines()
        if lines[-1].strip()!='```':raise ValueError('Unclosed JSON fence')
        value='\n'.join(lines[1:-1])
    data=json.loads(value)
    if not isinstance(data,dict) or set(data)!={'items'} or not isinstance(data['items'],list):
        raise ValueError('Invalid page schema')
    allowed={t['id'] for t in targets};catalog={r['id']:r for r in ocr};seen=set();rows=[];rejected=[]
    for raw in data['items']:
        if not isinstance(raw,dict) or set(raw)!={'id',*FIELDS,'ocr_ids','uncertain_fields'}:
            raise ValueError('Invalid item fields; geometry/type output forbidden')
        ident=raw['id']
        if not isinstance(ident,str) or ident not in allowed or ident in seen:raise ValueError('Unknown/duplicate component ID')
        seen.add(ident)
        uncertain=raw['uncertain_fields'];sources=raw['ocr_ids']
        if not isinstance(uncertain,list) or any(f not in FIELDS for f in uncertain) or len(set(uncertain))!=len(uncertain):
            raise ValueError('Invalid uncertainty fields')
        if not isinstance(sources,dict) or set(sources)!=set(FIELDS):raise ValueError('Invalid OCR source schema')
        item={'id':ident,'fields':{}}
        for field in FIELDS:
            text=raw[field];oid=sources[field];reason=None
            if text is not None and (not isinstance(text,str) or not text.strip() or len(text)>160 or any(ord(c)<32 for c in text)):
                reason='invalid_literal'
            elif oid is not None and (not isinstance(oid,str) or oid not in catalog):reason='unknown_ocr_id'
            elif field in uncertain and (text is not None or oid is not None):reason='uncertain_must_be_null'
            elif oid is not None and text!=catalog[oid]['raw']:reason='ocr_literal_mismatch'
            elif field=='reference' and text is not None and (text.startswith(('∅','__','unresolved_')) or any(c in text for c in '\n\r\t')):
                reason='hidden_or_reserved_reference'
            if reason:
                rejected.append({'id':ident,'field':field,'reason':reason})
                item['fields'][field]={'status':'uncertain','text':None,'ocr_id':None,'source':reason}
            else:
                status='uncertain' if field in uncertain else 'absent' if text is None else 'read'
                item['fields'][field]={'status':status,'text':text,'ocr_id':oid,
                    'source':'ocr_raw' if oid else 'image_literal_unverified' if text is not None else status}
        rows.append(item)
    if seen!=allowed:raise ValueError('Page omitted component IDs')
    # Global ownership conflict rejection is simultaneous, independent of row order.
    claims=[(r,f,v) for r in rows for f,v in r['fields'].items() if v['status']=='read' and v['ocr_id']]
    blocked=set()
    for i,(left,lf,lv) in enumerate(claims):
        for right,rf,rv in claims[i+1:]:
            if left['id']==right['id']:continue
            if lv['ocr_id']==rv['ocr_id'] or _overlap(catalog[lv['ocr_id']]['bbox'],catalog[rv['ocr_id']]['bbox']):
                blocked.update([(left['id'],lf),(right['id'],rf)])
    refs=defaultdict(list)
    for row in rows:
        f=row['fields']['reference']
        if f['status']=='read':refs[f['text']].append(row['id'])
    for ids in refs.values():
        if len(ids)>1:blocked.update((ident,'reference') for ident in ids)
    for row in rows:
        for field in FIELDS:
            if (row['id'],field) in blocked:
                row['fields'][field]={'status':'uncertain','text':None,'ocr_id':None,'source':'global_ownership_conflict'}
                rejected.append({'id':row['id'],'field':field,'reason':'global_ownership_conflict'})
    return rows,rejected


def apply_readings(baseline,targets,readings,mode):
    """Resolve renames simultaneously; never drop objects or change bbox/type."""
    if mode not in MODES:raise ValueError('Unknown merge mode')
    byid={r['id']:r for r in readings};staged=[];audit=[]
    assert len({t['id'] for t in targets})==len(targets)==len(baseline['components'])
    assert {t['baseline_key'] for t in targets}==set(baseline['components'])
    for target in targets:
        oldkey=target['baseline_key'];old=baseline['components'][oldkey]
        assert old==target['baseline_record'],'Frozen baseline changed'
        record=deepcopy(old);key=oldkey;reading=byid.get(target['id']);changes=[]
        unresolved=oldkey.startswith('__unresolved')
        if reading:
            for field in FIELDS:
                proposal=reading['fields'][field]
                if proposal['status']=='uncertain':continue
                if field=='reference':
                    permitted=mode in ('full_replace','keys_only') or unresolved
                    if permitted and (mode not in ('fill_empty','unresolved_keys') or proposal['status']=='read'):
                        key=proposal['text'] if proposal['text'] is not None else (oldkey if unresolved else '__unresolved_fullpage_'+target['id'])
                        if key!=oldkey:changes.append(field)
                elif mode=='full_replace' or (mode=='fill_empty' and old[field] is None and proposal['status']=='read'):
                    record[field]=proposal['text']
                    if record[field]!=old[field]:changes.append(field)
        staged.append({'id':target['id'],'key':key,'record':record,'baseline_key':oldkey,'changed_fields':changes})
    # Cascades matter: reverting B->C can invalidate A->B. Stable swaps are valid.
    for _ in range(len(staged)+1):
        counts=Counter(r['key'] for r in staged);duplicates={k for k,v in counts.items() if v>1}
        if not duplicates:break
        changed=False
        for row in staged:
            if row['key'] in duplicates and row['key']!=row['baseline_key']:
                audit.append({'id':row['id'],'field':'reference','reason':'rename_collision_reverted','attempted_key':row['key']})
                row['key']=row['baseline_key'];row['changed_fields']=[f for f in row['changed_fields'] if f!='reference'];changed=True
        if not changed:raise AssertionError('Non-resolvable duplicate baseline keys')
    out=deepcopy(baseline);out['components']={}
    for row in staged:
        assert row['key'] not in out['components']
        old=baseline['components'][row['baseline_key']]
        assert row['record']['bbox']==old['bbox'] and row['record']['type']==old['type']
        out['components'][row['key']]=row['record']
    assert len(out['components'])==len(baseline['components'])
    assert out['pins']==baseline['pins'] and out['nets']==baseline['nets']
    return out,{'objects':staged,'blocked':audit,'changed_objects':sum(bool(r['changed_fields']) for r in staged)}
