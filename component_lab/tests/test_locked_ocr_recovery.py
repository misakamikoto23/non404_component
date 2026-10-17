from copy import deepcopy
from component_lab.locked_ocr_recovery import lock_tokens,merge_locked,named_hash,remove_anchor_duplicates


def fixture():
    baseline={'components':{'R1':{'type':'r','bbox':[0,0,10,10],'Name':None,'value':None},
                            '__unresolved_1':{'type':'r','bbox':[20,0,30,10],'Name':None,'value':' 10kΩ '}},'pins':{},'nets':{}}
    targets=[{'id':str(i),'baseline_key':k,'baseline_record':deepcopy(v)} for i,(k,v) in enumerate(baseline['components'].items())]
    return baseline,targets


def token(ident,raw,owner,box=(20,20,30,30)):
    return {'id':ident,'raw':raw,'bbox':list(box),'owner_ids':[owner],'source_tier':'fallback_v6'}


def test_named_records_locked_but_unresolved_reference_can_recover():
    base,targets=fixture();tokens=[token('a','22K','0'),token('b','R2','1')]
    props=[{'id':'0','fields':{'value':{'text':'22K','ocr_id':'a'}}},
           {'id':'1','fields':{'reference':{'text':'R2','ocr_id':'b'}}}]
    out,_=merge_locked(base,targets,props,tokens)
    assert named_hash(out,['R1'])==named_hash(base) and out['components']['R2']['value']==' 10kΩ '
    assert base['components']['R1']['value'] is None and '__unresolved_1' in base['components']


def test_shadow_fills_only_none_never_rewrites_existing_raw():
    base,targets=fixture();tokens=[token('a','22K','0'),token('b','10k','1',(40,20,50,30))]
    props=[{'id':'0','fields':{'value':{'text':'22K','ocr_id':'a'}}},
           {'id':'1','fields':{'value':{'text':'10k','ocr_id':'b'}}}]
    out,_=merge_locked(base,targets,props,tokens,'empty_fields_shadow')
    assert out['components']['R1']['value']=='22K' and out['components']['__unresolved_1']['value']==' 10kΩ '


def test_primary_spatial_lock_cannot_be_bypassed_by_new_id():
    bindings=[{'owner':'0','bbox':[20,20,30,30]}]
    kept,blocked=lock_tokens([token('new','R2','1'),token('far','R2','1',(100,20,110,30))],bindings)
    assert [t['id'] for t in kept]==['far'] and len(blocked)==1


def test_primary_evidence_and_wrong_roi_owner_cannot_fill():
    base,targets=fixture();a=token('a','R2','0');b=token('b','R3','1',(50,20,60,30));b['source_tier']='primary'
    for selected in [a,b]:
        out,audit=merge_locked(base,targets,[{'id':'1','fields':{'reference':{'text':selected['raw'],'ocr_id':selected['id']}}}],[selected])
        assert out==base and audit['blocked']


def test_existing_key_collision_preserves_named_and_object_count():
    base,targets=fixture();tokens=[token('a','R1','1')]
    out,audit=merge_locked(base,targets,[{'id':'1','fields':{'reference':{'text':'R1','ocr_id':'a'}}}],tokens)
    assert out==base and audit['merge']['blocked']


def test_duplicate_new_key_rejected_simultaneously():
    base,targets=fixture();record=deepcopy(base['components']['__unresolved_1']);record['bbox']=[40,0,50,10]
    base['components']['__unresolved_2']=record;targets.append({'id':'2','baseline_key':'__unresolved_2','baseline_record':deepcopy(record)})
    tokens=[token('a','R2','1'),token('b','R2','2',(80,20,90,30))]
    props=[{'id':str(i),'fields':{'reference':{'text':'R2','ocr_id':ident}}} for i,ident in [(1,'a'),(2,'b')]]
    left,_=merge_locked(base,targets,props,tokens);right,_=merge_locked(base,targets,list(reversed(props)),tokens)
    assert left==right==base


def test_exact_literal_required():
    base,targets=fixture();tokens=[token('a','r2','1')]
    out,audit=merge_locked(base,targets,[{'id':'1','fields':{'reference':{'text':'R2','ocr_id':'a'}}}],tokens)
    assert out==base and audit['blocked'][0]['reason']=='raw_literal_changed'


def test_anchor_reference_duplicate_removed_only_for_same_spatial_owner():
    anchors=[token('anchor','R1','0')]
    local=[token('a','R1','0'),token('b','R1','0',(80,20,90,30)),token('c','R2','0')]
    kept,blocked=remove_anchor_duplicates(local,anchors)
    assert [t['id'] for t in kept]==['b','c'] and len(blocked)==1


def test_primary_anchor_cannot_migrate_to_closer_neighbor():
    from component_lab.text_matching import associate_structured
    components=[{'id':'a','type':'resistor','bbox':[0,0,10,10]},
                {'id':'b','type':'resistor','bbox':[20,0,30,10]}]
    texts=[{'id':'anchor','raw':'R1','bbox':[20,12,32,22],'score':.99}]
    unlocked=associate_structured(components,texts,clamp_costs=False)
    locked=associate_structured(components,texts,clamp_costs=False,allowed_text_owners={'anchor':['a']})
    assert unlocked['components'][1]['reference']=='R1'
    assert locked['components'][1]['reference'] is None and locked['components'][0]['reference']=='R1'


def test_unknown_owner_mask_is_rejected():
    import pytest
    from component_lab.text_matching import associate_structured
    with pytest.raises(ValueError,match='unknown IDs'):
        associate_structured([{'id':'a','type':'resistor','bbox':[0,0,10,10]}],
                             [{'id':'x','raw':'R1','bbox':[0,12,12,22]}],allowed_text_owners={'x':['missing']})


def test_locked_anchor_cannot_create_numeric_context_for_neighbor():
    from component_lab.text_matching import associate_structured
    components=[{'id':'a','type':'resistor','bbox':[0,0,10,10]},
                {'id':'b','type':'resistor','bbox':[20,0,30,10]}]
    texts=[{'id':'anchor','raw':'R1','bbox':[20,12,32,22],'score':.99},
           {'id':'number','raw':'1000','bbox':[20,24,45,34],'score':.99}]
    unlocked=associate_structured(components,texts,clamp_costs=False)
    locked=associate_structured(components,texts,clamp_costs=False,allowed_text_owners={'anchor':['a']})
    assert unlocked['components'][1]['value']=='1000'
    assert locked['components'][1]['reference'] is None and locked['components'][1]['value'] is None
