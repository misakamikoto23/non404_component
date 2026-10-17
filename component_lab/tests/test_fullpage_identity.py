from copy import deepcopy
import json
import pytest
from component_lab.fullpage_identity import parse_readings,apply_readings


def fixtures():
    base={'components':{'R1':{'type':'r','bbox':[0,0,10,10],'Name':None,'value':None},
                        'R2':{'type':'r','bbox':[20,0,30,10],'Name':None,'value':'10K'}},'pins':{},'nets':{}}
    targets=[{'id':'c0','baseline_key':'R1','baseline_record':deepcopy(base['components']['R1'])},
             {'id':'c1','baseline_key':'R2','baseline_record':deepcopy(base['components']['R2'])}]
    return base,targets


def raw(ident,ref=None,name=None,value=None,uncertain=()):
    return {'id':ident,'reference':ref,'Name':name,'value':value,
            'ocr_ids':{'reference':None,'Name':None,'value':None},'uncertain_fields':list(uncertain)}


def parsed(items,targets,ocr=()):return parse_readings(json.dumps({'items':items}),targets,list(ocr))


def test_simultaneous_swap_preserves_geometry_type_and_literal():
    base,targets=fixtures();rows,_=parsed([raw('c0','R2',value=' 10kΩ '),raw('c1','R1',value='10K')],targets)
    out,audit=apply_readings(base,targets,rows,'full_replace')
    assert out['components']['R2']['bbox']==base['components']['R1']['bbox']
    assert out['components']['R2']['value']==' 10kΩ '
    assert len(out['components'])==2 and base['components']['R1']['value'] is None


def test_duplicate_claims_are_rejected_together_not_first_wins():
    base,targets=fixtures();rows,rejected=parsed([raw('c0','R3'),raw('c1','R3')],targets)
    out,_=apply_readings(base,targets,rows,'keys_only')
    assert out==base and len(rejected)==2


def test_collision_with_unchanged_owner_falls_back_without_deleting():
    base,targets=fixtures();rows,_=parsed([raw('c0','R2'),raw('c1',uncertain=['reference'])],targets)
    out,audit=apply_readings(base,targets,rows,'keys_only')
    assert out==base and audit['blocked'][0]['reason']=='rename_collision_reverted'


def test_rename_collision_cascade_preserves_all_original_objects():
    base,targets=fixtures()
    base['components']['R3']={'type':'r','bbox':[40,0,50,10],'Name':None,'value':'100K'}
    targets.append({'id':'c2','baseline_key':'R3','baseline_record':deepcopy(base['components']['R3'])})
    rows,_=parsed([raw('c0','R2'),raw('c1','R3'),raw('c2',uncertain=['reference'])],targets)
    out,audit=apply_readings(base,targets,rows,'keys_only')
    assert out==base and len(audit['blocked'])==2


def test_uncertain_preserves_baseline_but_confirmed_absent_can_clear():
    base,targets=fixtures();rows,_=parsed([raw('c0','R1'),raw('c1','R2',uncertain=['value'])],targets)
    out,_=apply_readings(base,targets,rows,'full_replace')
    assert out['components']['R2']['value']=='10K'
    rows,_=parsed([raw('c0','R1'),raw('c1','R2')],targets)
    out,_=apply_readings(base,targets,rows,'full_replace')
    assert out['components']['R2']['value'] is None


def test_mismatched_ocr_literal_downgrades_field_without_normalization():
    base,targets=fixtures();a=raw('c0','R1',value='10k');a['ocr_ids']['value']='o0'
    rows,rejected=parsed([a,raw('c1','R2')],targets,[{'id':'o0','raw':'10K','bbox':[0,12,10,18]}])
    assert rows[0]['fields']['value']['status']=='uncertain'
    assert rejected[0]['reason']=='ocr_literal_mismatch'


def test_overlapping_ocr_claims_by_different_components_quarantine_both():
    base,targets=fixtures();a=raw('c0','R1',value='10K');b=raw('c1','R2',value='10K')
    a['ocr_ids']['value']='o0';b['ocr_ids']['value']='o1'
    ocr=[{'id':'o0','raw':'10K','bbox':[0,12,10,18]}, {'id':'o1','raw':'10K','bbox':[0,12,10,18]}]
    rows,rejected=parsed([a,b],targets,ocr)
    assert all(r['fields']['value']['status']=='uncertain' for r in rows) and len(rejected)==2


def test_equal_strings_at_distinct_positions_are_allowed():
    base,targets=fixtures();a=raw('c0','R1',value='10K');b=raw('c1','R2',value='10K')
    a['ocr_ids']['value']='o0';b['ocr_ids']['value']='o1'
    ocr=[{'id':'o0','raw':'10K','bbox':[0,12,10,18]}, {'id':'o1','raw':'10K','bbox':[20,12,30,18]}]
    rows,rejected=parsed([a,b],targets,ocr)
    assert not rejected and all(r['fields']['value']['status']=='read' for r in rows)


def test_unresolved_keys_locks_every_named_record_and_other_fields():
    base,targets=fixtures();base['components']['__unresolved_detection_2']=base['components'].pop('R2')
    targets[1]['baseline_key']='__unresolved_detection_2'
    rows,_=parsed([raw('c0','R9',value='100K'),raw('c1','R3',value='200K')],targets)
    out,_=apply_readings(base,targets,rows,'unresolved_keys')
    assert out['components']['R1']==base['components']['R1']
    assert out['components']['R3']==base['components']['__unresolved_detection_2']


def test_unknown_missing_or_geometry_fields_are_rejected():
    base,targets=fixtures()
    for items in [[raw('c0','R1')],[raw('c0','R1'),raw('unknown','R2')],
                  [{**raw('c0','R1'),'bbox':[1,2,3,4]},raw('c1','R2')]]:
        with pytest.raises(ValueError):parsed(items,targets)


def test_provider_strings_are_redacted_recursively_before_persistence():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/pilot_fullpage_component_identity.py'
    spec=importlib.util.spec_from_file_location('fullpage_pilot_redaction_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    secret='sk-UNIT_TEST_NOT_A_REAL_SECRET_0123456789'
    out=module.redact({'content':secret,'other':[{'model':secret}], 'tokens':123},secret)
    assert secret not in json.dumps(out) and out['content']=='[REDACTED]' and out['tokens']==123
