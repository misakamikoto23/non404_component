import json
import pytest
from component_lab.ocr_slot_choices import parse_id_choices
from component_lab.fullpage_identity import parse_readings

TOKENS = {'O1': {'id': 'O1', 'raw': 'u2', 'bbox': [0, 0, 10, 10]},
          'O2': {'id': 'O2', 'raw': '0.01 μF', 'bbox': [0, 20, 50, 30]}}


def test_raw_strings_and_null_uncertain_are_distinct():
    row = parse_id_choices('{"items":[["C1","O1","?","O2"]]}', ['C1'], TOKENS, {'C1': ['O1', 'O2']})[0]
    assert row['reference'] == 'u2' and row['value'] == '0.01 μF'
    assert row['Name'] is None and row['uncertain_fields'] == ['Name']
    assert row['ocr_ids']['value'] == 'O2'


@pytest.mark.parametrize('content', [
    '{"items":[],"items":[]}', '{"items":[],"extra":0}', '{"items":[["C1",true,null,null]]}',
    '{"items":[["C1",1,null,null]]}', '{"items":[["C1","invented",null,null]]}',
    '{"items":[["C1","O2",null,null]]}', '{"items":[]}',
    '{"items":[["C1",null,null,null],["C1",null,null,null]]}',
    '{"items":[["C2",null,null,null]]}'
])
def test_invalid_choices_rejected(content):
    with pytest.raises(ValueError):
        parse_id_choices(content, ['C1'], TOKENS, {'C1': ['O1']})


def test_shared_token_claims_are_quarantined_by_existing_global_validator():
    verbose = parse_id_choices('{"items":[["C1","O1",null,null],["C2","O1",null,null]]}',
                               ['C1', 'C2'], TOKENS, {'C1': ['O1'], 'C2': ['O1']})
    rows, rejected = parse_readings(json.dumps({'items': verbose}), [{'id': 'C1'}, {'id': 'C2'}], list(TOKENS.values()))
    assert all(r['fields']['reference']['status'] == 'uncertain' for r in rows)
    assert len(rejected) == 2
