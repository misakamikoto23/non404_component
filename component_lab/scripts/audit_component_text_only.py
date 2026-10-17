"""Read-only DEV text-shape audit. GT is diagnostic, never an inference gate."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from component_lab.eval.metric import canonical_type, text_ok
from component_lab.text_matching import _distance


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = row
    return previous[-1]


def classify(pred, gt):
    if pred is None and gt is not None:
        return 'A_pred_null_gt_nonnull', None
    if pred is not None and gt is None:
        return 'B_pred_nonnull_gt_null', None
    if pred is None or gt is None or pred == gt:
        raise ValueError('Expected strictly different text fields')
    distance = edit_distance(pred, gt)
    same_casefold = pred.casefold() == gt.casefold()
    same_whitespace = ''.join(pred.split()) == ''.join(gt.split())
    close = same_casefold or same_whitespace or (distance <= 2 and distance*3 <= max(len(pred), len(gt)))
    return ('C_small_literal_difference' if close else 'D_larger_literal_difference'), {
        'levenshtein': distance, 'casefold_equal_diagnostic_only': same_casefold,
        'whitespace_compacted_equal_diagnostic_only': same_whitespace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/component_text_only_audit_20261017')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a fresh immutable audit output')
    base = ROOT/'outputs/advance_until_bottleneck_20261017/lexical_experiment'
    source = read(base/'report.json')
    rows, correct_null, hashes = [], [], {}
    for diag_id, diag in source['diagnostics'].items():
        if not diag_id.endswith('/identity_v1'): continue
        case = diag_id.split('/')[0]
        association_path = base/'identity_v1/associations'/f'{case}.json'
        association = read(association_path)
        hashes[case] = hashlib.sha256(association_path.read_bytes()).hexdigest()
        owners = {}
        for component in association['components']:
            for field, evidence in component.get('association_evidence', {}).items():
                if evidence.get('text_id'):
                    owners.setdefault(evidence['text_id'], []).append({'source_id': component['id'], 'field': field})
        traces = read(base/'identity_v1/traces'/f'{case}.json')['components']
        source_for_key = {t['key']: t['source_id'] for t in traces}
        with Image.open(ROOT/'outputs/component_identity_round_001_20261017/inputs'/case/'source.png') as image:
            height = image.height
        for row in diag['named_components']:
            kind = canonical_type(row['gt']['type'])
            if kind not in {'ic', 'resistor', 'capacitor'}: continue
            field = 'Name' if kind == 'ic' else 'value'
            pred, gt = row['prediction'], row['gt']
            if row['state'] == 'correct' and pred.get(field) is None and gt.get(field) is None:
                correct_null.append({'case': case, 'key': row['key'], 'field': field, 'prediction': pred, 'gt': gt})
            if row['state'] != 'key_hit_other_field_error' or row['checks'] != {'type': True, 'bbox': True, 'text': False}:
                continue
            category, diff = classify(pred.get(field), gt.get(field))
            x1, y1, x2, y2 = gt['bbox']; body = [x1, height-y2, x2, height-y1]
            exact = []
            if gt.get(field) is not None:
                for text in association['texts']:
                    if text['raw'] != gt[field]: continue
                    ownership = owners.get(text['id'], [])
                    exact.append({'text_id': text['id'], 'raw': text['raw'], 'bbox': text['bbox'],
                                  'distance_to_gt_body': _distance(text['bbox'], body),
                                  'within_60px_body_neighborhood': _distance(text['bbox'], body) <= 60,
                                  'role_candidates': text['role_candidates'], 'machine_owners': ownership,
                                  'is_unassigned': not ownership,
                                  'has_other_owner': any(o['source_id'] != source_for_key[row['key']] for o in ownership)})
            rows.append({'case': case, 'key': row['key'], 'type': gt['type'], 'field': field,
                         'pred_text': pred.get(field), 'gt_text': gt.get(field), 'category': category,
                         'literal_difference': diff, 'exact_ocr_candidates': exact})
    assert len(rows) == 57
    counts = Counter(r['category'] for r in rows)
    subtype = {category: dict(Counter(r['field'] for r in rows if r['category'] == category)) for category in counts}
    group_a = [r for r in rows if r['category'] == 'A_pred_null_gt_nonnull']
    availability = {
        'A_exact_target_anywhere_on_page': sum(bool(r['exact_ocr_candidates']) for r in group_a),
        'A_exact_target_within_60px_body': sum(any(t['within_60px_body_neighborhood'] for t in r['exact_ocr_candidates']) for r in group_a),
        'A_no_exact_target_anywhere': sum(not r['exact_ocr_candidates'] for r in group_a),
        'A_near_exact_has_other_machine_owner': sum(any(t['within_60px_body_neighborhood'] and t['has_other_owner'] for t in r['exact_ocr_candidates']) for r in group_a),
        'A_near_exact_has_unassigned_candidate': sum(any(t['within_60px_body_neighborhood'] and t['is_unassigned'] for t in r['exact_ocr_candidates']) for r in group_a)}
    cohort_qwen = {}
    for variant in ['direct_empty_only', 'direct_text_replacement']:
        recovered = []
        for row in rows:
            pred = read(ROOT/'outputs/qwen_direct_text_pilot_20261017'/variant/'predictions'/f"{row['case']}.json")['components'].get(row['key'])
            if pred and pred.get(row['field']) == row['gt_text']:
                recovered.append(f"{row['case']}/{row['key']}")
        cohort_qwen[variant] = {'recovered_original_57': recovered, 'remaining_original_57': 57-len(recovered)}
    # A counterexample uses memory copies only; it does not create predictions.
    counterexample = None
    if correct_null:
        row = correct_null[0]; original = row['prediction']; changed = deepcopy(original)
        changed[row['field']] = '10k' if row['field'] == 'value' else 'EXAMPLE_MODEL'
        as_tuple = lambda r: (r['type'], r.get('Name'), r.get('value'), r['bbox'])
        counterexample = {'case': row['case'], 'key': row['key'], 'field': row['field'],
                         'gt_text': None, 'hypothetical_fill': changed[row['field']],
                         'original_text_check': text_ok(as_tuple(original), as_tuple(row['gt'])),
                         'after_fill_text_check': text_ok(as_tuple(changed), as_tuple(row['gt'])),
                         'memory_only_no_prediction_written': True}
    report = {'kind': 'fixed_DEV57_required_text_literal_shape_audit_not_root_cause_labels',
              'counts': dict(counts), 'counts_by_field': subtype, 'A_availability': availability,
              'rows': rows, 'correct_null_controls': correct_null, 'correct_null_control_count': len(correct_null),
              'null_fill_counterexample': counterexample, 'existing_qwen_on_original_57': cohort_qwen,
              'similarity_rule': 'casefold equality OR whitespace-compacted equality OR Levenshtein<=2 and 3*distance<=max_length; diagnostics only',
              'neighborhood_rule': 'bbox-to-bbox Euclidean gap <=60 original pixels; GT-assisted diagnosis only',
              'source_report_sha256': hashlib.sha256((base/'report.json').read_bytes()).hexdigest(),
              'association_sha256': hashes, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'api_calls': 0, 'model_forwards': 0, 'training_runs': 0, 'sealed_test_used': False,
              'labels_modified': False, 'predictions_modified': False}
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    labels = {'A_pred_null_gt_nonnull': 'A 预测空、GT非空', 'B_pred_nonnull_gt_null': 'B 预测非空、GT空',
              'C_small_literal_difference': 'C 均非空、字面差异小', 'D_larger_literal_difference': 'D 均非空、字面差异较大'}
    lines = ['# 57项必要文本失败审计 — 2026-10-17', '',
             '固定纯本地基线17.42%的dev20 cohort，只读缓存；未调用OCR/Qwen或训练，未修改标签与预测。', '',
             '|形态分组|数量|Name|value|', '|---|---:|---:|---:|']
    for category in labels:
        lines.append(f"|{labels[category]}|{counts[category]}|{subtype.get(category,{}).get('Name',0)}|{subtype.get(category,{}).get('value',0)}|")
    lines += ['', 'C的规则：仅大小写/去空白相同，或编辑距离≤2且不超过较长串长度的1/3。C/D只是字面差异，不直接证明OCR错字或槽位错配；不可自动替换输出。', '',
              '## A类已有OCR证据', '', '```json', json.dumps(availability,ensure_ascii=False,indent=2), '```', '',
              '60px使用GT主体框与文本框间距，只作诊断，不能作为部署门控。全页同字符串可能属于别的同值元件，机器owner也可能错误；这些统计不能当可恢复TP数量。', '',
              f'## 补空风险与正确对照\n\n在170个完整正确的非匿名元件中，有{len(correct_null)}个必要文本预测和GT都为null。只要填入错误非空字符串，原有TP就可能变为FP+FN。key锁不保护字段正确性。下面是只在内存复制上的反例，没有写入任何预测：',
              '', '```json', json.dumps(counterexample,ensure_ascii=False,indent=2), '```', '',
              '## 既有Qwen对原57项的作用', '', '```json', json.dumps(cohort_qwen,ensure_ascii=False,indent=2), '```', '',
              'Qwen总增益4/5TP不是全部来自这57项，原cohort与补回位号的其他样本必须分开。57项满足本地key/type/bbox检查，并不证明物理owner100%正确。', '',
              '## 逐项原文', '', '|case/key|field|pred原文|GT原文|分组|', '|---|---|---|---|---|']
    literal = lambda s: json.dumps(s, ensure_ascii=False).replace('|','\\|')
    lines += [f"|{r['case']}/{r['key']}|{r['field']}|{literal(r['pred_text'])}|{literal(r['gt_text'])}|{r['category'][0]}|" for r in rows]
    lines += ['', '复现：`python scripts/audit_component_text_only.py --output outputs/text_only_audit_fresh`。输出目录须不存在。全部证据、字面距离和机器ownership见audit.json。']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'counts':dict(counts),'by_field':subtype,'A_availability':availability,
                      'correct_null_controls':len(correct_null),'null_fill_counterexample':counterexample,
                      'qwen_original57':cohort_qwen},ensure_ascii=True,indent=2))


if __name__ == '__main__': main()
