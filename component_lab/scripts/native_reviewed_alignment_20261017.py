"""Validate native source boxes against frozen human HF geometry; export silver pairs."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'src'))
from component_lab.review_workflow import load_batch, atomic_json
import native_schematic_label_pilot as native
import native_geometry_calibration as calibration

OUT = LAB / 'outputs/native_reviewed_alignment_20261017'
NATIVE = LAB / 'outputs/unattended_20261016/native_labels'
TYPE_MAP = {
    'Device:R': 'resistor', 'Device:R_Small': 'resistor', 'Device:R_US': 'resistor',
    'Device:R_Small_US': 'resistor', 'Device:C': 'capacitor', 'Device:C_Small': 'capacitor',
    'Device:C_Polarized': 'capacitor', 'Device:C_Polarized_Small': 'capacitor',
    'Device:L': 'inductor', 'Device:L_Small': 'inductor',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ious(a, b):
    a, b = np.asarray(a).reshape(-1,4), np.asarray(b).reshape(-1,4)
    lo = np.maximum(a[:,None,:2], b[None,:,:2])
    hi = np.minimum(a[:,None,2:], b[None,:,2:])
    intersection = np.prod(np.maximum(hi-lo, 0), axis=2)
    aa = np.prod(np.maximum(a[:,2:]-a[:,:2], 0), axis=1)
    ba = np.prod(np.maximum(b[:,2:]-b[:,:2], 0), axis=1)
    return intersection / np.maximum(aa[:,None]+ba[None,:]-intersection, 1e-9)


def match_boxes(source, human, minimum=.5):
    matrix = ious(source, human)
    if not matrix.size:
        return matrix, []
    good = matrix >= minimum
    cardinality = min(matrix.shape) + 1
    rows, cols = linear_sum_assignment((~good)*cardinality + 1-matrix)
    return matrix, [(int(i),int(j)) for i,j in zip(rows,cols) if good[i,j]]


def association_gate(matrix, i, j):
    score = float(matrix[i,j])
    row_other = max((float(v) for k,v in enumerate(matrix[i]) if k != j), default=0.)
    col_other = max((float(v) for k,v in enumerate(matrix[:,j]) if k != i), default=0.)
    return score >= .75 and score-row_other >= .15 and score-col_other >= .15


def run(resume=False):
    if OUT.exists():
        assert resume and not (OUT/'summary.json').exists(), 'Keep completed outputs unchanged'
    else:
        OUT.mkdir(parents=True)
    atomic_json(OUT / 'PLAN.json', dict(
        purpose='Independent human-geometry check of source projections, plus conservative machine semantic association.',
        gates=dict(native_iou=.75, row_margin=.15, column_margin=.15,
                   page='Existing wire-holdout and pin-line tests; no fitting to human boxes',
                   type_whitelist=TYPE_MAP, geometry_issue_allowlist=['default_stroke_width_not_resolved']),
        policy='Crops are source-derived silver semantics paired with human geometry, never human-confirmed classes. No training/default change/full-page YOLO export.',
        test_used=False, api_calls=0))
    active = read(LAB / 'data/reviewed_versions/manual_round_005/yolo/manifest.json')
    active_ids = {s['case'] for s in active['samples'] if s['split']=='train'}
    groups = read(LAB / 'outputs/eval_split_audit_20261017/groups.json')
    gf = {i:g for g,row in groups.items() for i in row['members']}
    banned = set(read(LAB / 'outputs/eval_split_audit_20261017/forbidden_for_clean_retrain.json')['held_out_groups'])
    fps = read(LAB / 'outputs/eval_split_audit_20261017/fingerprints.json')['images']
    banned.update(gf[f['id']] for f in fps if f.get('source')=='component_lab' and f.get('role') in ('val','test'))
    atomic_json(OUT / 'protected_groups.json', sorted(banned))
    rows, silver, page_cards, ignored, seen = [], [], [], [], set()
    for snapshot in sorted((LAB / 'data/review_snapshots').glob('batch_*_round_*')):
        state = load_batch(snapshot)
        for item in state['items']:
            source_id = item.get('source_metadata', {}).get('source_record_id')
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            group = gf.get('hf_os:'+source_id)
            if item['id'] not in active_ids or group is None or group in banned:
                ignored.append(dict(case=item['id'], reason='not_active_train_or_protected_or_unknown_group'))
                continue
            source = read(NATIVE / 'sources' / (source_id+'.json'))
            if not source.get('paper_mm'):
                ignored.append(dict(case=item['id'], reason='unknown_paper_size_fail_closed'))
                continue
            image_path = snapshot / item['image_file']
            assert sha(image_path)==source['image_sha256'], 'Human/source image mismatch'
            native_path = Path(source['source_file'])
            assert sha(native_path)==source['source_sha256'], 'Source changed'
            image = Image.open(image_path).convert('RGB')
            root = native.parse_sexpr(native_path.read_text(encoding='utf-8-sig'))
            reg, endpoints, params, fields = calibration.registration(root, np.asarray(image), source['paper_mm'])
            pins = calibration.pin_diagnostics(root, np.asarray(image), reg, endpoints, params)
            page_gate = reg['registration_supported'] and pins['pin_alignment_supported'] and not source['hierarchical_sheets']
            sx, sy = reg['paper_scales']
            sources = [s for s in source['symbols'] if s['bbox_mm'] is not None]
            boxes = [[b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy] for b in (s['bbox_mm'] for s in sources)]
            human = [b for b in item['boxes'] if b['review_status']=='accepted']
            matrix, matches = match_boxes(boxes, [b['bbox'] for b in human])
            page_matches, suggestions = [], []
            for i,j in matches:
                symbol, reviewed = sources[i], human[j]
                error = max(abs(a-b) for a,b in zip(boxes[i],reviewed['bbox']))
                row = dict(source_index=symbol['index'], reference=symbol['reference'],
                           source_value=symbol['value'], lib_id=symbol['lib_id'],
                           human_box_id=reviewed['id'], iou=float(matrix[i,j]), max_coordinate_error=error,
                           source_bbox=boxes[i], human_bbox=reviewed['bbox'],
                           source_geometry_issues=symbol['issues'])
                page_matches.append(row)
                allowed_issues = set(symbol['issues']) <= {'default_stroke_width_not_resolved'}
                if not (page_gate and allowed_issues and symbol['lib_id'] in TYPE_MAP
                        and symbol['unit']==1 and symbol['body_style']==1 and symbol.get('dnp')!='yes'
                        and association_gate(matrix,i,j)):
                    continue
                suggestion = dict(row, case=item['id'], source_id=source_id, source_project=source['source_repository'],
                    design_group=group, class_name=TYPE_MAP[symbol['lib_id']],
                    label_source='native_library_semantics_via_conservative_geometry_association_to_human_bbox',
                    category_confirmed=False, intended_split='train_candidate_only',
                    source_sha256=source['source_sha256'], image_sha256=source['image_sha256'],
                    source_properties=symbol['properties'],
                    caveat='Source facts are not guaranteed displayed contest fields. Default stroke/body bounds unresolved; crop uses human geometry. Not independent manual class ground truth.')
                crop_path = OUT / 'crops' / suggestion['class_name'] / (item['id']+'_'+str(symbol['index'])+'.png')
                crop_path.parent.mkdir(parents=True,exist_ok=True)
                x1,y1,x2,y2=reviewed['bbox']
                image.crop((max(0,int(x1)-2),max(0,int(y1)-2),min(image.width,int(x2)+3),min(image.height,int(y2)+3))).save(crop_path)
                suggestion['crop_file']=crop_path.relative_to(OUT).as_posix()
                silver.append(suggestion)
                suggestions.append(suggestion)
            page_dir = OUT / item['id']
            page_dir.mkdir(exist_ok=resume)
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for b in human:
                draw.rectangle(b['bbox'], outline='#0087cc', width=2)
            for b in boxes:
                draw.rectangle(b, outline='#ed8500', width=1)
            overlay.save(page_dir/'overlay.png')
            record = dict(case=item['id'], source_id=source_id, group=group,
                snapshot=str(snapshot), human_boxes=len(human), source_candidates=len(boxes),
                source_count_without_bbox=len(source['symbols'])-len(boxes), matched_iou50=len(matches),
                unmatched_human=len(human)-len(matches), unmatched_source=len(boxes)-len(matches),
                page_gate=page_gate, registration=reg, pins=pins, matches=page_matches,
                silver_suggestions=len(suggestions), wholepage_yolo_exported=False)
            atomic_json(page_dir/'alignment.json',record)
            rows.append(record)
            cells=''.join(f'<div class="crop"><img loading="lazy" src="../{s["crop_file"]}"><p>{html.escape(str(s["reference"]))} / {s["class_name"]}<br>Value={html.escape(str(s["source_value"]))}<br>IoU={s["iou"]:.3f}</p></div>' for s in suggestions)
            page_dir.joinpath('index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1400px;margin:24px auto}img{max-width:100%}.crop{display:inline-block;vertical-align:top;padding:12px;max-width:220px}.crop img{max-height:120px}</style><a href="../gallery.html">返回</a><h1>'+item['id']+'</h1><p>蓝：人工审核框；橙：原生投影候选。不是模型预测。页面质量门='+str(page_gate)+'</p><img src="overlay.png"><h2>源文件语义配对建议（不是人工类别真值）</h2>'+cells,encoding='utf-8')
            page_cards.append(f'<li><a href="{item["id"]}/index.html">{item["id"]}</a>：人工{len(human)}，原生候选{len(boxes)}，IoU50匹配{len(matches)}，页面门{page_gate}，建议{len(suggestions)}</li>')
            print(item['id'],len(matches),'matches',len(suggestions),'source-semantic suggestions',flush=True)
    counts=Counter(s['class_name'] for s in silver)
    summary=dict(pages=len(rows),page_gate_passed=sum(r['page_gate'] for r in rows),
        human_boxes=sum(r['human_boxes'] for r in rows),source_candidates=sum(r['source_candidates'] for r in rows),
        matched_iou50=sum(r['matched_iou50'] for r in rows),silver_crops=len(silver),class_counts=dict(counts),
        geometry_checks='Source projections vs independent frozen human geometry; IoU50 counts are correspondence diagnostics, not detector evaluation.',
        ignored=ignored,training_modified=False,fullpage_yolo_exported=False,api_calls=0,test_used=False)
    atomic_json(OUT/'summary.json',summary)
    atomic_json(OUT/'silver_pairs.json',dict(items=silver,category_confirmed=False,training_modified=False))
    OUT.joinpath('gallery.html').write_text('<meta charset="utf-8"><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:28px auto;padding:16px}a{color:#176bc4}</style><h1>原生源文件 × 已审核框：独立几何验证</h1><p>只使用当前训练保留的HF人工审核图。原生框未按人工框拟合；橙框来自源文件与纸张变换，蓝框来自冻结人工审核。位号/参数保留源字符串，不将源字段自动视为图面可见文字。</p><p>配对建议需同时通过导线独立线段/引脚像素门、无子页、IoU≥.75及双向竞争差≥.15、窄类型白名单与几何问题筛选。它们仍是源语义银标，非人工类别真值；不改变现有ResNet或训练集。不能将仅有白名单框的整图作为完整YOLO标签。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><ul>'+''.join(page_cards)+'</ul>',encoding='utf-8')
    OUT.joinpath('REPORT.md').write_text('# 原生源文件与人工框独立对照\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n完整预声明门见PLAN.json，逐图alignment.json与silver_pairs.json保留来源。结果只为配对建议，不是模型准确率或人工分类真值。未修改训练集、旧审核、任何权重或测试。默认线宽/完整符号轮廓仍需独立渲染验证；源文件Reference/Value不自动等于比赛key/Name/value。\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-incomplete',action='store_true')
    run(parser.parse_args().resume_incomplete)
