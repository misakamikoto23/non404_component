"""Publish a trained balanced checkpoint, reproducible statistics and visual review."""
from __future__ import annotations
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys
from PIL import Image
from bootstrap_resnet_data import LAB,OUT,rows,sha
from component_lab.review_workflow import atomic_json

def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from component_lab.models import ResnetClassifier
    run=LAB/'runs/resnet18_balanced_20261016';data=LAB/'data/resnet/balanced_20261016'
    evaluation=json.loads((run/'evaluation.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
    baseline=json.loads((LAB/'runs/resnet18_bootstrap_20261016/evaluation.json').read_text(encoding='utf-8'))
    classes=summary['classes'];entries=rows(data/'manifest.jsonl')
    target=LAB/'weights/resnet18_component_balanced_v1.pt'
    if target.exists() and sha(target)!=sha(run/'best.pt'):raise ValueError('Named checkpoint already differs; do not overwrite')
    shutil.copy2(run/'best.pt',target)
    assert sha(target)==sha(run/'best.pt')
    # Load through the actual public inference class, on the requested 4060 Ti.
    model=ResnetClassifier(str(target),'cuda:0')
    assert model.classes==classes
    predictions=json.loads((run/'val_predictions.json').read_text(encoding='utf-8'))
    subset=predictions[::max(1,len(predictions)//12)][:12]
    checks=[]
    for row in subset:
        with Image.open(row['file']) as im:result=model.predict(im.convert('RGB'))
        checks.append({'file':row['file'],'evaluation_prediction':row['predicted'],'public_classifier_prediction':result['type'],
                       'absolute_score_difference':abs(row['score']-result['score'])})
    assert all(r['evaluation_prediction']==r['public_classifier_prediction'] for r in checks)
    atomic_json(OUT/'classifier_load_check.json',{'device':'cuda:0','device_name':torch.cuda.get_device_name(0),'weights_sha256':sha(target),'samples':checks,'all_argmax_consistent':True})
    real=[summary['per_class'][c]['selected_real_crops'] for c in classes]
    augmented=[summary['per_class'][c]['augmented_crops'] for c in classes]
    fig,ax=plt.subplots(figsize=(10,10))
    ax.barh(classes,real,label='Real crop files',color='#328ba8');ax.barh(classes,augmented,left=real,label='Train-only augmented views',color='#d7a843')
    ax.invert_yaxis();ax.set_xlabel('Training files per class');ax.set_title('Equal class quotas: 256 each; source diversity remains unequal');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(OUT/'class_balance.png',dpi=150);plt.close(fig)
    history=json.loads((run/'history.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history]);axes[0].set_title('Training loss');axes[0].set_xlabel('Epoch')
    for key,label in [('accuracy','Val label agreement'),('macro_f1_present_classes','Val macro F1')]:axes[1].plot([r['epoch'] for r in history],[r['val'][key] for r in history],label=label)
    axes[1].set_ylim(0,1);axes[1].set_xlabel('Epoch');axes[1].legend();axes[1].set_title('Weak/silver validation labels');fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=150);plt.close(fig)
    missing=[c for c,n in zip(classes,evaluation['val']['support']) if n==0]
    rare={c:summary['per_class'][c]['available_real_crops'] for c in classes if summary['per_class'][c]['available_real_crops']<10}
    teacher=json.loads((OUT/'teacher/summary.json').read_text(encoding='utf-8'));confirm=json.loads((OUT/'teacher_confirm/summary.json').read_text(encoding='utf-8'))
    result={'weights':str(target),'weights_sha256':sha(target),'architecture':'resnet18','classes':classes,'balanced_dataset':str(data),
        'train_files':summary['train_crops'],'per_class_files':256,'real_files':sum(real),'augmented_files':sum(augmented),'val_files':summary['val_crops'],
        'device':config['device'],'device_name':config['device_name'],'best_epoch':evaluation['best_epoch'],'epochs_completed':evaluation['epochs_completed'],
        'val':evaluation['val'],'original_proxy_val':evaluation['original_proxy_val'],'baseline_val':baseline['val'],
        'classes_without_val':missing,'classes_with_under_10_real_crops':rare,'new_api_requests':teacher['requests']+confirm['requests'],
        'new_api_reported_tokens':teacher['reported_usage']['total_tokens']+confirm['reported_usage']['total_tokens'],
        'cached_predictions_reused':1687,'independent_agent_audit_crops':72,'all_targets_weak_or_silver':True,'test_used':False}
    atomic_json(OUT/'RESULT.json',result)
    atomic_json(LAB/'data/resnet/active_dataset.json',{'dataset':str(data),'status':'trained',
        'classes':27,'per_class_training_files':256,'train_files':6912,'val_files':951,
        'weights':str(target),'run_directory':str(run),'preferred_device':'cuda:0','device_name':config['device_name'],
        'label_quality':'weak/silver; augmentation is not independent real data'})
    lines=['# ResNet18元件分类：均衡数据训练完成','',
        f"权重：`{target}`。训练设备：{config['device_name']}（{config['device']}）。最佳第{evaluation['best_epoch']}轮，共运行{evaluation['epochs_completed']}轮。",'',
        f"27类，每类256份，共{summary['train_crops']}份训练文件，其中{sum(real)}份真实裁剪、{sum(augmented)}份训练侧增广。每轮不放回遍历，因此每类恰好出现256次。验证集951份，原图分组和文件字节保持不变，测试集未使用。",'',
        '| 模型 | 验证标签一致率 | 宏平均F1 | 宏平均召回率 |','|---|---:|---:|---:|']
    for title,stats in [('此前不等量基线',baseline['val']),('每类256份均衡训练',evaluation['val'])]:
        lines.append(f"| {title} | {stats['accuracy']:.2%} | {stats['macro_f1_present_classes']:.2%} | {stats['macro_recall_present_classes']:.2%} |")
    lines+=['',f"宏指标只统计验证集中有样本的{27-len(missing)}类。没有验证样本的类别：{', '.join(missing)}。",'',
        '验证标签由保守原标注与AI银标构成，人工仅确认了部分检测框几何。上述数值是标签一致率，真实分类准确率仍需独立人工类别核验。均衡训练和原基线的数据组成不同，比较用于观察本次改动，不是单变量消融。','',
        f"只按原始保守类别的验证子集：{evaluation['original_proxy_val']['samples']}份，标签一致率{evaluation['original_proxy_val']['accuracy']:.2%}。",'',
        '## 真实来源与增广数量','','| 类别 | 选用真实裁剪 | 原图来源数 | 增广 | 训练总数 | 验证样本 | 验证F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for i,c in enumerate(classes):
        s=summary['per_class'][c];f1=f"{evaluation['val']['per_class_f1'][i]:.1%}" if evaluation['val']['support'][i] else '无验证样本'
        lines.append(f"| {c} | {s['selected_real_crops']} | {s['real_source_images']} | {s['augmented_crops']} | 256 | {s['validation_crops']} | {f1} |")
    lines+=['','8类不足10个真实裁剪；增广不会增加独立符号模板或原图来源。数量均衡不能保证这些类别的泛化能力。','',
        '## 自动分类与数据筛选','',
        '先从60张审核图及原PCBnet训练/验证图裁出11323个元件，复用1687个Qwen缓存。未知类别不强制分类，未审核的第四批HF图未参与。抽查发现ANT/Battary/Motor原标签存在语义冲突，停止将这三类直接映射为分类真值。',
        '新Qwen上下文分类997个，得到996个有效结果；其中312个用紧框视角再次复核，得到310个有效结果。仅机器标签需要双视角或独立智能体一致；独立5.6-Sol轻度抽查72个，分歧/不确定样本隔离。原始类别清晰的验证标签保留，不按教师是否同意来挑选。',
        f"本轮新付费API共{result['new_api_requests']}次，服务返回{result['new_api_reported_tokens']:,} tokens；无自动付费重试。智能体独立tokens/费用未取得，不估算金额。",'',
        '共同验证源：原PCBnet val的30张原图。训练、验证原图互斥，并排除跨两边完全相同的裁剪；训练内按来源多样性选样。增广只在训练侧，使用整块符号旋转、轻微角度/缩放/位移和色彩对比度变化，保持完整符号。',
        '优化：ImageNet预训练ResNet18，输入224×224保持比例补白，前2轮只训练分类头，后续整体微调，AdamW、余弦学习率、标签平滑0.05、AMP。仅机器银标的损失权重较低；类别数已严格相等，不再按类重复抽样。','',
        '## 文件与复现','',
        f"数据：`{data}`；训练日志：`{run}`。模型兼容现有`ResnetClassifier`，已在4060 Ti通过公开加载器对12个验证裁剪做推理核对。",'',
        '```powershell',
        '.\\.venv-yolo\\Scripts\\python.exe scripts\\train_resnet_components.py --data data/resnet/balanced_20261016 --output runs/resnet18_balanced_new_run --device cuda:0 --epochs 25 --batch 96',
        '```','',
        '训练输入使用外扩8%且至少2像素的元件裁剪，之后按比例补白至224×224。此模型只负责符号类别，位号/参数OCR由后续模块处理。',
        '[均衡样本图库](gallery.html) · [HF数据核验](HF_DATASETS.md) · [结果JSON](RESULT.json) · [均衡数据校验](balanced_integrity.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    table='<table><tr><th>类别</th><th>真实/原图来源</th><th>增广</th><th>总数</th><th>验证F1</th><th>真实样本与增广示例</th></tr>'
    for i,c in enumerate(classes):
        group=[r for r in entries if r['split']=='train' and r['category']==c]
        examples=[r for r in group if not r['is_augmented']][:2]+[r for r in group if r['is_augmented']][:2]
        thumbs=''.join(f'<img loading="lazy" src="{html.escape(os.path.relpath(data/r["file"],OUT).replace(chr(92),"/"))}" title="{r["crop_id"]}; augmented={r["is_augmented"]}">' for r in examples)
        s=summary['per_class'][c];f1=f"{evaluation['val']['per_class_f1'][i]:.1%}" if evaluation['val']['support'][i] else '无验证样本'
        table+=f'<tr><td>{c}</td><td>{s["selected_real_crops"]} / {s["real_source_images"]}</td><td>{s["augmented_crops"]}</td><td>256</td><td>{f1}</td><td>{thumbs}</td></tr>'
    table+='</table>'
    (OUT/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>ResNet18均衡分类数据</title><style>body{font:16px system-ui;margin:24px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}td img{width:95px;height:95px;object-fit:contain;background:white;margin:4px}.plot{max-width:1000px;width:100%}</style><h1>ResNet18：27类，每类256份</h1><p>真实裁剪与训练侧增广分别列出。原图来源不足的类别，文件数相同也不代表来源多样性相同。类别标签为弱标注/银标，验证分数表示标签一致率。</p><p><a href="REPORT.md">完整报告</a> · <a href="HF_DATASETS.md">HF核验</a></p><img class="plot" src="class_balance.png"><img class="plot" src="training_curves.png">'+table,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
