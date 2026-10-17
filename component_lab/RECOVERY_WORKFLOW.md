# 未命名元件的安全恢复

此入口用于将机器读取结果写回冻结的预测JSON，不调用检测器、OCR或API。主infer流程默认不变。调用者准备好基线、带哈希的目标manifest与模型响应后，再显式执行。

默认只恢复key/type，保留原机器Name/value。不能因此认定保留的字段正确，只是减少未经验证的改写。所有建议均为机器结果，不是人工标签或训练gold。

## 固定行为

- 只有`__unresolved_detection_*`记录可改；所有旧named元件连同null字段逐项保持不变。
- 读取前核验整体baseline及目标记录哈希；不一致拒绝写入。
- reference不确定、目标不完整、语法不支持、撞上已有key，或同批争夺同一key，整项弃权。
- 成功恢复后才允许改变该记录的类型；`component`仅由明确前缀或受限机器家族建议补充，其他已有类型先保留。
- bbox、检测总数、pins和nets不变；不得删除难例来提高F1。
- 可选`--with-fields`只改成功恢复的新记录，仍不触碰任何旧named元件。当前固定dev试验中它不如仅恢复key/type。

## 已验证命令

在`component_lab`目录，先令Python能找到`src`（已安装项目则不需要）：

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
python -m component_lab.cli recover-identities `
  --baseline outputs/unresolved_identity_recovery_20261017/baselines.json `
  --manifest outputs/unresolved_identity_recovery_20261017/manifest.json `
  --readings outputs/unresolved_identity_recovery_20261017/new_readings.json `
  --output outputs/recovery_replay_new
```

输出目录必须不存在。响应文件顶层为`items`，每项包含`id/reference/Name/value/family/target_complete/uncertain_fields`；允许只提供成功请求的目标子集，未知或重复ID拒绝。完整schema与family枚举见`identity_recovery.py`的`parse_recovery`，真实格式见上述new_readings.json。

输出每case一个JSON及`writeback_audit.json`，附原named快照哈希、逐项更改与弃权原因。已经完成的真实重放位于`outputs/unresolved_identity_recovery_20261017/cli_keys_replay/`，与实验new_keys的20份输出逐字段一致。

## 当前效果与限制

固定dev20，基线17.4207%/170TP。本轮24个新目标恢复13个命名记录，其中7个完整正确；F1为17.7680%，原170TP全部保留。6次Qwen调用合计11,832 token，约1,690 token/本轮净增完整TP；离线CLI重放无新增token。

新目标来自两个纯预测tier：有未认领同类位号候选的12目标产生6完整TP；只有附近参数/型号线索的12目标产生1完整TP。检测/分类分数仅作筛选信号，不代表校准后的正确概率。该池是保守子集，不覆盖全部漏检或无位号器件。

Qwen同时改写新记录的字段时净增仅4；4个key-only新增TP因可见非空文字与GT null的差异再次失分，另补对1个值。没有按GT逐目标择优。

历史缓存＋新读取全字段shadow为18.0826%/177TP，新增位置与key-only不同，宏F1可不同。它不是默认CLI或独立泛化成绩。旧权重存在历史暴露，正式评分规则也仍有未澄清项。

## 免费查看与后续边界

```powershell
python scripts/pilot_unresolved_recovery.py report
python scripts/audit_unresolved_recovery.py
```

两条均只读缓存/开发标签，不调用API；第二条产物仅供事后评测诊断，不能进入目标筛选。`run`是新的付费动作，本轮已完成，不要为重看结果新开付费目录。输入、预算及响应均有日志，失败不自动重试。

下一阶段优先核实复合器件拆分、图面位号与GT位置冲突。当前证据不足以支持全局改框、按单例修改类型映射或普遍补空。没有新增训练、改变标签、恢复Small或使用test40。
