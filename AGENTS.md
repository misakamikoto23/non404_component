# AGENTS.md — EIT&BTD 仓库共享规则

本文件是本仓库 **唯一的共享代理规则来源**，Codex（AGENTS.md）与 Claude Code（CLAUDE.md `@AGENTS.md` 导入）共同遵守。
`EIT`、`BTD` 是项目代号，不要翻译、不要展开成全称。

## 1. 任务与赛题

赛题六：AI 驱动的 PCB 原理图自动解析（芯粒研究中心 & 比昂芯）。

输入：单张原理图 PNG。输出：一个 JSON，含三部分（字段定义见仓库根 `README.md`）：

- `components`：`Name`（位号）、`type`、`value`、`bbox [x_min, y_min, x_max, y_max]`
- `pins`：`pinname`、`point {x, y}`
- `nets`：`hyperGraph`（如 `(U1.7)`）、`edges`（若干坐标点组成的线段）

评分由 Component F1、Pin F1、Net-hypergraph F1、Net-line F1 组成；权重 0.3 / 0.25 / 0.35 / 0.1。
**权重与容差以 `research/` 下的赛题文本为准（spec §五、附录 4），引用前先读原文，不要凭记忆报数字。**

## 2. 目录分工（不要越界改动）

| 目录 | 角色 | 改动规则 |
|---|---|---|
| `non404/` | 早期端到端流程（DiagramNet baseline），本地评分约 22/100 | **只读参考，不要修改** |
| `component_lab/` | 当前主线：元器件检测与分类 | 当前主要改动区 |
| `Data/` | 额外训练数据（PCBnet300、open_schematics、yolo_train） | 新增数据放这里 |
| `research/` | 笔记、审计、调研、汇报讲稿 | 新笔记写成 `research/<主题>_<YYYYMMDD>.md` |
| `weights/` | 仓库级权重存放 | 模型权重不要提交大文件 |

`component_lab/` 实验报告统一写到 `component_lab/outputs/<实验名>/REPORT.md`。

## 3. component_lab 现状（截至 2026-10-16，先读 `component_lab/START_HERE.md`）

```text
原理图 → YOLO26-Nano（单类 component 框）
           ├→ 符号裁剪 → ResNet18 → 27 个语义家族
           └→ 符号 + 周边裁剪 → Qwen（可选旁路）→ 位号 / 数值 / 型号候选
```

- YOLO 第 4 轮：`weights/pcbnet_component_yolo26n_review_round_004.pt`，P 88.83% / R 84.00% / mAP50 86.20%
- ResNet18：`weights/resnet18_component_balanced_v1.pt`，27 类 × 256 份；**弱标签/AI 银标**一致率 96.74%、宏 F1 73.94%，不是人工真值
- 目标双 95% **尚未达到**；引脚、网络、官方格式导出、端到端评分均未构建
- 尚未在官方风格用例上按赛题规则评分——报告指标时说明用的是哪套口径

## 4. 环境与命令

建议 Python 3.11，在 `component_lab/` 下：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[vision,api]"
```

- 本机 GPU：4060 Ti 用 `--device 0` / `cuda:0`；3050 Ti 在 PyTorch 里是 `cuda:1`（与 nvidia-smi 序号不同）
- 训练：`component-lab train-yolo --data data/yolo/pcbnet_component_v1/dataset.yaml --output runs --device 0`
- 分类：`component-lab train-resnet --data data/resnet --output runs/resnet --device cuda:0`
- 推理：`component-lab infer --image <png> --yolo <pt> --resnet <pt> --device 0`（大图加 `--tile-size 1024 --tile-overlap 0.25 --tile-max-tiles 100`）
- 人工复核页面：<http://127.0.0.1:8768/>（`启动元件复核.ps1` / `开始检查元件.cmd`）

## 5. 硬约束

1. **`qwenapi.txt` 是明文 API 凭据**：不提交、不外发、不贴进任何文档、不打印到日志。需要密钥时用环境变量。
2. **数值字符串精确照抄，禁止归一化**：`10kΩ`、`10K`、`0.01 μF`、`100 nF` 是不同字符串。图上没有就输出 `null`。
3. **按整张原理图划分 train/val 再裁剪**，避免同图泄漏。
4. `disable-windows-update*.ps1` / `revert-windows-update*.ps1` / `wu-*.txt` 与赛题无关，**不要动**。
5. 不要用 AI 银标 / 弱标签冒充人工真值；报告里必须标明标签来源。
6. 不要静默更改既有结论数字；新结果与旧结果并存，注明来源与权重文件名。

## 6. 双代理协作规范

- **交接文档**：跨代理协作写 `research/handoff.md`（或对应主题文件），写清「已做什么 / 未做什么 / 下一步 / 涉及的权重与命令」。另一个代理接手先读它。
- **并行改动同一仓库时用 git worktree 隔离**（`codex --worktree`、`claude -w`），不要两个代理同时在一个工作树里改同一批文件。
- **不要回退对方未提交的改动**；提交前先 `git status`，发现不属于自己的改动要保留并说明。
- 报告结论时附「证据」（命令、文件路径、日志），不要写无法复现的指标。

## 7. 语言与术语

与用户交流用中文；技术名词保留英文原词（bbox、recall、hyperGraph、golden case、union-find）。

固定译法：元器件 component ｜ 位号 designator / Name ｜ 引脚 pin ｜ 网络 net ｜ 标注 annotation ｜ 边界框 bbox ｜ 连通性 connectivity ｜ 召回率 recall ｜ 精确率 precision ｜ 切片推理 tiled inference ｜ 金标准用例 golden case。
