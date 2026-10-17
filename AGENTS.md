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

## 3. component_lab 现状（截至 2026-10-18，先读 `component_lab/START_HERE.md` 与 `research/status_20261018.md`）

```text
原理图 → YOLO26-Nano（单类 component 框）
           ├→ 符号裁剪 → ResNet18 → 27 个语义家族
           └→ 符号 + 周边裁剪 → Qwen（可选旁路）→ 位号 / 数值 / 型号候选
```

- **当前默认生产配置**：YOLO `weights/pcbnet_component_yolo26n_review_round_004.pt` + ResNet `weights/resnet18_component_balanced_v1.pt` + `--association identity-v1 --ocr-spans --display-names --contest-export` + 局部 v6 补读（`--no-local-v6` 关闭）。
- 固定 dev20 严格 Component F1 **17.42% / 170TP**（1040 预测 / 931 GT）；再接 `recover-identities` 未命名恢复为 **17.77% / 177TP**。
- **官方格式导出（components-only）与官方规则评分器已建成**（`component-lab score`，GT-vs-GT 自检 100.00）；`pins`/`nets` 仍恒为空，端到端加权总分未测。
- 第 5 轮权重 `review_round_005.pt` 已训练：旧 30 图 val mAP50 87.25%，但严格 dev20 只有 17.32%/169TP → **未替换默认**。
- **口径警告（最重要）**：仓库里存在 1040 框、632 框、76 框三套都叫“严格 ComponentF1”的数字，框集与权重不同，**不可横向比较**。引用时必须同时写「框数 + 权重文件名 + 来源报告」。
- ResNet18 27 类；**弱标签/AI 银标**一致率 96.74%、宏 F1 73.94%，不是人工真值。
- 目标双 95% 尚未达到；引脚、网络仍未完成。已知天花板与协议风险见 `research/status_20261018.md`。

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

## 6. 多代理协作与版本控制

**仓库**：根目录已是 git 仓库，`main` 跟踪 `origin/main`（<https://github.com/misakamikoto23/non404_component>）。

- **入库范围**：`component_lab/src|scripts|tests|configs`、`component_lab/*.md`、`research/`、`AGENTS.md`、`README.md`、`.gitignore`、`.gitattributes`。
- **不入库**：`outputs/`、`runs/`、`data/`、`weights/`、`Data/`、`non404/`（另有独立仓库）、`*.pt|onnx|tar|zip|pdf|png`、`qwenapi.txt`、`.env`。规则见根 `.gitignore`，**不要用 `git add -f` 绕过**。
- **提交前自检**：`git diff --cached --name-only`，确认没有 `env` / `api` / `weights` / `outputs` 混入。
- **同一工作树只允许一个写入者**。并行实验用 worktree 隔离：`git worktree add ..\wt-<实验名> -b exp/<实验名>`；不要两个代理在同一工作树改同一批文件。
- **提交信息**：`exp(<实验名>): <结论> <旧分>-> <新分>` 或 `fix|docs|chore(<范围>): ...`；一个实验一个 commit，便于单独回退。
- **有效状态打 tag**：`git tag -a exp/<实验名>-<YYYYMMDD> -m "<指标 + 权重文件名>"`，再 `git push origin main --tags`。
- **不要回退对方未提交的改动**；提交前先 `git status`，发现不属于自己的改动要保留并说明。
- **交接文档**：跨代理协作写 `research/handoff_<主题>_<YYYYMMDD>.md`，写清「已做什么 / 未做什么 / 下一步 / 涉及的权重与命令」。
- **训练环境必须可复现**：记录训练实际使用的解释器路径与 `pip freeze`（`torch` / `ultralytics` / `onnxruntime` 版本）。注意：项目内 `.venv-*` 当前**没有 torch**，训练依赖外部解释器，新环境必须显式记录。
- **产物不可回退**：`outputs/`、权重、数据不入库，靠目录命名 + `STATUS.json` 里的 SHA256 追溯，不要指望 `git checkout` 恢复实验结果。
- 报告结论时附「证据」（命令、文件路径、日志、SHA256），不要写无法复现的指标。

## 7. 语言与术语

与用户交流用中文；技术名词保留英文原词（bbox、recall、hyperGraph、golden case、union-find）。

固定译法：元器件 component ｜ 位号 designator / Name ｜ 引脚 pin ｜ 网络 net ｜ 标注 annotation ｜ 边界框 bbox ｜ 连通性 connectivity ｜ 召回率 recall ｜ 精确率 precision ｜ 切片推理 tiled inference ｜ 金标准用例 golden case。
