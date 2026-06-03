# Low Voc Refresh Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于当前 `dataset_task_avg_isc_ge_1p5_no_data_error_normal300` 数据集，重新生成其中已有 `low_voc` 标签对应的故障串曲线，并用同一 task 内的正常串作为 donor，定点替换这 16 条 `low_voc` 数据。

**Architecture:** `low_voc` 的生成逻辑继续复用 `D:\sigen-project2\sigen-iv-diagnosis\.worktrees\dataset-eval\src\analysis\four_fault_demo_dataset.py` 里当前最新的 `_apply_low_voc(...)` 相关函数，不在主工程里重复造一套故障生成逻辑。主工程新增一个 refresh 工具，只识别目标数据集中的 `low_voc` 行，对每个 `task_id + inverter_id` 就地使用该 task 自己的正常串重建 low_voc 曲线，先输出预览图，确认后再原位替换 `data.csv / labels.csv` 中对应的 `low_voc` 行。

**Tech Stack:** Python、pandas、主工程 CSV 数据集、`dataset-eval` 的 synthetic 生成逻辑、`unittest`

---

## 背景

当前目标数据集是：

- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\data.csv`
- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\labels.csv`

当前其中 `low_voc` 行是：

- 总数 `16` 条
- 分布在 `8` 个 task
- 每个 task `2` 条 `low_voc`
- 当前统一在 `inverter_id = 120A13AT0233`

当前识别到的 `low_voc` task 是：

- `27348`
- `27353`
- `27358`
- `27363`
- `27368`
- `27373`
- `27378`
- `27383`

用户已经调整了 `four_fault_demo_dataset.py` 里的 `low_voc` 参数，因此这次目标不是再设计新规则，而是：

1. 使用当前最新的 `_apply_low_voc(...)` 生成口径
2. 用每个 `low_voc task` 自己内部的正常串作为 donor
3. 只替换这 16 条 `low_voc` 故障串，不替换同 task 内正常串

## 当前确认下来的执行口径

1. 本次只处理标签为 `low_voc` 的行
2. 不删除 task，不替换整批 task
3. 每个 `low_voc` task 的 donor 仅来自该 task 自己内部的正常串
4. 每个 `low_voc` 行保持原来的：
   - `row_id`
   - `task_id`
   - `inverter_id`
   - `mppt_id`
   - `string_id`
   - `timestamp`
5. 只更新这些行的故障曲线数据和必要的标注说明
6. 标签仍保持：
   - `label_code = 3`
   - `label_name = low_voc`
7. 替换前先生成图片给人检查
8. 通过后再回写到目标数据集

## 本次计划默认参数

### 输入数据集

- 源 `data.csv`：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\data.csv`
- 源 `labels.csv`：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\labels.csv`

### 故障生成逻辑来源

- `D:\sigen-project2\sigen-iv-diagnosis\.worktrees\dataset-eval\src\analysis\four_fault_demo_dataset.py`

重点复用：

- `_build_clean_baseline_curve_from_donor(...)`
- `_pick_clean_donor_row(...)`
- `_apply_low_voc(...)`
- `apply_fault_transform(...)`
- `build_task_plot(...)`

### 预览输出目录

- `D:\sigen-project2\sigen-iv-diagnosis-analyze\augmented_dataset_eval\low_voc_refresh_<timestamp>`

### 目标输出

默认回写到原数据集目录：

- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300`

### 输出文件

至少包含：

- 预览图目录
- `low_voc_refresh_summary.json`
- `low_voc_refresh_preview.csv`
- 替换前备份目录

## 文件结构

本次落地时建议改动这些主工程文件。

### 新增文件

- `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\low_voc_refresh_tools.py`
- `D:\sigen-project2\sigen-iv-diagnosis\src\run_refresh_low_voc_dataset.py`
- `D:\sigen-project2\sigen-iv-diagnosis\tests\test_low_voc_refresh_tools.py`

### 修改文件

- `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

## 实现思路

1. 读取当前 `normal300` 的 `data.csv + labels.csv`
2. 找出所有 `low_voc` 行
3. 以 `task_id + inverter_id` 分组，收集这些 task 内的正常串
4. 对每条 `low_voc` 行：
   - 从同 task 正常串里挑 donor
   - 先重建 clean baseline
   - 再调用当前 `_apply_low_voc(...)` 生成新的 low_voc 曲线
5. 生成每个 task 的预览图
6. 人工确认图后，再回写这 16 条 `low_voc` 对应的 `data.csv`
7. `labels.csv` 保持 low_voc 标签不变，但可以更新 `notes`
8. 输出替换摘要和备份

## Task 1: 明确 low_voc 定点刷新对象

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\low_voc_refresh_tools.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_low_voc_refresh_tools.py`

- [ ] **Step 1: 读取目标数据集并筛出 low_voc 行**

要求：

- 只认 `label_code = 3` 或标准化后 `label_name = low_voc`
- 输出这 16 条行的明细

- [ ] **Step 2: 统计每个 low_voc task 内的正常串**

要求：

- 批次键固定为 `task_id + inverter_id`
- 对每个目标 task 检查：
  - `low_voc` 行数
  - `normal` 行数
  - donor 是否充足

- [ ] **Step 3: 写测试覆盖 task 内 donor 选择**

至少覆盖：

- task 内有正常串时正常返回
- task 内没有正常串时报错
- 只筛到 low_voc 行时能正确返回目标 row

## Task 2: 复用 four_fault_demo_dataset.py 的 low_voc 生成逻辑

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\low_voc_refresh_tools.py`
- Reference: `D:\sigen-project2\sigen-iv-diagnosis\.worktrees\dataset-eval\src\analysis\four_fault_demo_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_low_voc_refresh_tools.py`

- [ ] **Step 1: 从 dataset-eval 侧导入当前 low_voc 生成函数**

要求：

- 不在主工程重新手写 `_apply_low_voc`
- 直接复用当前最新版本

- [ ] **Step 2: 为单条 low_voc 行生成替换曲线**

流程：

- donor normal row
- clean baseline
- `_apply_low_voc(...)`

- [ ] **Step 3: 保持元数据不变**

替换时保持：

- `row_id`
- `task_id`
- `inverter_id`
- `mppt_id`
- `string_id`
- `timestamp`

只改：

- `voltage`
- `current`
- 可选 `notes`

- [ ] **Step 4: 写测试覆盖生成结果**

至少检查：

- 新曲线点数和原曲线一致
- 新曲线仍单调
- `Voc` 低于 donor 正常串
- `Isc` 不出现明显 low_isc 风格塌陷

## Task 3: 生成预览图并支持人工确认

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\low_voc_refresh_tools.py`
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\run_refresh_low_voc_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_low_voc_refresh_tools.py`

- [ ] **Step 1: 输出 low_voc preview CSV**

至少包含：

- `row_id`
- `task_id`
- `inverter_id`
- `string_id`
- `old_voc`
- `new_voc`
- `old_isc`
- `new_isc`
- `plot_path`

- [ ] **Step 2: 为每个 task 输出预览图**

要求：

- 同时画 task 内正常串和刷新后的 low_voc 串
- 文件名带 `task_id`
- 目录结构清晰，方便人工逐 task 看图

- [ ] **Step 3: CLI 默认只生成预览，不替换**

要求：

- 先生成图片
- 不加 `--apply-replace` 时不改数据集

## Task 4: 确认后替换数据集中的 low_voc 行

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\low_voc_refresh_tools.py`
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\run_refresh_low_voc_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_low_voc_refresh_tools.py`

- [ ] **Step 1: 备份原始 data.csv / labels.csv**

要求：

- 在目标目录下创建：
  - `backup_before_low_voc_refresh_<timestamp>`

- [ ] **Step 2: 仅替换 low_voc 对应行**

要求：

- 不改 normal 行
- 不改其他 fault 行
- 行数不变
- `row_id` 不变

- [ ] **Step 3: labels.csv 保持 low_voc 标签**

要求：

- `label_code = 3`
- `label_name = low_voc`
- 可选更新 `notes`

- [ ] **Step 4: 输出替换摘要**

写出：

- `low_voc_refresh_summary.json`

至少包含：

- 替换行数
- 覆盖 task 数
- 备份目录
- 预览目录

## Task 5: 提供命令行入口和文档

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\run_refresh_low_voc_dataset.py`
- Modify: `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

- [ ] **Step 1: 新增 CLI 入口**

建议参数：

- `--python-exe`
- `--dataset-eval-src`
- `--output-root`
- `--target-data-csv`
- `--target-labels-csv`
- `--apply-replace`

- [ ] **Step 2: 默认先只生成图**

要求：

- 运行脚本不加 `--apply-replace` 时，只输出 preview

- [ ] **Step 3: 文档写清两步法**

文档要包含：

1. 先看图
2. 确认后再替换

## Task 6: 验证

**Files:**
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_low_voc_refresh_tools.py`

- [ ] **Step 1: 跑单测**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -m unittest D:\sigen-project2\sigen-iv-diagnosis\tests\test_low_voc_refresh_tools.py -v
```

- [ ] **Step 2: 先只生成 preview**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 D:\sigen-project2\sigen-iv-diagnosis\src\run_refresh_low_voc_dataset.py
```

- [ ] **Step 3: 人工查看图片**

重点看：

- 每个 low_voc task 的 2 条故障串是否像 low_voc
- 同 task 内正常串是否保持正常风格

- [ ] **Step 4: 确认后执行替换**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 D:\sigen-project2\sigen-iv-diagnosis\src\run_refresh_low_voc_dataset.py --apply-replace
```

## 你可以直接修改的地方

如果你准备改完再让我执行，优先改这里：

1. 目标输出目录
2. preview 根目录
3. 是否回写到原数据集还是兄弟目录
4. 预览图文件夹结构

## 当前默认建议

如果你不想再改复杂逻辑，建议先按这版执行：

- 输入数据集：`dataset_task_avg_isc_ge_1p5_no_data_error_normal300`
- 只刷新已有 `low_voc` 行
- donor 仅来自同 task 正常串
- 先出图，再替换
- 默认回写原数据集

## 执行前确认

等你修改完这份计划文档后，你只需要告诉我：

- “按这份 low_voc refresh plan 执行”

我就会按文档里的最终参数，先生成预览图，再按你的确认流程替换这 16 条 `low_voc` 数据。
