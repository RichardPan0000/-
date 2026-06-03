# Normal300 Relabel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 `dataset_task_avg_isc_ge_1p5_no_data_error_normal300` ，生成新的label.csv文件：先把 `isc < 1.5A` 的样本统一改成 `data_error`，再把剩余样本里 `dIsc_max <= -0.16` 且原标签为 `normal` 的样本改成 `low_isc`。

**Architecture:** 输入沿用当前的 `normal300` 数据集，`data.csv` 保持原始数据行不删不增；`labels.csv` 在复制后按规则批量改写 `label_code / label_name / is_abnormal`。整个过程不改源目录，只生成新的兄弟文件，并输出重标摘要和前后标签分布，方便回滚和比对。

**Tech Stack:** Python、pandas、主工程 CSV 数据集、主工程 `docs/` 文档、`unittest`

---

## 背景

当前主工程里已经有这套训练数据：

- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\data.csv`
- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\labels.csv`

并且这套 `data.csv` 已经补充了分析列：

- `isc`
- `max_isc`
- `dIsc_max`
- `voc`

现在需要在这套数据集基础上，按规则重整标签，而不是重新筛样本：

1. `isc < 1.5A` 的样本，统一改成 `data_error`
2. 对于 `isc >= 1.5A` 的剩余样本，如果 `dIsc_max <= -0.16`
   - 且原标签是 `normal`
   - 就改成 `low_isc`（标签编码 `2`）
3. 已经改成 `data_error` 的样本，不再参与第二条规则

## 当前确认下来的执行口径

1. 本次操作对象是整套 `dataset_task_avg_isc_ge_1p5_no_data_error_normal300`
2. 不删除样本，只改标签
3. `isc < 1.5A` 优先级最高，先打成 `data_error`
4. 只有未改成 `data_error` 的样本，才继续判 `dIsc_max <= -0.16`
5. 第二条规则只改原本就是 `normal` 的样本
6. `low_isc` 目标编码固定为 `2`
8. `data_error` 目标编码固定为 `-1`

## 本次计划默认参数

### 输入数据集

- 源数据目录：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300`
- 源 `data.csv`：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\data.csv`
- 源 `labels.csv`：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\labels.csv`

### 输出数据集

- 目标输出目录：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300`

### 重标规则

- 规则 1：`isc < 1.5` -> `data_error`
- 规则 2：`isc >= 1.5` 且 `dIsc_max <= -0.16` 且原标签为 `normal` -> `low_isc`

### 需要同步修改的标签字段

- `label_code`
- `label_name`
- `is_abnormal`

建议同步约定：

- `data_error`
  - `label_code = -1`
  - `label_name = data_error`
  - `is_abnormal = True`
- `low_isc`
  - `label_code = 2`
  - `label_name = low_isc`
  - `is_abnormal = True`

### 输出文件

新数据集至少包含：

- `data.csv`
- `labels.csv`
- `summary.json`
- `label_counts_before.csv`
- `label_counts_after.csv`
- `relabel_changes.csv`

## 文件结构

本次落地时建议改动这些主工程文件。

### 新增文件

- `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\relabel_training_dataset.py`
- `D:\sigen-project2\sigen-iv-diagnosis\src\run_relabel_training_dataset.py`
- `D:\sigen-project2\sigen-iv-diagnosis\tests\test_relabel_training_dataset.py`

### 修改文件

- `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

## 实现思路

1. 读取源 `data.csv` 和 `labels.csv`
2. 用 `row_id` 对齐两份数据
3. 先检查 `data.csv` 是否已存在 `isc` 和 `dIsc_max`
4. 对每一行应用规则 1：`isc < 1.5` -> `data_error`
5. 对剩余行应用规则 2：`isc >= 1.5` 且 `dIsc_max <= -0.16` 且原标签为 `normal` -> `low_isc`
6. 回写新的 `labels.csv`
7. 原样复制 `data.csv`
8. 输出标签变更明细和前后分布统计

## Task 1: 明确标签归一化与重标规则

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\relabel_training_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_relabel_training_dataset.py`

- [ ] **Step 1: 统一识别原始 `normal` 标签**

至少覆盖：

- `label_code = 0`
- `label_name = normal`
- `label_name = 正常`
- `label_name = 正常 (Normal)`

- [ ] **Step 2: 写出两条重标规则的纯函数**

要求：

- 规则 1 先执行
- 规则 2 只在规则 1 未命中时执行
- 输出目标标签名和目标编码

- [ ] **Step 3: 写测试覆盖边界值**

至少覆盖：

- `isc = 1.49`
- `isc = 1.50`
- `dIsc_max = -0.16`
- `dIsc_max = -0.159`
- 原标签不是 `normal` 时不应改成 `low_isc`

## Task 2: 实现数据集重标逻辑

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\relabel_training_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_relabel_training_dataset.py`

- [ ] **Step 1: 读取并对齐 `data.csv` 和 `labels.csv`**

要求：

- 基于 `row_id` 对齐
- 发现 `row_id` 不一致时直接报错

- [ ] **Step 2: 校验必需列**

至少检查：

- `row_id`
- `isc`
- `dIsc_max`
- `label_code`
- `label_name`

- [ ] **Step 3: 应用规则 1**

逻辑：

- `isc < 1.5`
- 直接改成 `data_error`

- [ ] **Step 4: 应用规则 2**

逻辑：

- `isc >= 1.5`
- `dIsc_max <= -0.16`
- 原标签是 `normal`
- 改成 `low_isc`

- [ ] **Step 5: 同步更新 `is_abnormal`**

规则：

- `label_code == 0` -> `False`
- 其他编码 -> `True`

- [ ] **Step 6: 写出变更明细**

`relabel_changes.csv` 至少包含：

- `row_id`
- `task_id`
- `inverter_id`
- `string_id`
- `old_label_code`
- `old_label_name`
- `new_label_code`
- `new_label_name`
- `isc`
- `dIsc_max`
- `change_reason`

## Task 3: 提供命令行入口

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\run_relabel_training_dataset.py`
- Modify: `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

- [ ] **Step 1: 新增 CLI 入口**

建议参数：

- `--processed-csv`
- `--labels-csv`
- `--output-dir`
- `--isc-threshold`
- `--disc-max-threshold`

- [ ] **Step 2: 设定默认值**

默认值建议：

- 输入：`dataset_task_avg_isc_ge_1p5_no_data_error_normal300`
- 输出：`dataset_task_avg_isc_ge_1p5_no_data_error_normal300_relabel_isc15_disc016`
- `isc-threshold = 1.5`
- `disc-max-threshold = -0.16`

- [ ] **Step 3: 在文档里补充命令**

写入：

- `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

## Task 4: 验证新数据集

**Files:**
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_relabel_training_dataset.py`

- [ ] **Step 1: 运行单测**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -m unittest D:\sigen-project2\sigen-iv-diagnosis\tests\test_relabel_training_dataset.py -v
```

- [ ] **Step 2: 实际生成新数据集**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 D:\sigen-project2\sigen-iv-diagnosis\src\run_relabel_training_dataset.py --processed-csv D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\data.csv --labels-csv D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\labels.csv --output-dir D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300_relabel_isc15_disc016 --isc-threshold 1.5 --disc-max-threshold -0.16
```

- [ ] **Step 3: 检查输出**

需要人工确认：

- `data.csv` 行数不变
- `labels.csv` 行数不变
- `isc < 1.5` 的样本是否全部变成 `data_error`
- `dIsc_max <= -0.16` 且原标签为 `normal` 的样本是否改成 `low_isc`
- 已变成 `data_error` 的样本是否没有再被改成 `low_isc`
- `relabel_changes.csv` 是否记录完整



## 你可以直接修改的地方

如果你准备改完再让我执行，优先改这里：

1. `目标输出目录`
2. `isc-threshold`
3. `disc-max-threshold`
4. 最后跑诊断时的 `output-dir`

## 当前默认建议

如果你不想再改复杂逻辑，建议先按这版执行：

- 源数据：`dataset_task_avg_isc_ge_1p5_no_data_error_normal300`

  

- `isc-threshold = 1.5`

- `disc-max-threshold = -0.16`

## 执行前确认

等你修改完这份计划文档后，你只需要告诉我：

- “按这份 relabel plan 执行”

我就会按文档里的最终参数，去主工程里生成这套新的重标数据集并验证。
