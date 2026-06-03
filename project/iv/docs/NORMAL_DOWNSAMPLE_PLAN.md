# Normal Downsample Dataset Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 `dataset_task_avg_isc_ge_1p5_no_data_error` 重新生成一套新数据集，按 `task_id + inverter_id` 批次做筛选，只对纯 `normal` 批次做下采样，把最终保留数据中的 `normal` 数量压到接近 `break_shadow`，默认目标按 `300` 条执行。

**Architecture:** 先按 `task_id + inverter_id` 聚合批次，凡是批次里出现任何非 `normal`，该批次就整批保留；只从“整批全是 `normal`”的批次里做可复现的整批下采样。输出新的 `data.csv + labels.csv + summary.json`，同时在 `data.csv` 中补充 `isc`、`max_isc`、`dIsc_max`、`voc` 四个分析字段。整个过程不改源数据集，只生成一个新的兄弟目录，方便回滚和重复试验。

**Tech Stack:** Python、pandas、现有主工程数据集 CSV、主工程 `docs/` 文档、`unittest`

---

## 背景

当前主工程里已有这套训练数据：

- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\data.csv`
- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\labels.csv`

这套数据已经满足：

- `task avg Isc >= 1.5A`
- 排除 `data_error`

但当前问题是：

- `normal` 数量明显偏多
- 训练时类别不平衡
- 需要一套新的、更适合训练或调试的数据集

## 当前确认下来的执行口径

1. 批次键固定为 `task_id + inverter_id`
2. 只要一批里出现任何非 `normal`，这一整批都保留
3. 只从“整批全是 `normal`”的批次里做整批删除
4. 目标 `normal` 数量按 `300` 执行
5. 新 `data.csv` 里需要新增：
   - `isc`
   - `max_isc`
   - `dIsc_max`
   - `voc`

## 本次计划默认参数

这部分是给你直接修改的。你改完后，我就按这份计划执行。

### 输入数据集

- 源数据目录：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error`
- 源 `data.csv`：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\data.csv`
- 源 `labels.csv`：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\labels.csv`

### 输出数据集

- 目标输出目录：`D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300`

### 下采样规则

- 批次键使用 `task_id + inverter_id`
- 只对“整批全是 `normal`”的批次做下采样
- 含异常批次全部保留
- 目标 `normal_target_count = 300`
- 默认 `random_seed = 42`

### 需要新增到 data.csv 的分析列

- `isc`
  - 定义：当前组串曲线的 `Isc`
- `max_isc`
  - 定义：同一 `task_id + inverter_id` 批次内，各组串 `Isc` 的最大值
- `dIsc_max`
  - 定义：`(isc - max_isc) / max_isc`
- `voc`
  - 定义：当前组串曲线的 `Voc`

### 输出文件

新数据集至少包含：

- `data.csv`
- `labels.csv`
- `summary.json`
- `label_counts_before.csv`
- `label_counts_after.csv`

## 文件结构

本次落地时建议改动这些主工程文件。

### 新增文件

- `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\downsample_training_dataset.py`
- `D:\sigen-project2\sigen-iv-diagnosis\src\run_build_downsampled_training_dataset.py`
- `D:\sigen-project2\sigen-iv-diagnosis\tests\test_downsample_training_dataset.py`

### 修改文件

- `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

## 实现思路

1. 读取源 `data.csv` 和 `labels.csv`
2. 统一标签名口径，把中文/英文 `normal` 都归一成同一类
3. 以 `task_id + inverter_id` 为批次键统计每批的标签构成
4. 整批保留所有含非 `normal` 的批次
5. 只从“整批全是 `normal`”的批次里做整批下采样，直到最终 `normal` 数量接近目标值
6. 用保留下来的 `row_id` 从 `data.csv` 中同步筛出对应样本
7. 在保留后的 `data.csv` 中补充 `isc`、`max_isc`、`dIsc_max`、`voc`
8. 写出新的 `data.csv`、`labels.csv`
9. 输出采样前后计数和摘要信息

## Task 1: 确认标签归一化口径

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\downsample_training_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_downsample_training_dataset.py`

- [ ] **Step 1: 明确 `normal` 的统一口径**

需要统一处理这些标签名：

- `normal`
- `正常`
- `正常 (Normal)`

目标是保证它们在下采样时都被当作同一类。

- [ ] **Step 2: 设计一个最小标签归一化函数**

要求：

- 优先根据 `label_code` 判断
- `label_code` 不可靠时，再根据 `label_name` 兜底
- 输出统一标签名，例如 `normal / break_shadow / low_isc / low_voc`

- [ ] **Step 3: 写测试覆盖常见输入**

至少覆盖：

- `label_code=0`
- `label_name=normal`
- `label_name=正常`
- `label_name=正常 (Normal)`
- `label_code=99` 对应 `break_shadow`

## Task 2: 实现批次级 normal 下采样逻辑

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\downsample_training_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_downsample_training_dataset.py`

- [ ] **Step 1: 读取源数据**

输入文件：

- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\data.csv`
- `D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\labels.csv`

- [ ] **Step 2: 以 `task_id + inverter_id` 统计批次**

要求：

- 每一行都能映射到自己的批次
- 每个批次要统计：
  - 总行数
  - `normal` 行数
  - 是否包含非 `normal`

- [ ] **Step 3: 决定保留哪些批次**

要求：

- 含异常批次全部保留
- 只从“整批全是 `normal`”的批次里做整批删除
- 优先统计含异常批次自带的 `normal` 数量
- 再从纯 `normal` 批次里补到目标附近

- [ ] **Step 4: 用保留的 `row_id` 同步筛 `data.csv`**

要求：

- `labels.csv` 和 `data.csv` 行数一致
- `row_id` 一一对应
- 不出现脏数据或丢失数据

## Task 3: 在 data.csv 中新增分析列

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\analysis\downsample_training_dataset.py`
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_downsample_training_dataset.py`

- [ ] **Step 1: 计算每行的 `isc`**

要求：

- 从 `current` 曲线直接取最大值

- [ ] **Step 2: 计算每行的 `voc`**

要求：

- 从 `voltage` 曲线直接取最大值

- [ ] **Step 3: 计算每批的 `max_isc`**

要求：

- 按 `task_id + inverter_id` 分组
- 每组取 `isc` 最大值

- [ ] **Step 4: 计算每行的 `dIsc_max`**

公式：

```text
dIsc_max = (isc - max_isc) / max_isc
```

- [ ] **Step 5: 把这些字段写入新的 `data.csv`**

最终新 `data.csv` 至少新增：

- `isc`
- `max_isc`
- `dIsc_max`
- `voc`

## Task 4: 提供命令行入口

**Files:**
- Create: `D:\sigen-project2\sigen-iv-diagnosis\src\run_build_downsampled_training_dataset.py`
- Modify: `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

- [ ] **Step 1: 新增 CLI 入口**

建议参数：

- `--processed-csv`
- `--labels-csv`
- `--output-dir`
- `--normal-target-count`
- `--random-seed`

- [ ] **Step 2: 设定默认值**

默认值建议：

- 输入：`dataset_task_avg_isc_ge_1p5_no_data_error`
- 输出：`dataset_task_avg_isc_ge_1p5_no_data_error_normal300`
- `normal-target-count=300`
- `random-seed=42`

- [ ] **Step 3: 在文档里补充命令**

示例命令应写入：

- `D:\sigen-project2\sigen-iv-diagnosis\docs\LABELED_CASE_TOOLS_GUIDE.md`

## Task 5: 验证新数据集

**Files:**
- Test: `D:\sigen-project2\sigen-iv-diagnosis\tests\test_downsample_training_dataset.py`

- [ ] **Step 1: 运行单测**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -m unittest D:\sigen-project2\sigen-iv-diagnosis\tests\test_downsample_training_dataset.py -v
```

- [ ] **Step 2: 实际生成新数据集**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 D:\sigen-project2\sigen-iv-diagnosis\src\run_build_downsampled_training_dataset.py --processed-csv D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\data.csv --labels-csv D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error\labels.csv --output-dir D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300 --normal-target-count 300 --random-seed 42
```

- [ ] **Step 3: 检查输出**

需要人工确认：

- `normal` 是否接近 `300`
- `break_shadow` 等其他类别是否原样保留
- 含异常批次是否被整批保留
- `data.csv` 与 `labels.csv` 的 `row_id` 是否一致
- `data.csv` 中是否新增了：
  - `isc`
  - `max_isc`
  - `dIsc_max`
  - `voc`

## Task 6: 用新数据集跑全量诊断

**Files:**
- Reference: `D:\sigen-project2\sigen-iv-diagnosis\src\run_iv_diagnosis_debug.py`

- [ ] **Step 1: 用新数据集跑全量**

建议命令：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 D:\sigen-project2\sigen-iv-diagnosis\src\run_iv_diagnosis_debug.py --processed-csv D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\data.csv --labels-csv D:\sigen-project2\sigen-iv-diagnosis\datasets\augmented_dataset_eval\dataset_task_avg_isc_ge_1p5_no_data_error_normal300\labels.csv --output-dir D:\sigen-project2\sigen-iv-diagnosis-analyze\debug_diagnosis_full_filtered_normal300 --workers 8
```

- [ ] **Step 2: 检查输出文件**

重点看：

- `diagnosis_results.csv`
- `summary.json`
- 评估图

## 你可以直接修改的地方

如果你准备改完再让我执行，优先改这里：

1. `目标输出目录`
2. `normal_target_count`
3. `random_seed`
4. 最后跑诊断时的 `output-dir`
5. 如果你后面想调整批次键，再改 `task_id + inverter_id`

## 当前默认建议

如果你不想再改复杂逻辑，建议先按这版执行：

- 源数据：`dataset_task_avg_isc_ge_1p5_no_data_error`
- 新数据：`dataset_task_avg_isc_ge_1p5_no_data_error_normal300`
- `normal_target_count = 300`
- `random_seed = 42`
- 批次键：`task_id + inverter_id`
- 含异常批次：整批保留

## 执行前确认

等你修改完这份计划文档后，你只需要告诉我：

- “按这份 plan 执行”

我就会按文档里的最终参数，去主工程里生成新数据集并验证。
