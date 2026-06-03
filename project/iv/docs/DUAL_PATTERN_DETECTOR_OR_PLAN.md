# 双 Pattern Detector OR 合并计划

## 背景

当前主工程里 `RulePredictor4` 只会挂一个 `pattern_detector`：

- 要么是 `legacy`
- 要么是 `step_detect`

你现在希望两个检测器都用上：

- `legacy` 继续保留原来的阶梯/畸变检测能力
- `step_detect` 使用新方法做阶梯检测
- 只要二者中任意一个认为当前组串存在 pattern 异常，就把该组串视为 `break_shadow` 候选

核心诉求是：

- 不是二选一
- 而是两个检测器同时运行
- 结果按 `OR` 逻辑合并

## 目标

在尽量少改现有结构的前提下，新增一个“组合检测器”模式：

- 同时运行 `legacy` 和 `step_detect`
- `step_abnormal` 采用 OR
- `deform_abnormal` 采用 OR
- `reason_candidates` 合并去重
- 调试信息里同时保留两个检测器各自的原始输出，便于后续分析

## 推荐方案

推荐新增一个组合 detector，而不是把 OR 逻辑直接塞进 `RulePredictor4.predict()`。

### 方案 A：新增 `CombinedPatternDetector`（推荐）

新增一个 detector 类，统一实现 `BasePatternDetector.analyze(...)`：

- 内部持有两个子 detector：
  - `LegacyPatternDetector`
  - `StepDetectPatternDetector`
- 对外仍返回一个统一的 `pattern` 结构
- `RulePredictor4` 继续只依赖一个 `pattern_detector`

优点：

- 改动边界清晰
- 不会把 `RulePredictor4` 的业务逻辑再搞复杂
- 后面如果还要接第三个 detector，也更容易扩展

### 方案 B：直接在 `RulePredictor4` 里跑两个 detector

做法：

- `RulePredictor4.predict()` 里手动调用两次 detector
- 然后在 `predict()` 内部合并输出

缺点：

- `RulePredictor4` 会越来越像“流程编排器”
- detector 配置、输入差异、调试输出都会散落在 predictor 里

### 方案 C：只保留一个主 detector，另一个当辅助信号

做法：

- `legacy` 或 `step_detect` 作为主 detector
- 另一个只补充 `reason_candidates`

缺点：

- 不符合“任一满足就算异常”的目标
- 规则口径不够直接

## 最终设计

采用方案 B。

新增：

- `CombinedPatternDetector`

配置里支持一个新类型，例如：

- `type: "legacy_or_step_detect"`

行为如下：

1. 同时运行 `legacy` 和 `step_detect`

2. 保留二者原始输出：
   - `legacy_pattern`
   - `step_detect_pattern`
   
3. 合并统一输出：
   - `step_abnormal = legacy.step_abnormal OR step_detect.step_abnormal`
   - `deform_abnormal = legacy.deform_abnormal `
   
   
   
5. `steps` 字段：
   - 默认以 `step_detect` 为主
   - 如果 `step_detect` 没有有效 step，再回退到 `legacy`
   - 同时保留：
     - `legacy_steps`
     - `step_detect_steps`
   
6. `deformation` 字段：
   - 默认以 `legacy` 为主
   - `step_detect` 的特殊模式信息单独保留

## 输入口径

这部分需要特别明确。

### `legacy`

继续使用当前 `RulePredictor4` 给 pattern detector 的归一化曲线输入。

### `step_detect`

保持你之前的要求：

- 应该吃原始曲线
- 不应再吃组件归一化/0-1 标准化后的曲线

因此组合 detector 需要支持双输入口径：

- 给 `legacy` 传归一化 curve
- 给 `step_detect` 传原始 curve

这意味着实现时大概率要补一层上下文输入，而不是简单复用同一个 `curve` 参数给两个 detector。

## 配置设计

位置：

- `D:\sigen-project2\sigen-iv-diagnosis\src\config\diagnosis_config.yaml`

建议增加：

```yaml
healthy_curve_flow:
  pattern_detector:
    type: "legacy_or_step_detect"
```



## 代码改动范围

### 1. 新增组合 detector

新增文件：

- `D:\sigen-project2\sigen-iv-diagnosis\src\diagnosis\pattern_detectors\combined_pattern_detector.py`

职责：

- 初始化两个子 detector
- 分发输入
- OR 合并输出
- 保留双 detector 的原始结果

### 2. `RulePredictor4` 接入新 detector 类型

修改文件：

- `D:\sigen-project2\sigen-iv-diagnosis\src\diagnosis\predictors\rule_predictor4.py`

计划：

- `_build_pattern_detector()` 支持：
  - `legacy_or_step_detect`
- 必要时补 detector-specific 输入：
  - 归一化 `curve`
  - 原始 `raw_curve`

### 3. 配置更新

修改文件：

- `D:\sigen-project2\sigen-iv-diagnosis\src\config\diagnosis_config.yaml`

计划：

- 增加组合 detector 的示例配置
- 默认是否切换到组合模式，建议先由你手动切换，不直接改默认

### 4. 单元测试

修改文件：

- `D:\sigen-project2\sigen-iv-diagnosis\tests\test_rule_predictor4_logic.py`
- `D:\sigen-project2\sigen-iv-diagnosis\tests\test_predictor_config_selection.py`

至少补这几类测试：

- 配置 `legacy_or_step_detect` 时，orchestrator 能正确构建组合 detector
- `legacy=True, step_detect=False` 时，最终 pattern 异常为真
- `legacy=False, step_detect=True` 时，最终 pattern 异常为真
- `legacy=False, step_detect=False` 时，最终 pattern 异常为假
- 合并输出里同时保留两个 detector 的原始结果

## 验证方式

