# Voc Arithmetic Outlier 低 Isc 过滤计划

## 背景

当前 `RulePredictor4._analyze_total_voc()` 在做同组 `Voc_total` 等差参考拟合时，会直接使用 `group_stats["voc_total_by_string"]` 里的所有组串。

在实际数据里，如果同一组中存在 `Isc` 明显偏低的组串，例如 `< 0.2A`，这些组串通常已经接近失效或测量退化状态。它们的 `Voc_total` 继续参与等差拟合时，容易把参考线拖偏，进而影响正常组串或轻微异常组串的 `low_Voc` 判定。

## 目标

在不改动现有 `Voc` 判定主体逻辑的前提下，增加一个前置过滤：

- 同组中 `Isc < 0.2A` 的组串，不参与 `Voc_total` 等差参考拟合
- 这些低 `Isc` 组串自身仍然可以被判定，只是不再用于拟合参考线
- 阈值配置化，默认值为 `0.2A`

## 改动范围

### 1. `rule_predictor4.py`

位置：
- `D:\sigen-project2\sigen-iv-diagnosis\src\diagnosis\predictors\rule_predictor4.py`

计划：
- 在 `_analyze_total_voc()` 中读取 `group_stats["isc_by_string"]`
- 新增 `voc_ap_min_fit_isc_abs` 阈值
- 构造拟合输入时，排除 `Isc < voc_ap_min_fit_isc_abs` 的组串
- 如果当前串也被排除：
  - 仍然使用拟合得到的 `best_a / best_d` 对当前串单独计算 `nearest_ref / residual / signed_residual`
- 在返回的 `voc_total_analysis` 中加入调试字段，便于后续排查：
  - `fit_input_string_ids`
  - `fit_excluded_low_isc_string_ids`
  - `fit_min_isc_threshold`

### 2. `diagnosis_config.yaml`

位置：
- `D:\sigen-project2\sigen-iv-diagnosis\src\config\diagnosis_config.yaml`

计划：
- 在 `rule_predictor4` 下新增：
  - `voc_ap_min_fit_isc_abs: 0.2`

### 3. 单元测试

位置：
- `D:\sigen-project2\sigen-iv-diagnosis\tests\test_rule_predictor4_logic.py`

计划：
- 新增测试，验证低 `Isc` 组串会被从 `Voc` 拟合集合中排除
- 验证当前串即使被排除，也仍然会得到 `nearest_ref / residual / outlier` 结果

## 验证方式

执行：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -m unittest D:\sigen-project2\sigen-iv-diagnosis\tests\test_rule_predictor4_logic.py -v
```

检查点：

- 现有 `low_voc` 相关测试保持通过
- 新增测试通过
- `voc_total_analysis` 中能看到低 `Isc` 排除信息

## 预期效果

- 同组内 `Isc` 极低的退化串不再干扰 `Voc_total` 的等差参考拟合
- `low_Voc` 判定在“组内混有近失效串”的场景下更加稳定
- 后续如果你想调阈值，只需要改配置，不需要再改代码
