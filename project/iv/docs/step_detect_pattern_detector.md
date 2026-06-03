# step_detect_pattern_detector 算法说明

## 1. 这份文档讲什么

`src/diagnosis/pattern_detectors/step_detect_pattern_detector.py` 是当前仓库里用于台阶检测的实现。

这份文档的目标不是复述代码，而是把它拆成可以直接理解的流程：

- 输入是什么
- 先做了哪些预处理
- pre-MPP 和 post-MPP 分别怎么找台阶
- 什么情况下会被标成异常
- 最终返回给上层什么结果
- 它在 `RulePredictor4` 里怎么影响最终诊断

如果你想看旧版基于 `LegacyPatternDetector` 的说明，可以继续参考 `docs/原有step_detection_logic.md`。

---

## 2. 入口与调用关系

### 2.1 直接调用入口

这个文件提供了两个常用入口：

- `detect_step(...)`
- `StepDetectPatternDetector.analyze(...)`

它们的区别是：

- `detect_step(...)` 是批量工具函数，输入一组曲线数组，内部逐条调用检测
- `StepDetectPatternDetector.analyze(...)` 是面向主诊断流程的接口，`RulePredictor4` 会通过它来拿到 pattern 结果

### 2.2 在主流程中的位置

主流程里，`RulePredictor4` 会先构造 pattern detector，然后调用：

```python
pattern = self.pattern_detector.analyze(...)
```

当前 `RulePredictor4` 里，`break_shadow` 的判断是：

```python
break_shadow = pattern.get("step_abnormal") or pattern.get("deform_abnormal")
```

也就是说，这个 detector 的输出会直接影响是否把样本判成 `break_shadow`。

### 2.3 归一化与原始曲线

`RulePredictor4` 在把曲线交给 pattern detector 前，会先做归一化，再按 `Voc` 截断到 `voltage <= 1.0` 的部分。

这点很重要：

- 主流程里的 `StepDetectPatternDetector.analyze(...)` 接到的是归一化后的曲线
- `detect_step(...)` 则是直接吃原始数组，通常用于批处理或调试

---

## 3. 输入与输出

### 3.1 输入

`StepDetectPatternDetector.analyze(...)` 的核心输入是：

- `curve = (voltage_array, current_array)`

可选输入：

- `features`
- `healthy_ref`
- `group_stats`

但在当前实现里，`features` 和 `healthy_ref` 基本没有参与台阶检测本身，主要用于接口兼容。

### 3.2 输出

返回一个字典，核心字段如下：

```python
{
    "steps": {...},
    "deformation": {...},
    "analysis_plot_path": str | None,
    "step_abnormal": bool,
    "deform_abnormal": bool,
    "reason_candidates": list[str],
}
```

其中：

- `steps` 存台阶数量、台阶位置、台阶深度、偏离量、跨度信息以及 pre/post 分数
- `deformation` 主要是兼容字段，当前实现里大部分内容是占位
- `step_abnormal` 表示台阶异常
- `deform_abnormal` 表示特殊形变异常
- `reason_candidates` 是上层用于解释诊断原因的候选字符串

---

## 4. 配置来源

### 4.1 `DetectConfig`

文件顶部的 `DetectConfig` 定义了所有检测阈值的默认值。

这些阈值分成三类：

- pre-MPP 检测阈值
- post-MPP 检测阈值
- 评分阈值

### 4.2 在 `RulePredictor4` 中的配置优先级

`RulePredictor4` 构造 detector 时，会合并三层配置：

1. `healthy_curve_flow.thresholds`
2. `healthy_curve_flow.rule_predictor4`
3. `healthy_curve_flow.pattern_detector.thresholds`

其中 `pattern_detector.thresholds` 的优先级最高。

### 4.3 当前默认 YAML

在 `src/config/diagnosis_config.yaml` 里，`healthy_curve_flow.pattern_detector.type` 默认是：

- `step_detect`

常用阈值默认值大致是：

- `front_discard_ratio: 0.025`
- `end_discard_ratio: 0.025`
- `pre_mpp_discard_ratio: 0.025`
- `post_mpp_discard_ratio: 0`
- `pre_window_size: 10`
- `pre_slope_diff_ratio: 0.3`
- `slope_diff_min: 0.15`
- `min_region_ratio: 0.03`
- `min_interpolate_points: 200`
- `pre_min_current_drop: 0.05`
- `pre_min_voltage_span: 0.5`
- `post_power_n_min: 1.0`
- `post_power_n_max: 5.0`
- `post_d2i_std_multiplier: 2.0`
- `post_min_step_height: 0.01`
- `post_residual_threshold: 0.035`
- `post_verify_window: 2`
- `post_left_convex_threshold: 0.0`
- `score_position_decay: 0`
- `score_left_convex_penalty: 10.0`
- `score_fit_fail_penalty: 5.0`

注意：`diagnosis_config.yaml` 里的值会覆盖代码默认值。

另外要注意：

- 当前 pre-MPP 新算法真正使用的是 `slope_diff_min`、`min_region_ratio`、`min_interpolate_points`
- `pre_window_size` 和 `pre_slope_diff_ratio` 仍保留在配置里做兼容，但已经不是 pre-MPP 新算法的主判定阈值

---

## 5. 预处理：`StringData`

### 5.1 为什么要先包装成 `StringData`

`StringData` 的作用是把一条 IV 曲线切成几个有意义的区间，方便后续分别做 pre-MPP 和 post-MPP 检测。

### 5.2 它做了什么

初始化时它会：

1. 按 `front_discard_ratio` 和 `end_discard_ratio` 裁掉电压两端的边缘点
2. 在裁剪后的范围内，用 `power = V * I` 找 MPP 点
3. 根据 MPP 位置，把曲线切成：
   - `voltage_pre / current_pre`
   - `voltage_post / current_post`

### 5.3 MPP 切分逻辑

它不是简单按 MPP 把曲线一刀切开，而是按 `detect_optimal.py` 的四个丢弃比例控制输入区间。

原因是：

- pre-MPP 和 post-MPP 的异常形态不同
- MPP 附近本身更容易受噪声影响

当前主字段是：

- `front_discard_ratio`：低压端丢弃比例，用于去掉低压插值/噪声区
- `end_discard_ratio`：高压末端丢弃比例，用于去掉 Voc 噪声区
- `pre_mpp_discard_ratio`：pre 区域从 MPP 往前丢弃的比例
- `post_mpp_discard_ratio`：post 区域相对 MPP 的起点偏移比例

切分公式是：

```python
pre_end_v = mpp_v - pre_mpp_discard_ratio * v_range
post_start_v = mpp_v + post_mpp_discard_ratio * v_range
```

其中 `post_mpp_discard_ratio` 的符号语义要特别注意：

- `post_mpp_discard_ratio > 0`：从 MPP 往后推，丢弃 MPP 后附近区域
- `post_mpp_discard_ratio = 0`：post 从 MPP 附近开始
- `post_mpp_discard_ratio < 0`：post 扩展到 MPP 前

旧字段仍做兼容 fallback，但不再推荐写入配置：

| 旧字段 | 新字段 | 兼容映射 |
| --- | --- | --- |
| `min_voltage_ratio` | `front_discard_ratio` | 同值 |
| `max_voltage_ratio` | `end_discard_ratio` | 同值 |
| `pre_mpp_voltage_ratio_after` | `pre_mpp_discard_ratio` | 同值 |
| `post_mpp_voltage_ratio_before` | `post_mpp_discard_ratio` | 取负值，即 `post_mpp_discard_ratio = -post_mpp_voltage_ratio_before` |

---

## 6. pre-MPP 检测

pre-MPP 检测的核心函数是 `_pre_mpp_detect(...)`。

### 6.1 输入不足时直接退出

如果点数小于 `10`，直接返回空结果：

```python
StepResult(step_loc=[], step_degree=[])
```

### 6.2 先插值，再平滑

如果 pre 段点数不足 `min_interpolate_points`，代码会先把曲线线性插值到该点数。

这样做的目的有两个：

- 把原本太稀疏的 pre 段拉到更稳定的采样密度
- 让后面的平滑窗口和局部 slope 统计不至于被少数点强烈扰动

插值后的平滑规则是：

- 如果工作点数 `>= 100`，Savgol 窗口固定为 `11`
- 否则使用不超过 `7` 的自适应奇数窗口，最小为 `3`

### 6.3 滑窗计算 slope 中位数

接下来会在工作曲线上滑动一个“约等于总长度 10%”的窗口，对每个窗口做一次线性拟合：

```python
np.polyfit(v_win, i_win, 1)[0]
```

得到的是局部 slope 序列 `slopes`。

然后再计算：

- `slope_median = median(slopes)`

这个中位数斜率会作为“正常走势”的基线。

### 6.4 找连续偏离区段

新实现不再用“相对斜率偏差比值”，而是直接计算：

```python
deviations = slopes - slope_median
```

如果某个窗口满足：

```python
abs(deviations[idx]) > slope_diff_min
```

就把它视为 slope 偏离点。

连续偏离点会被合并成一个候选区域。区域长度如果小于：

```python
max(3, int(len(i_smooth) * min_region_ratio))
```

就会被当成太短的局部扰动而过滤掉。

### 6.5 有效台阶过滤

候选区域通过长度过滤后，会取绝对偏离最大的点作为台阶位置。

然后再计算该区域的：

- `current_span`
- `voltage_span`
- `deviation`

其中：

- `current_span` 是区域起终点的电流差
- `voltage_span` 是区域起终点的电压差
- `deviation` 是该区域峰值位置相对于 `slope_median` 的偏离量

只有同时满足：

- `current_span >= pre_min_current_drop`
- `voltage_span >= pre_min_voltage_span`

才会把这个台阶写入结果。

### 6.6 pre-MPP 的 `step_degree`

pre-MPP 的 `step_degree` 现在等价于 `current_span`。

这部分的物理意义比较直观：

- 数值越大，说明这个台阶导致的电流下降越明显

另外，pre-MPP 每个台阶还会额外保留：

- `step_deviation`
- `current_span`
- `voltage_span`

---

## 7. post-MPP 检测

post-MPP 检测的核心函数是 `_post_mpp_detect(...)`。

当前默认方法是 `post_detection_method: "optimal"`。这部分逻辑来自 `post_step_detect/detect_optimal.py` 的后阶梯检测思路，但生产代码没有运行时引入根目录实验脚本，而是把 post-only 逻辑直接放进 `src/diagnosis/pattern_detectors/step_detect_pattern_detector.py`。

要点是：

- pre-MPP 检测逻辑没有变化
- post-MPP 默认不再走旧版“幂次拟合 + 二阶导”路线
- 旧版 post 检测仍保留，可通过 `post_detection_method: "legacy"` 回退

默认 post 配置如下：

```yaml
post_detection_method: "optimal"
post_opt_window_size: 10
post_opt_window_step: 1
post_opt_sos_window_mult: 1.5
post_opt_sos_window_step: 1
post_opt_min_positive_region_len: 5
```

### 7.1 点数不足直接退出

如果长度小于 5，直接返回空结果。

### 7.2 PAVA 单调递减预处理

optimal post 检测会先对 post 段电流做 PAVA 单调递减处理。

目的不是重新塑造曲线，而是降低局部非单调噪声对后续 slope 检测的影响。PAVA 会尽量保留大的跳变，同时把不符合 IV 曲线单调下降趋势的小波动压平。

### 7.3 滑窗拟合局部 slope

在处理后的 post 段上，算法按 `post_opt_window_size` 和 `post_opt_window_step` 滑动窗口。

每个窗口做一次线性拟合：

```python
slope, intercept = np.polyfit(v_window, i_window, 1)
```

由此得到 post 段的局部 slope 序列。当前实现还会把每个窗口的调试信息保存到 `post_step_result.window_fits`：

- `start_idx/end_idx`
- `slope/intercept`
- `baseline`
- `deviation`
- `slope_change`
- `is_step`
- `start_voltage/end_voltage`

这些字段主要用于画图调试，不进入 run_jianheng 的 CSV 输出。

### 7.4 slope-of-slope 找候选区域

接下来不是直接用 slope 本身判台阶，而是继续在 slope 序列上计算 slope-of-slope。

窗口大小由下面参数控制：

```python
sos_window = post_opt_sos_window_mult * post_opt_window_size
```

如果某段 slope-of-slope 满足：

```python
slope_of_slope >= 0
```

则说明局部 slope 出现“回升/凸起”趋势，这在 post-MPP 段通常对应电流下降形态中的异常折点或台阶候选。

连续满足条件的点会被合并成候选区域，区域长度必须至少达到：

```python
post_opt_min_positive_region_len
```

太短的区域会被当作局部扰动过滤。

### 7.5 在候选区域内定位电流下跳

每个候选区域内部会继续找最明显的电流下跳点。

代码不是只看单点差，而是在候选区域内取一段前后窗口均值：

```python
jump = mean(after_window) - mean(before_window)
```

只有 `jump < 0` 才代表电流下跳。

算法会选择最负的 `jump` 作为该区域的后阶梯位置，并计算：

- `current_drop = -jump`
- `voltage_span`
- `step_deviation = jump`

最终只保留：

```python
current_drop >= post_min_step_height
```

这一步保证 optimal post 检测只输出真实的电流下跳台阶，不把正向跳变当成后阶梯。

### 7.6 post-MPP 的 `step_degree`

这里要特别注意：当前实现里，post-MPP 每个有效台阶会同时记录位置加 4 个度量值：

- `step_loc`
- `step_degree`
- `step_deviation`
- `current_span`
- `voltage_span`

它们的真实含义是：

- `step_degree = max(current_drop, voltage_span)`
- `step_deviation = jump`
- `current_span = current_drop`
- `voltage_span = 候选区域电压跨度`

这里的 `jump` 是 `mean(after_window) - mean(before_window)`，所以后阶梯电流下跳时 `step_deviation` 为负值。

这意味着：

- pre-MPP 的 `step_degree` 是“电流跌落”
- post-MPP 的 `step_degree` 是为了延续原有 `post_score` 打分接口而构造的后阶梯强度

因此：

- `current_span` 更接近“实际电流落差”
- `voltage_span` 是该后阶梯候选区域的横向跨度
- `step_degree` 是 `post_score` 使用的汇总强度

这是当前代码的真实行为，不能把 pre/post 的 `step_degree` 理解成统一物理量。

### 7.7 post-MPP 严苛度调参建议

这里的“更严苛”主要指减少 post-MPP 弱扰动、噪声和尾部小波动导致的误报。当前默认 post 方法是：

```yaml
post_detection_method: "optimal"
```

optimal post 检测的严苛度分两层：

1. 候选区域是否形成：由滑窗 slope 和 slope-of-slope 决定
2. 候选区域是否最终输出为 post step：由电流下跳幅度和 `post_score` 决定

优先调下面这些参数：

| 优先级 | 参数 | 更严苛方向 | 作用 | 推荐试调 |
| --- | --- | --- | --- | --- |
| 1 | `post_min_step_height` | 调大 | 最终过滤电流下跳幅度，直接决定小下跳是否算 post step | `0.01 -> 0.02/0.03`，很严可到 `0.05` |
| 2 | `post_mpp_discard_ratio` | 从负值/0 调到正值 | 控制 post 输入段从 MPP 后更远处开始，减少 MPP 附近干扰 | `0 -> 0.01/0.025`；如果原来是负值，先调到 `0` |
| 3 | `post_opt_min_positive_region_len` | 调大 | 要求 slope-of-slope 连续正区域更长，过滤短噪声区 | `5 -> 8/10/12` |
| 4 | `post_opt_window_size` | 调大 | 滑窗 slope 更平滑，小局部波动不容易形成候选 | `10 -> 15/20` |
| 5 | `post_opt_sos_window_mult` | 调大 | slope-of-slope 计算窗口更大，进一步压短周期扰动 | `1.5 -> 2.0/2.5/3.0` |
| 6 | `post_opt_window_step` | 适度调大 | 降低窗口重叠密度，减少密集局部候选 | `1 -> 2/3`，不建议优先调 |
| 7 | `score_position_decay` | 调大 | 降低远离 MPP 的 step 对 `post_score` 的贡献 | `0 -> 0.5/1.0`，但会同时影响 pre/post 评分 |

推荐调参顺序：

1. 先调 `post_min_step_height`。这是最直接、最可控的严苛开关。
2. 如果误报集中在 MPP 附近，把 `post_mpp_discard_ratio` 从 `0` 调到 `0.01/0.025`。
3. 如果图里红色 `window_fits` 很碎、很短，再调大 `post_opt_min_positive_region_len`。
4. 如果蓝色/红色窗口线明显受局部噪声影响，再调大 `post_opt_window_size` 或 `post_opt_sos_window_mult`。
5. 只有确认可以牺牲窄台阶敏感性时，再调 `post_opt_window_step`。
6. `score_position_decay` 不作为首选，因为它影响的是评分，不是候选生成，并且会同时影响 pre/post。

常用组合：

| 严苛级别 | 配置建议 | 适用场景 |
| --- | --- | --- |
| 轻度严苛 | `post_min_step_height: 0.02`，`post_opt_min_positive_region_len: 8` | 当前误报不多，只想过滤很弱的小下跳 |
| 中度严苛 | `post_min_step_height: 0.03`，`post_mpp_discard_ratio: 0.01`，`post_opt_min_positive_region_len: 10`，`post_opt_window_size: 15` | 图里出现较多短红色窗口或 MPP 附近噪声候选 |
| 高度严苛 | `post_min_step_height: 0.05`，`post_mpp_discard_ratio: 0.025`，`post_opt_min_positive_region_len: 12`，`post_opt_window_size: 20`，`post_opt_sos_window_mult: 2.5` | 误报成本高，可以接受漏掉弱/窄后阶梯 |

注意：

- `post_min_step_height` 在归一化电流曲线上可理解为相对 Isc 的下跳幅度；例如 `0.02` 约等于 2% Isc。
- `post_mpp_discard_ratio` 越大，post 输入段越远离 MPP，MPP 附近的真实后阶梯也越容易被丢弃。
- 参数越严，小台阶、窄台阶、靠近 Voc 端的真实后阶梯越容易漏检。
- `post_score >= 0.06` 是当前代码常量，不是 YAML 配置；如果只想通过配置调参，优先不要动这个阈值。
- `post_residual_threshold` 和 `post_verify_window` 主要服务于 legacy post 检测，不是 optimal post 的首选调参项。

### 7.8 legacy post 检测回退

旧版 post 检测仍保留在 `_post_mpp_detect_legacy(...)`，可通过配置回退：

```yaml
post_detection_method: "legacy"
```

legacy 方法的核心流程是：

1. 拟合幂次衰减模型：

```python
i(v) = i0 - a * (v - v0 + 1e-6)^n + i_end
```

2. 用拟合曲线判断特殊形态：

- `-1` -> `left_convex`
- `-2` -> `fit_failed`

3. 用二阶导找候选点：

```python
d2i_threshold = std(d2i) * post_d2i_std_multiplier
```

4. 用残差差异和电流跌落做二次确认：

- `step_height > post_min_step_height`
- `res_change > post_residual_threshold`

legacy 主要用于回归对比和问题定位，不是当前默认路径。

---

## 8. 评分逻辑

评分函数是 `_calculate_score(...)`。

### 8.1 没有台阶时

如果 `step_loc` 为空，分数就是 `0.0`。

### 8.2 特殊状态

如果首个台阶位置是：

- `-1`，返回 `score_left_convex_penalty`
- `-2`，返回 `score_fit_fail_penalty`

### 8.3 正常状态

正常情况下，分数是所有台阶的加权和：

```python
score += degree * exp(-score_position_decay * distance / v_range)
```

其中：

- `degree` 来自 `step_degree`
- `distance` 是台阶位置到 MPP 的距离
- `score_position_decay` 越大，离 MPP 越远的台阶权重越低

### 8.4 当前默认值的含义

默认 `score_position_decay = 0` 时：

- 位置权重恒等于 1
- 分数主要就是台阶“度量值”本身的加总

这会让 pre/post 的 `step_degree` 单位差异更加明显，所以更适合把它理解成“检测强度”而不是严格物理量。

---

## 9. `analyze()` 的最终结果

`StepDetectPatternDetector.analyze(...)` 会把 pre/post 的结果合并成一个上层友好的结构。

### 9.1 `steps`

这个字段里会包含：

- `num_steps`
- `max_step_depth`
- `step_depths`
- `step_deviations`
- `current_spans`
- `voltage_spans`
- `step_locs`
- `pre_step_locs`
- `post_step_locs`
- `pre_step_deviations`
- `post_step_deviations`
- `pre_current_spans`
- `post_current_spans`
- `pre_voltage_spans`
- `post_voltage_spans`
- `pre_score`
- `post_score`
- `pre_step`
- `post_step`
- `analysis_plot_path`
- `post_detection_method`

注意：

- `step_locs` 是 pre 和 post 的位置合并
- `step_depths` 只保留有效台阶的 `step_degree`
- `step_deviations/current_spans/voltage_spans` 和 `step_locs` 一一对应
- pre 段里，`step_depths` 更接近电流跌落
- optimal post 段里，`step_depths` 使用 `max(current_drop, voltage_span)`，所以 `post_score` 更接近“后阶梯强度”的累加
- `post_detection_method` 记录当前 post 检测方法，默认是 `optimal`
- `window_fits` 是内部调试数据，只跟随 `StepResult` 进入画图函数，不写入 `steps` 字段和 CSV
- `pre_step = (pre_score >= 0.02)`
- `post_step = (post_score >= 0.06)`

### 9.2 `deformation`

当前实现里这个字段主要是兼容结构，内容如下：

- `low_voltage: {"res_energy": None}`
- `high_voltage: None`
- `special_pattern: [...]`

真正会起作用的是 `special_pattern`，也就是：

- `left_convex`
- `fit_failed`

### 9.3 异常标记

`step_abnormal` 的判断有两层：

1. `num_steps >= num_steps_limit`
2. 如果 `pre_score >= 0.02` 或 `post_score >= 0.06`，也直接视为异常

为了让上层更容易区分异常主要来自 pre 还是 post，当前 `steps` 里还会直接输出：

- `pre_step`
- `post_step`

`deform_abnormal` 则等价于：

- 是否出现了 `left_convex` 或 `fit_failed`

### 9.4 原因串

`reason_candidates` 的生成逻辑是：

- 如果 `step_abnormal`，加入 `steps=<num>`
- 再追加特殊原因

所以它更像“给上层看的摘要”，不是完整的诊断解释树。

---

## 10. 可视化与调试输出

### 10.1 `detect_step(..., save_fig=...)`

如果传了 `save_fig`，会给每条曲线保存一张图：

- 文件名格式：`string_0000.png`
- 内容包括：
  - 原始曲线
  - MPP 位置
  - pre-MPP 拟合线
  - post-MPP 拟合线；optimal 模式下画 `window_fits` 的滑窗线性拟合，legacy 模式下画旧 power fit
  - 台阶标记

### 10.2 `analyze(..., save_fig_dir)`

如果 `thresholds` 里配置了 `save_fig_dir`，`analyze()` 会把图保存到：

- `save_fig_dir/<task_id>_<inverter_id>/...png`

文件名会尽量把 `task_id`、`inverter_id` 和 `string_id` 一起编码进去，方便回溯。

### 10.3 当前代码里的调试输出

当前实现里仍可能有异常 warning。

这不影响算法流程，但说明这份代码还保留了较强的调试痕迹。阅读文档时，应该把这些输出视作辅助信息，而不是算法核心。

---

## 11. 这个 detector 在业务上的作用

在 `RulePredictor4` 里，这个 detector 主要服务于 `break_shadow` 相关判断。

简化来说：

- detector 发现明显台阶
- 或者 detector 发现特殊形变
- 上层就会倾向把样本视为 `break_shadow`

这也是为什么它的输出字段设计得比较“诊断化”，而不只是返回几个局部候选点。

---

## 12. 结合测试看它是否符合预期

仓库里的测试已经覆盖了几个关键点：

- 能检测清晰的 pre-MPP 台阶
- 默认 optimal post 检测能识别 MPP 后电流下跳
- 默认 optimal post 检测不会把平滑线性 post 段误判成后阶梯
- `detect_step(...)` 会保存图
- `analyze(...)` 保存的 `analysis_plot_path` 能回到结果里
- `analyze(...)` 和 `detect_string(...)` 的核心结果一致
- 没有特殊情况时，`reason_candidates` 会保持为空

这些测试对应的文件是：

- `tests/test_rule_predictor4_logic.py`
- `tests/test_predictor_config_selection.py`

---

## 13. 一句话总结

这个 detector 的思路可以概括成三步：

1. 先围绕 MPP 把曲线切成 pre 和 post 两段
2. pre 段用滑窗斜率找台阶，post 段默认用 PAVA + 滑窗 slope + slope-of-slope 找后阶梯电流下跳
3. 再把台阶数量、评分和特殊形变一起交给上层做 `break_shadow` 判断
