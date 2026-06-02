# load_pred_holiday_on_schedule.py - 节假日负荷预测调度模块

## 概述

本模块负责处理**节假日模式（Holiday On）**的负荷预测任务。当站点处于节假日模式（mode_type=2, operation_type=1）时，会启用专门的预测策略，使用历史最低负荷天的数据进行建模，更准确地预测节假日期间的低负荷特征。

## 核心特性

### 1. 节假日专用预测策略
- **数据选择**：从历史数据中选择日均负荷最低的7天（7×288=2016个点）
- **模型组合**：使用DLinear + TimeXer深度学习模型融合
- **后处理优化**：线性拟合到历史平均负荷曲线形态

### 2. 多模型融合
- **DLinear**：负荷预测深度学习模型（`Dlinear-load-dayahead`）
- **TimeXer**：时间序列预测模型（`Timexer-load-dayahead`）
- **DeepHoliday**：融合后的节假日专用模型（`DeepHoliday-load-dayahead`）

### 3. 时区支持
- 支持夏令时（DST）调整
- 支持冬令时（WST）调整
- 自动检测并处理时间跳变

---

## 主要类和函数

### 类：LoadModelPredictionHolidayOn
线程类，用于调度节假日负荷预测任务。

#### 参数
- `time_area` (str): 时区名称（如 "America/New_York"）
- `mode` (int): 运行模式，1=定时调度，2=立即执行一次
- `exp_dlinear`: DLinear模型实验对象
- `exp_timexer`: TimeXer模型实验对象
- `lock`: 线程锁（用于TimeXer串行执行）

#### 方法
- `run()`: 线程主入口
- `_run_once()`: 执行一次预测
- `_run_schedule()`: 定时调度（每天18:00执行）

---

## 核心函数详解

### 1. prepare_data_for_multi_model_holiday()
**节假日数据准备函数 - 核心逻辑**

从历史数据中提取日均负荷最低的7天数据，用于节假日预测。

#### 参数
- `df` (DataFrame): 包含 `['station_id', 'statistics_time', 'load']` 列
- `station_mode_list` (list): 站点模式列表 `[(station_id, feature_a), ...]`
- `pred_now_date` (datetime): 预测基准日期
- `target_points` (int): 目标数据点数，默认2016（7天×288点/天）

#### 返回值
- `result_array` (numpy.ndarray): shape `[n_stations, 2016, 2]`，最后一维为 `[statistics_time, load]`
- `new_station_mode_list` (list): 处理后的站点模式列表

#### 算法流程
```python
for each station:
    1. 按日期分组，计算每天的平均负荷
    2. 筛选完整天（>=200个点/天）
    3. 选择日均负荷最小的7天
    4. 提取这7天的所有数据点（最多2016个）
    5. 如果数据不足，使用线性插值填充
    6. 如果数据过多，截取最新的2016个点
```

#### 关键日志
```
站点 {station_id}: 选择了日均负荷最低的 7 天
站点 {station_id}: 选择的日期: [2024-01-05, 2024-01-12, ...]
站点 {station_id}: 日均负荷: [12.34, 13.45, ...]
```

---

### 2. batch_forecast_dlinear_model_holiday()
**DLinear节假日预测函数**

使用DLinear模型对节假日数据进行预测，返回未写入Kafka的结果。

#### 参数
- `batch_np` (numpy.ndarray): 历史数据 `[n_stations, 2016, 2]`
- `pred_date` (datetime): 预测日期（第一天）
- `new_station_mode_list` (list): 站点模式列表
- `gap_time`: 夏令时时间gap
- `winter_extra_time`: 冬令时时间gap
- `exp_dlinear`: DLinear实验对象

#### 返回值（字典）
```python
{
    'day1_pred': np.ndarray,  # [n_stations, 288, 1] 第一天预测
    'day1_date': str,         # "2024-01-15"
    'statistics_time_day1': list,  # 时间戳列表
    'day2_pred': np.ndarray,  # [n_stations, 288, 1] 第二天预测
    'day2_date': str,         # "2024-01-16"
    'statistics_time_day2': list,
    'model': 'Dlinear-load-dayahead',
    'station_mode_list': list
}
```

---

### 3. batch_forecast_timexer_model_holiday()
**TimeXer节假日预测函数**

与`batch_forecast_dlinear_model_holiday()`结构相同，使用TimeXer模型。

---

### 4. deep_holiday_dayahead_postprocess()
**节假日深度模型后处理 - 融合核心**

对DLinear和TimeXer预测结果进行线性拟合融合，使预测曲线形态接近历史平均负荷。

#### 参数
- `timexer_result` (dict): TimeXer预测结果
- `dlinear_result` (dict): DLinear预测结果
- `batch_np` (numpy.ndarray): 历史数据 `[n_stations, 2016, 2]`
- `new_station_mode_list` (list): 站点模式列表

#### 算法流程
```python
1. 提取历史7天负荷数据 [n_stations, 2016]
2. Reshape成 [n_stations, 7, 288]
3. 计算每个站点的平均日负荷曲线 [n_stations, 288]

for each station:
    # 第一天
    dlinear_d1_adjusted = linear_fit_to_pattern(dlinear_day1, hist_pattern)
    timexer_d1_adjusted = linear_fit_to_pattern(timexer_day1, hist_pattern)
    adjusted_day1 = 0.5 * dlinear_d1_adjusted + 0.5 * timexer_d1_adjusted

    # 第二天
    dlinear_d2_adjusted = linear_fit_to_pattern(dlinear_day2, hist_pattern)
    timexer_d2_adjusted = linear_fit_to_pattern(timexer_day2, hist_pattern)
    adjusted_day2 = 0.5 * dlinear_d2_adjusted + 0.5 * timexer_d2_adjusted

4. 确保预测值非负
5. 返回融合结果
```

#### 返回值
```python
{
    'model': 'DeepHoliday-load-dayahead',
    # 其他字段同DLinear返回值
}
```

---

### 5. linear_fit_to_pattern()
**线性拟合函数 - 后处理核心算法**

使用最小二乘法将预测曲线线性拟合到目标模式（历史平均曲线）。

#### 目标
找到系数 `a` 和 `b`，使得 `a × pred_curve + b` 尽可能接近 `target_pattern`。

#### 算法
```python
# 最小二乘法求解 y = a*x + b
n = len(pred_curve)
sum_x = Σ pred_curve
sum_y = Σ target_pattern
sum_xy = Σ (pred_curve × target_pattern)
sum_xx = Σ (pred_curve × pred_curve)

denominator = n × sum_xx - sum_x²
a = (n × sum_xy - sum_x × sum_y) / denominator
b = (sum_y - a × sum_x) / n

adjusted_curve = a × pred_curve + b
```

#### 约束条件
- 如果 `a < 0` 或 `a > 3`，使用保守策略：归一化到 `[0,1]` 然后缩放到目标范围
- 如果预测曲线几乎为常数（`denominator < 1e-10`），直接使用目标平均值

---

### 6. predict_area_load_multi_model_parallel_worker_holiday()
**节假日预测Worker - 主执行函数**

按批次处理站点数据，执行完整的预测流程。

#### 参数
- `area` (str): 时区名称
- `gap_time`: 夏令时gap
- `winter_extra_time`: 冬令时gap
- `pred_now_date` (datetime): 预测基准日期
- `exp_dlinear`: DLinear实验对象
- `exp_timexer`: TimeXer实验对象
- `lock2`: TimeXer线程锁
- `global_holiday_list` (list): 节假日站点列表

#### 执行流程
```python
1. 获取站点数据生成器（batch_size=20）
   ↓
2. 对每个批次：
   2.1 准备数据（选择最低负荷7天）
   2.2 DLinear预测（并行）
   2.3 TimeXer预测（串行，加锁）
   2.4 深度模型后处理融合
   2.5 写入Kafka（4次写入）：
       - 第一天 DeepHoliday 结果（带DST/WST）
       - 第一天 select_best_7 结果（带DST/WST）
       - 第二天 DeepHoliday 结果（无DST/WST）
       - 第二天 select_best_7 结果（无DST/WST）
   2.6 清理内存
   ↓
3. 打印时间统计
```

#### Kafka写入格式
```
{pred_date};{station_id};{pred_type};{model};{version};{record_time};{statistics_time};{predicted_values}
```

示例：
```
2024-01-15;12024061505406;2;DeepHoliday-load-dayahead;0.1;1705305600;[1705305600,1705305900,...];[0.12345,0.23456,...]
```

---

### 7. pred_area_load_multi_model_parallel_holiday()
**节假日预测主调度函数**

入口函数，协调整个节假日预测流程。

#### 参数
- `time_area` (str): 时区名称
- `exp_dlinear`: DLinear实验对象
- `exp_timexer`: TimeXer实验对象
- `lock`: 线程锁

#### 执行流程
```python
1. 获取时区和目标时间
2. 检测夏令时/冬令时
3. 获取AI站点列表
4. 获取节假日站点列表（调用 get_ai_station_holiday_on()）
5. 提交Worker任务到线程池
6. 等待任务完成
7. 打印时间统计
```

#### 关键日志
```
地区 America/New_York 该地区AI站点总数: 150
地区 America/New_York 该地区节假日--开启中--站点数量为 23
```

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 获取节假日站点列表                                          │
│    db.get_ai_station_holiday_on(dt, ai_station)              │
│    → 返回 mode_type=2, operation_type=1 的站点                │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 批量获取历史数据（过去60天）                                │
│    get_area_station_data_his_generator_holiday()             │
│    batch_size=20                                             │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 节假日数据准备                                             │
│    prepare_data_for_multi_model_holiday()                    │
│    → 选择日均负荷最低的7天（2016个点）                         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
┌──────────────────┐       ┌──────────────────┐
│ 4a. DLinear预测  │       │ 4b. TimeXer预测  │
│ (并行)           │       │ (串行，加锁)      │
└────────┬─────────┘       └────────┬─────────┘
         └─────────────┬─────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 深度模型后处理融合                                         │
│    deep_holiday_dayahead_postprocess()                       │
│    → 线性拟合到历史平均曲线                                    │
│    → 0.5 × DLinear + 0.5 × TimeXer                           │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 写入Kafka（4次）                                           │
│    - Day1 DeepHoliday（带DST/WST）                           │
│    - Day1 select_best_7（带DST/WST）                         │
│    - Day2 DeepHoliday（无DST/WST）                           │
│    - Day2 select_best_7（无DST/WST）                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 与其他模块的关系

### 依赖模块
- `database.py`: 数据库操作（获取站点列表、节假日状态）
- `multi_model_dispatch.py`: DLinear/TimeXer模型调用
- `get_batch_station_data_handler.py`: 批量获取站点数据
- `util/dst_util.py`: 夏令时处理
- `util/wst_util.py`: 冬令时处理
- `util/time_recorder_util.py`: 时间统计

### 协同模块
- `load_pred_holiday_off_schedule.py`: 节假日关闭模式预测
- `load_pred_holiday_off_call.py`: 节假日关闭模式选优
- `load_holiday_intra_on_call.py`: 节假日日内预测

### 数据库交互

#### 查询节假日站点
```sql
-- database.py:get_ai_station_holiday_on()
WITH latest_records AS (
    SELECT station_id, dt, mode_type, operation_type, record_time,
           ROW_NUMBER() OVER (PARTITION BY station_id, dt ORDER BY record_time DESC) as rn
    FROM sigen_ai.station_mode_operation_log
    WHERE dt = '{dt}' AND station_id IN ({station_ids})
)
SELECT DISTINCT station_id
FROM latest_records
WHERE rn = 1 AND mode_type = 2 AND operation_type = 1
```

---

## 时间处理逻辑

### 夏令时（DST）处理
```python
# 检测是否有DST
dst_timezone_set = get_dst_timezones(target_time)
if time_area in dst_timezone_set:
    gap_time = get_loss_hour(time_area, target_time)
    # gap_time 示例: (datetime(2024,3,10,2,0), datetime(2024,3,10,3,0))

# 应用DST校正（仅第一天）
statistics_time, each_pred = check_dst_time(statistics_time, each_pred, gap_time)
```

### 冬令时（WST）处理
```python
# 检测是否有WST
wst_timezone_set = get_wst_timezones(target_time)
if time_area in wst_timezone_set:
    winter_extra_time = get_repeat_hour(time_area, target_time)

# 应用WST校正（仅第一天）
statistics_time, each_pred = check_wst_time(statistics_time, each_pred, winter_extra_time)
```

**注意**：第二天预测不进行DST/WST校正（`gap_time=None, winter_extra_time=None`）

---

## 性能优化

### 1. 批量处理
- 批大小：20个站点/批次
- 减少数据库查询次数
- 优化内存使用

### 2. 内存管理
```python
# 每批次处理后清理
del batch_np
del batch_df
gc.collect()
```

### 3. 线程锁策略
- **DLinear**：无锁，并行执行
- **TimeXer**：加锁（`lock2.acquire()`），串行执行（避免GPU资源竞争）

### 4. 时间统计
使用 `time_recorder` 记录各阶段耗时：
- 数据获取
- 数据预处理
- DLinear推理
- TimeXer推理
- 后处理
- Kafka写入

---

## 错误处理

### 1. 数据不足处理
```python
if len(valid_days) < 7:
    logger.warning(f"站点 {station_id}: 有效天数不足7天，跳过该站点")
    continue
```

### 2. 模型推理失败
```python
try:
    dlinear_result = batch_forecast_dlinear_model_holiday(...)
except Exception as e:
    logger.error(f"dlinear holiday 推理时出错: {e}")
    exception_handler.log_exception(e)
```

### 3. 后处理降级策略
```python
# 如果后处理失败，优先返回DLinear结果
if dlinear_result is not None:
    logger.info("后处理失败，使用DLinear结果作为fallback")
    return dlinear_result
elif timexer_result is not None:
    logger.info("使用TimeXer结果作为fallback")
    return timexer_result
```

---

## 调度配置

### 定时任务
```python
# 每天18:00（当地时间）执行
scheduler.add_job(
    pred_area_load_multi_model_parallel_holiday,
    'cron',
    hour=18, minute=0
)
```

### 运行模式
```python
# 模式1：定时调度
thread = LoadModelPredictionHolidayOn(time_area='America/New_York', mode=1)

# 模式2：立即执行一次
thread = LoadModelPredictionHolidayOn(time_area='America/New_York', mode=2)
```

---

## 配置参数

### 全局配置（nacos_config）
```yaml
load_history:
  model_name_list:
    - "Dlinear-load-dayahead"
    - "Timexer-load-dayahead"
    - "GridXGB5fold-load-dayahead"
  Backtesting_days: 60
```

### 模型权重
```python
# deep_holiday_dayahead_postprocess() 中的融合权重
adjusted_day1 = 0.5 * dlinear_d1_adjusted + 0.5 * timexer_d1_adjusted
adjusted_day2 = 0.5 * dlinear_d2_adjusted + 0.5 * timexer_d2_adjusted
```

可根据模型性能调整权重，例如：
```python
adjusted_day1 = 0.6 * dlinear_d1_adjusted + 0.4 * timexer_d1_adjusted
```

---

## 日志示例

### 正常执行日志
```
INFO: 开始预测位于地区America/New_York的场站模型
INFO: 地区 America/New_York 该地区AI站点总数: 150
INFO: 地区 America/New_York 该地区节假日--开启中--站点数量为 23
INFO: 数据获取完成，耗时: 2.35秒
INFO: 站点 12024061505406: 选择了日均负荷最低的 7 天
INFO: 站点 12024061505406: 选择的日期: [2024-01-05, 2024-01-12, 2024-01-19, ...]
INFO: 站点 12024061505406: 日均负荷: ['12.34', '13.45', '14.23', ...]
INFO: 节假日数据处理完成: 20个站点, 2016个时间点
INFO: ThreadPoolExecutor-0_1 正在尝试获取 TimeXer 锁
INFO: ThreadPoolExecutor-0_1 成功获取到 TimeXer 锁
INFO: 节假日深度模型后处理完成，处理了20个站点（第一天和第二天）
INFO: 批次1第一天后处理结果写kafka成功
INFO: 批次1第二天后处理结果写kafka成功
INFO: ThreadPoolExecutor-0_1 已释放 TimeXer 锁
INFO: === 模型总时间统计 ===
INFO: DLinear: 总耗时 15.23秒，总站点数 23
INFO: TimeXer: 总耗时 18.45秒，总站点数 23
INFO: 后处理: 总耗时 1.23秒，总站点数 23
```

### 异常日志
```
WARNING: 站点 82025040300013: 有效天数不足7天（当前4天），跳过该站点
WARNING: 站点 82025040300013: 数据点不足，需要插值填充 432 个点
ERROR: dlinear holiday 推理时处理批次数据时出错: CUDA out of memory
ERROR: 批次1深度模型后处理失败: Length mismatch
INFO: 后处理失败，使用DLinear结果作为fallback
```

---

## 与常规预测的区别

| 特性 | 节假日预测（holiday_on） | 常规预测 |
|------|-------------------------|----------|
| **数据选择** | 历史最低负荷7天 | 最近连续7天 |
| **预测模型** | DLinear + TimeXer融合 | DLinear / TimeXer / XGBoost |
| **后处理** | 线性拟合到历史平均 | 无特殊后处理 |
| **目标场景** | 节假日低负荷 | 正常工作日 |
| **触发条件** | mode_type=2, operation_type=1 | 默认 |
| **Kafka模型名** | DeepHoliday-load-dayahead | Dlinear-load-dayahead / Timexer-load-dayahead |

---

## 故障排查指南

### 1. 站点预测失败
**症状**：日志显示"跳过该站点"

**检查步骤**：
1. 确认站点是否在节假日列表中（`get_ai_station_holiday_on()`）
2. 检查历史数据是否充足（至少7天，每天>=200个点）
3. 查看数据库 `station_mode_operation_log` 表中的记录

### 2. 内存不足
**症状**：`CUDA out of memory`

**解决方案**：
1. 减小批大小（默认20 → 10）
2. 检查GPU显存占用
3. 确保每批次后执行 `gc.collect()`

### 3. Kafka写入失败
**症状**：日志显示"结果写kafka失败"

**检查步骤**：
1. 确认Kafka连接正常
2. 检查数据格式是否正确（时间戳列表、预测值列表长度一致）
3. 查看是否有DST/WST导致的长度不匹配

### 4. 时间不匹配
**症状**：`Length mismatch: Expected axis has 648 elements, new values have 864 elements`

**原因**：数据库中预测历史记录的 `statistics_time` 和 `predicted_value` 长度不一致

**解决方案**：
- 已在 `database.py:742-755` 添加长度校验和过滤逻辑（见相关修复）

---

## 最佳实践

### 1. 数据质量保障
```python
# 确保每天数据完整性（至少200个点）
daily_count = group_sorted.groupby('date').size()
valid_days = daily_count[daily_count >= 200].index
```

### 2. 模型融合权重调优
根据历史回测结果调整DLinear和TimeXer的融合权重：
```python
# 如果DLinear在节假日场景下表现更好
adjusted_day1 = 0.6 * dlinear_d1_adjusted + 0.4 * timexer_d1_adjusted
```

### 3. 批大小优化
根据GPU显存和站点数量动态调整：
```python
# 大站点（>100个）
batch_size = 10

# 小站点（<50个）
batch_size = 30
```

### 4. 日志监控
关键日志字段：
- 节假日站点数量
- 每批次处理时间
- 数据点数（应为2016）
- 模型推理时间

---

## 未来改进方向

### 1. 自适应权重
根据站点特征动态调整DLinear和TimeXer的融合权重：
```python
# 伪代码
if station_load_variance > threshold:
    weight_dlinear = 0.6
else:
    weight_dlinear = 0.4
```

### 2. 更精细的数据选择
除了日均负荷最低，还可以考虑：
- 负荷曲线形态相似度
- 天气条件相似度
- 历史节假日数据优先

### 3. 在线学习
定期使用最新节假日数据微调模型参数。

### 4. A/B测试框架
对比不同融合策略的预测效果。

---

## 相关文件

- `load_pred_holiday_off_schedule.py`: 节假日关闭模式预测
- `load_pred_holiday_off_call.py`: 节假日关闭模式选优
- `get_batch_station_data_handler.py`: 批量数据获取
- `multi_model_dispatch.py`: 深度学习模型调度
- `database.py`: 数据库操作
- `util/select_best_7_util.py`: 模型选优工具

---

## 版本历史

- **v1.0** (2024-12): 初始版本，支持DLinear + TimeXer融合
- **v1.1** (2024-12-23): 修复数据长度不匹配bug，添加数据质量检查

---

## 联系方式

如有问题或建议，请联系开发团队。