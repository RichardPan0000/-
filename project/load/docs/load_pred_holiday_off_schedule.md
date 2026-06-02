# 负荷预测节假日关闭模式调度模块文档

## 文档信息
- **模块名称**: `load_pred_holiday_off_schedule.py`
- **文件路径**: `src/load_pred_holiday_off_schedule.py`
- **版本**: 1.0
- **最后更新**: 2025-12-24

---

## 1. 概述

本模块专门用于处理节假日关闭模式下的电站负荷预测任务。与常规负荷预测不同,该模块需要识别并过滤掉电站在节假日关闭期间的数据,仅使用正常工作模式下的历史数据进行预测。

### 1.1 核心功能
- 识别和处理节假日关闭模式的电站数据
- 支持多种预测模型(DLinear、TimeXer、XGBoost)的并行推理
- 自动处理夏令时/冬令时时间转换
- 批量处理电站数据以提高效率
- 支持多天预测(未来2天)

### 1.2 适用场景
- 电站在节假日期间完全关闭,无负荷数据
- 需要过滤掉节假日期间的异常数据点
- 需要基于正常工作日数据训练和预测

---

## 2. 主要类

### 2.1 LoadModelPredictionHolidayOff

负荷预测线程类,支持单次运行和定时调度两种模式。

#### 类定义
```python
class LoadModelPredictionHolidayOff(threading.Thread):
    def __init__(self, time_area, mode=1, exp_dlinear=None, exp_timexer=None, lock=None, global_white_list=[])
```

#### 参数说明
| 参数 | 类型 | 说明 |
|------|------|------|
| `time_area` | str | 时区名称,如 "Europe/Berlin" |
| `mode` | int | 运行模式: 1=定时调度, 2=单次运行 |
| `exp_dlinear` | object | DLinear模型实验对象 |
| `exp_timexer` | object | TimeXer模型实验对象 |
| `lock` | threading.Lock | 线程锁对象 |
| `global_white_list` | list | 白名单站点列表(仅这些站点运行XGBoost) |

#### 方法
- `run()`: 线程主函数,根据mode决定运行方式
- `_run_once()`: 执行一次预测任务
- `_run_schedule()`: 启动定时调度(每天20:00执行)

#### 示例
```python
# 单次运行模式
thread = LoadModelPredictionHolidayOff(
    time_area="Europe/Berlin",
    mode=2,
    exp_dlinear=dlinear_exp,
    exp_timexer=timexer_exp,
    lock=threading.Lock()
)
thread.start()

# 定时调度模式
thread = LoadModelPredictionHolidayOff(
    time_area="Asia/Shanghai",
    mode=1
)
thread.start()
```

---

## 3. 核心函数

### 3.1 节假日数据处理

#### get_holiday_mode_mask()

获取站点在指定时间范围内的节假日模式时间段mask。

**函数签名**:
```python
def get_holiday_mode_mask(station_id, start_timestamp, end_timestamp) -> set
```

**参数**:
- `station_id` (int): 站点ID
- `start_timestamp` (int): 开始时间戳(Unix timestamp)
- `end_timestamp` (int): 结束时间戳(Unix timestamp)

**返回值**:
- `set`: 需要mask掉的时间戳集合(5分钟间隔)

**核心逻辑**:
1. 查询 `sigen_ai.station_mode_operation_log` 表获取节假日模式操作记录
2. 扩大查询范围:向前多查60天,确保捕获未关闭的节假日期间
3. 通过 `operation_type=1`(开启) 和 `operation_type=0`(关闭) 配对构建节假日时间段
4. 处理异常情况:
   - 连续多次开启:保留最早的开启时间
   - 连续多次关闭:更新为最晚的关闭时间
   - 未关闭的节假日:延续到 `end_timestamp`
5. 将节假日时间段扩展到完整日期(从00:00到23:55)
6. 生成5分钟间隔的时间戳集合

**示例**:
```python
# 获取站点12345在2024年1月的节假日mask
start_ts = int(datetime(2024, 1, 1).timestamp())
end_ts = int(datetime(2024, 1, 31, 23, 59, 59).timestamp())
mask_set = get_holiday_mode_mask(12345, start_ts, end_ts)
print(f"需要过滤的时间点数量: {len(mask_set)}")
```

---

#### prepare_data_for_multi_model_holiday_off()

准备多模型输入数据,过滤掉节假日模式时间段的数据。

**函数签名**:
```python
def prepare_data_for_multi_model_holiday_off(
    df,
    station_mode_list,
    pred_now_date: datetime = None,
    target_points: int = 2016
) -> Tuple[np.ndarray, list]
```

**参数**:
- `df` (pd.DataFrame): 包含列 `['station_id', 'statistics_time', 'load']` 的DataFrame
- `station_mode_list` (list): 站点模式列表,格式为 `[(station_id, feature_a), ...]`
- `pred_now_date` (datetime, 可选): 预测基准日期
- `target_points` (int): 目标时间点数量,默认2016(7天 × 288点/天)

**返回值**:
- `np.ndarray`: 形状为 `[n_stations, target_points, 2]` 的数组,最后一维是 `[statistics_time, load]`
- `list`: 处理后的站点列表 `[(station_id, feature_a), ...]`

**核心流程**:
1. 按 `station_id` 分组处理数据
2. 为每个站点获取节假日模式mask
3. 过滤掉节假日时间段的数据点
4. 检查过滤后数据是否充足(至少7天 = 2016个点)
5. 生成完整时间轴(移除节假日时间点)
6. 对缺失值进行前向/后向填充
7. 保留最新的 `target_points` 个时间点
8. 转换为numpy数组并堆叠

**数据质量检查**:
```python
# 过滤后数据不足7天则跳过
if len(group_sorted) < 288 * 7:
    logger.warning(f"站点{station_id}过滤后数据不足,跳过")
    continue

# 时间点数量必须等于target_points
if len(merged_df) != target_points:
    logger.warning(f"站点{station_id}数据点数不符,跳过")
    continue
```

---

### 3.2 模型预测函数

#### batch_forecast_dlinear_model()

DLinear模型批量预测函数。

**函数签名**:
```python
def batch_forecast_dlinear_model(
    batch_np: np.ndarray,
    pred_date: datetime,
    new_station_mode_list: list,
    gap_time,
    winter_extra_time,
    exp_dlinear=None
)
```

**处理流程**:
1. 调用 `process_batch_for_dlinear()` 进行预测
2. 单位缩放:结果除以100(kW → 百kW)
3. 非负裁剪:将负值设为0
4. 分别处理第一天和第二天的预测结果
5. 处理夏令时/冬令时时间调整
6. 发送结果到Kafka

**示例**:
```python
# DLinear预测
batch_forecast_dlinear_model(
    batch_np=data_array,
    pred_date=datetime(2024, 1, 15),
    new_station_mode_list=station_list,
    gap_time=None,
    winter_extra_time=None,
    exp_dlinear=dlinear_experiment
)
```

---

#### batch_forecast_timexer_model()

TimeXer模型批量预测函数。

**函数签名**:
```python
def batch_forecast_timexer_model(
    batch_np: np.ndarray,
    pred_date: datetime,
    new_station_mode_list: list,
    gap_time,
    winter_extra_time=None,
    exp_timexer=None
)
```

**特点**:
- 使用线程锁控制并发访问(TimeXer模型资源密集)
- 处理流程与DLinear类似
- 支持夏令时/冬令时调整

---

#### batch_forecast_xgb_model()

XGBoost模型批量预测函数。

**函数签名**:
```python
def batch_forecast_xgb_model(
    lock,
    batch_np: np.ndarray,
    station_id_mode_list: list,
    gap_time=None,
    winter_extra_time=None,
    pred_now_date=None,
    pred_end_date=None
)
```

**特殊处理**:
1. 将numpy数组转换为pandas DataFrame
2. 从S3下载每个站点的模型文件
3. 支持两种模型格式:
   - 参数列表格式:需要重新构建模型对象
   - 完整模型对象格式:直接加载
4. 生成未来两天的预测数据
5. 自动清理本地模型文件
6. 结果非负裁剪

**模型加载示例**:
```python
# 从S3下载模型
s3_conn.download(
    f'pv_ai_model/{model}/{station_id}_{model}_model.pkl',
    f'{station_id}_{model}_model.pkl'
)

# 加载模型
with open(f'{station_id}_{model}_model.pkl', "rb") as f:
    load_model = pickle.load(f)

# 清理本地文件
os.remove(f'{station_id}_{model}_model.pkl')
```

---

### 3.3 主调度函数

#### pred_area_load_multi_model_parallel()

多模型并行预测主调度函数。

**函数签名**:
```python
def pred_area_load_multi_model_parallel(
    time_area,
    exp_dlinear=None,
    exp_timexer=None,
    lock=None,
    global_white_list=[]
)
```

**完整流程**:

1. **时间和时区处理**:
   ```python
   new_time_zone = pytz.timezone(time_area)
   target_time = datetime.now(new_time_zone) + timedelta(days=1)
   pred_now_date = (target_time - timedelta(days=1)).replace(tzinfo=None)
   ```

2. **夏令时/冬令时检测**:
   ```python
   dst_timezone_set = get_dst_timezones(target_time)
   if time_area in dst_timezone_set:
       gap_time = get_loss_hour(time_area, target_time.strftime('%Y-%m-%d'))

   wst_timezone_set = get_wst_timezones(target_time)
   if time_area in wst_timezone_set:
       winter_extra_time = get_repeat_hour(time_area, target_time.strftime('%Y-%m-%d'))
   ```

3. **获取站点列表**:
   ```python
   # 获取该地区所有AI站点
   ai_station = db_conn_global.get_ai_station(time_area)

   # 获取节假日关闭站点列表并取交集
   global_holiday_off_list_all = db_conn_global.get_ai_station_holiday_off(pred_date)
   global_holiday_off_list = list(set(global_holiday_off_list_all) & set(ai_station))
   ```

4. **多模型预测**:
   - 启动worker线程处理批量预测
   - 并行运行 DLinear、TimeXer、XGBoost 三个模型
   - 使用线程池管理并发

5. **模型选优**:
   ```python
   # 等待预测完成后进行模型选优
   time.sleep(62)  # 等待所有模型推理完成

   for station_id in global_holiday_off_list:
       select_best_model_hour_sum_v1_dst_wst_holiday_off(
           station_id,
           history_days=60,
           target=2,
           pred_now_date=pred_now_date,
           global_white_list=global_white_list,
           gap_time=gap_time,
           winter_extra_time=winter_extra_time
       )
   ```

---

#### predict_area_load_multi_model_parallel_worker()

实际执行多模型预测的工作函数。

**函数签名**:
```python
def predict_area_load_multi_model_parallel_worker(
    area: str,
    lock,
    gap_time=None,
    winter_extra_time=None,
    pred_now_date=None,
    exp_dlinear=None,
    exp_timexer=None,
    lock2=None,
    global_white_list=[],
    global_holiday_off_list=[]
)
```

**批量处理流程**:
```python
# 获取数据生成器(每批20个站点)
station_data_generator = get_area_station_data_his_generator_holiday(
    area,
    batch_size=20,
    end_date=pred_now_date,
    past_days_num=60,
    global_holiday_list=global_holiday_off_list
)

for batch_df, station_mode_list in station_data_generator:
    # 1. 准备数据(过滤节假日)
    batch_np, new_station_mode_list = prepare_data_for_multi_model_holiday_off(
        batch_df, station_mode_list, pred_now_date, target_points=2016
    )

    # 2. DLinear预测
    batch_forecast_dlinear_model(batch_np.copy(), ...)

    # 3. TimeXer预测(需要锁)
    lock2.acquire()
    batch_forecast_timexer_model(batch_np.copy(), ...)
    lock2.release()

    # 4. XGBoost预测(仅白名单站点)
    filtered_stations = [s for s in new_station_mode_list if s[0] in global_white_list]
    batch_forecast_xgb_model(lock, filtered_batch, filtered_stations, ...)

    # 5. 清理内存
    del batch_np, batch_df
    gc.collect()
```

---

### 3.4 辅助函数

#### convert_numpy_to_dataframe()

将numpy数组转换为pandas DataFrame列表。

**函数签名**:
```python
def convert_numpy_to_dataframe(
    batch_np: np.ndarray,
    station_id_list: list
) -> List[pd.DataFrame]
```

**参数**:
- `batch_np` (np.ndarray): 形状为 `[n_stations, n_timepoints, 2]` 的数组
- `station_id_list` (list): 站点ID列表,格式为 `[(station_id, feature_a), ...]`

**返回值**:
- `List[pd.DataFrame]`: 每个站点的DataFrame列表,包含列 `['station_id', 'statistics_time', 'load']`

---

#### send_station_dlinear_result_to_kafka()

将预测结果发送到Kafka。

**函数签名**:
```python
def send_station_dlinear_result_to_kafka(
    result: np.ndarray,
    station_id_mode_list: list,
    pred_date: str,
    statistics_time: list,
    model: str,
    record_time: int,
    gap_time,
    winter_extra_time
)
```

**Kafka消息格式**:
```
{pred_date};{station_id};{pred_type};{model};{version};{record_time};{statistics_time};{predictions}
```

**示例**:
```
2024-01-15;12345;2;Dlinear-load-dayahead;0.1;1705320000;[1705276800,...];[12.34,...]
```

**数据校验**:
- 检查预测长度和时间戳长度是否为288
- 处理夏令时/冬令时时间调整
- 仅在校验通过后发送到Kafka

---

#### make_second_day_data_with_first_day()

生成第二天预测所需的输入数据。

**函数签名**:
```python
def make_second_day_data_with_first_day(
    station_his: pd.DataFrame,
    now_date: datetime
) -> pd.DataFrame
```

**核心逻辑**:
1. 提取最近8天的历史数据
2. 将第7天前的数据复制一份,时间索引向后移动7天
3. 合并原始数据和移位数据
4. 按时间索引排序

**用途**:
- XGBoost模型需要使用历史周期性数据
- 通过复制7天前的数据来模拟未来第二天的特征

---

#### check_pred_his_data_ok()

检查历史数据是否充足,不足则从数据库补充。

**函数签名**:
```python
def check_pred_his_data_ok(
    station_his: pd.DataFrame,
    station_id_mode: tuple,
    load_data_end_date: str,
    pred_date: datetime
) -> pd.DataFrame
```

**检查标准**:
- 需要至少 288 × 4 = 1152 个数据点(4天)
- 不足则从数据库获取最近6个月的数据
- 使用 `FillDataHandler` 进行数据填充

---

## 4. 数据流程图

```
┌─────────────────────────────────────────────────────────┐
│  1. 获取节假日关闭站点列表                                 │
│     db_conn_global.get_ai_station_holiday_off()         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  2. 批量获取站点历史数据(60天)                            │
│     get_area_station_data_his_generator_holiday()       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  3. 过滤节假日数据                                        │
│     prepare_data_for_multi_model_holiday_off()          │
│     - 获取节假日mask                                     │
│     - 过滤节假日时间点                                   │
│     - 数据对齐和填充                                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
         ┌────────┴────────┐
         │                 │
         ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  DLinear预测  │  │ TimeXer预测   │  │  XGBoost预测  │
│  (全部站点)   │  │  (全部站点)   │  │  (白名单站点) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
        ┌────────────────────────────────┐
        │  4. 发送预测结果到Kafka          │
        │     - 夏令时/冬令时调整          │
        │     - 数据格式化                │
        │     - 写入Kafka                 │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │  5. 模型选优                    │
        │     select_best_model_...()     │
        │     - 基于历史回测选择最优模型  │
        └────────────────────────────────┘
```

---

## 5. 配置和依赖

### 5.1 全局配置
```python
# Nacos配置
pv_model_name_list = global_config['pv_history']['model_name_list']
Backtesting_days = int(global_config['pv_history']['Backtesting_days'])
load_model_name_list = global_config['load_history']['model_name_list']
```

### 5.2 主要依赖
- **数据库**: `database.db_conn_global` - 数据库连接
- **Redis**: `utils.RedisConn` - 缓存服务
- **Kafka**: `utils.kafka_writer` - 消息队列
- **S3**: `utils.s3_conn` - 模型文件存储
- **深度学习**: `torch`, `multi_model_dispatch`
- **时间处理**: `pytz`, `datetime`
- **数据处理**: `pandas`, `numpy`

---

## 6. 时间处理

### 6.1 夏令时(DST)处理
```python
# 检测夏令时
dst_timezone_set = get_dst_timezones(target_time)
if time_area in dst_timezone_set:
    gap_time = get_loss_hour(time_area, target_date)

# 应用夏令时调整
statistics_time, predictions = check_dst_time(
    statistics_time, predictions, gap_time
)
```

**夏令时影响**:
- 夏令时开始时:某个小时会"消失"(23小时)
- 需要删除对应时间点的预测结果

### 6.2 冬令时(WST)处理
```python
# 检测冬令时
wst_timezone_set = get_wst_timezones(target_time)
if time_area in wst_timezone_set:
    winter_extra_time = get_repeat_hour(time_area, target_date)

# 应用冬令时调整
statistics_time, predictions = check_wst_time(
    statistics_time, predictions, winter_extra_time
)
```

**冬令时影响**:
- 冬令时开始时:某个小时会"重复"(25小时)
- 需要复制对应时间点的预测结果

---

## 7. 错误处理

### 7.1 异常捕获
模块中所有关键操作都包含异常处理:
```python
try:
    # 执行操作
    result = some_operation()
except Exception as e:
    logger.error(f"操作失败: {e}")
    exception_handler.log_exception(e)
    continue  # 继续处理下一个
```

### 7.2 数据验证
```python
# 检查数据充足性
if len(group_sorted) < 288 * 7:
    logger.warning(f"站点{station_id}数据不足,跳过")
    continue

# 检查数据一致性
if len(each_pred) != 288 or len(cur_statistics_time) != 288:
    logger.warning(f"数据长度异常,跳过")
    continue
```

### 7.3 资源清理
```python
# 删除本地模型文件
try:
    os.remove(f'{station_id}_{model}_model.pkl')
except Exception as e:
    logger.error(f"文件删除失败: {e}")

# 显式释放内存
del batch_np, batch_df
gc.collect()
```

---

## 8. 性能优化

### 8.1 批量处理
```python
# 每批处理20个站点
batch_size = 20
station_data_generator = get_area_station_data_his_generator_holiday(
    area, batch_size=20, ...
)
```

### 8.2 并行执行
```python
# 使用线程池
with get_thread_pool_executor() as executor:
    future = executor.submit(worker_function, ...)
    result = future.result()
```

### 8.3 内存管理
```python
# 及时删除大对象
del batch_np
gc.collect()

# 使用copy()避免意外修改
batch_forecast_dlinear_model(batch_np.copy(), ...)
```

---

## 9. 使用示例

### 9.1 基本使用
```python
import threading
from datetime import datetime
import pytz

# 创建线程对象
predictor = LoadModelPredictionHolidayOff(
    time_area="Europe/Berlin",
    mode=2,  # 单次运行
    exp_dlinear=dlinear_experiment,
    exp_timexer=timexer_experiment,
    lock=threading.Lock(),
    global_white_list=[12345, 67890]  # XGBoost白名单
)

# 启动预测
predictor.start()
predictor.join()  # 等待完成
```

### 9.2 定时调度
```python
# 创建定时调度(每天20:00执行)
scheduler = LoadModelPredictionHolidayOff(
    time_area="Asia/Shanghai",
    mode=1,  # 定时调度模式
    exp_dlinear=dlinear_exp,
    exp_timexer=timexer_exp
)
scheduler.start()
# 线程将持续运行,每天定时执行
```

### 9.3 直接调用主函数
```python
from load_pred_holiday_off_schedule import pred_area_load_multi_model_parallel

# 直接调用主预测函数
pred_area_load_multi_model_parallel(
    time_area="America/New_York",
    exp_dlinear=dlinear_exp,
    exp_timexer=timexer_exp,
    lock=threading.Lock(),
    global_white_list=[111, 222, 333]
)
```

---

## 10. 注意事项

### 10.1 数据要求
1. **时间跨度**: 需要至少60天的历史数据
2. **数据密度**: 5分钟间隔,每天288个点
3. **最少数据量**: 过滤节假日后至少保留7天(2016个点)

### 10.2 模型要求
1. **DLinear**: 需要2016个历史点,预测288×2个未来点
2. **TimeXer**: 需要2016个历史点,预测288×2个未来点
3. **XGBoost**: 需要8天历史数据,预测2天

### 10.3 时区处理
1. 所有时间处理都考虑了目标时区
2. 内部统一使用UTC时间戳
3. 输出时转换回目标时区

### 10.4 并发控制
1. XGBoost预测使用独立锁(`lock`)
2. TimeXer预测使用共享锁(`lock2`)
3. 避免死锁:确保锁的获取和释放顺序一致

### 10.5 内存管理
1. 批量处理完成后立即清理内存
2. 大数组使用 `.copy()` 传递给函数
3. 定期调用 `gc.collect()`

---

## 11. 常见问题

### Q1: 为什么需要2016个时间点?
**A**: 2016 = 7天 × 288点/天。深度学习模型(DLinear/TimeXer)需要固定长度的输入,7天的历史数据能够捕获周期性模式。

### Q2: 节假日数据是如何过滤的?
**A**: 通过查询 `station_mode_operation_log` 表,找到节假日开启/关闭记录,构建时间段mask,然后从完整时间轴中移除这些时间点。

### Q3: 为什么XGBoost只预测白名单站点?
**A**: XGBoost模型需要从S3下载,推理较慢。白名单机制允许选择性地为重要站点提供XGBoost预测,其他站点只使用DLinear和TimeXer。

### Q4: 夏令时/冬令时如何处理?
**A**:
- **夏令时**: 删除"消失"的那个小时的预测点(287个点)
- **冬令时**: 复制"重复"的那个小时的预测点(289个点)

### Q5: 如何添加新的预测模型?
**A**:
1. 实现 `batch_forecast_xxx_model()` 函数
2. 在 `predict_area_load_multi_model_parallel_worker()` 中调用
3. 更新 `select_best_model_hour_sum_v1_dst_wst_holiday_off()` 选优逻辑

---

## 12. 相关文档
- [节假日关闭模式工具函数文档](holiday_off_tools.md)
- [多模型调度文档](multi_model_dispatch.md)
- [数据预处理文档](preprocessing_data.md)
- [时间处理工具文档](dst_util.md, wst_util.md)

---

## 附录A: 关键常量

| 常量名 | 值 | 说明 |
|--------|-----|------|
| `target_points` | 2016 | 深度学习模型输入时间点数(7天) |
| `batch_size` | 20 | 每批处理的站点数量 |
| `past_days_num` | 60 | 获取历史数据的天数 |
| `interval` | 300 | 时间间隔(秒,5分钟) |
| `points_per_day` | 288 | 每天的时间点数(5分钟间隔) |
| `pred_days` | 2 | 预测未来天数 |

---

## 附录B: 数据库表结构

### station_mode_operation_log 表
```sql
CREATE TABLE station_mode_operation_log (
    dt DATE,                    -- 日期
    mode_type INT,              -- 模式类型: 2=节假日模式
    operation_type INT,         -- 操作类型: 0=关闭, 1=开启
    operation_time TIMESTAMP,   -- 操作时间
    record_time TIMESTAMP,      -- 记录时间
    station_id BIGINT           -- 站点ID
);
```

### ai_station 表
```sql
CREATE TABLE ai_station (
    station_id BIGINT PRIMARY KEY,  -- 站点ID
    time_area VARCHAR(50),          -- 时区
    feature_a INT,                  -- 特征签名
    ...
);
```

---

## 版本历史
- **v1.0** (2025-12-24): 初始版本,完整功能实现