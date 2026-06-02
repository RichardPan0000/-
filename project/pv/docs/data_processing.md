# PV预测系统 - 数据处理文档

## 1. 数据源概览

| 数据源 | 存储 | 用途 |
|--------|------|------|
| NWP（数值天气预报） | StarRocks | 气象特征输入 |
| PV历史发电数据 | StarRocks | 模型训练/预测输入 |
| 站点配置信息 | MySQL | 站点元数据 |
| 模型参数 | S3 | XGBoost模型存储 |
| 深度模型Checkpoint | S3 + 本地 | TimeXer权重 |
| 缓存数据 | Redis | 中间结果/标记 |
| 预测结果 | Kafka + Redis | 输出 |

## 2. NWP数据处理

### 2.1 数据来源

6个气象源的格点化数值天气预报数据：

| 气象源 | 缩写 | 覆盖范围 |
|--------|------|---------|
| GFS | gfs | 全球 |
| DWD-ICON | dwd-icon | 全球 |
| Météo-France | meteofrance | 全球 |
| GEM | gem | 全球 |
| ECMWF | ecmwf | 全球 |
| BOM | bom | 澳洲 |

### 2.2 NWP特征列表

每个气象源提供以下14个特征：

```python
ori_feature = [
    'shortwave_radiation_instant',      # 瞬时短波辐射
    'shortwave_radiation',              # 短波辐射
    'direct_radiation_instant',         # 瞬时直接辐射
    'direct_radiation',                 # 直接辐射
    'direct_normal_irradiance_instant', # 瞬时法向直接辐射
    'direct_normal_irradiance',         # 法向直接辐射
    'apparent_temperature',             # 体感温度
    'temperature_2m',                   # 2米温度
    'relativehumidity_2m',              # 2米相对湿度
    'precipitation',                    # 降水量
    'cloudcover',                       # 总云量
    'cloudcover_low',                   # 低云量
    'cloudcover_high',                  # 高云量
    'weather_code'                      # 天气码
]
```

**关键辐射特征（PV预测核心）：**
```python
pv_key_feature = [
    'shortwave_radiation_instant',
    'shortwave_radiation',
    'direct_radiation_instant',
    'direct_radiation',
    'direct_normal_irradiance_instant',
    'direct_normal_irradiance'
]
```

### 2.3 NWP数据加载

**批量加载函数：** `IO.load_total_nwp_data_from_db_batch_grids`

```
输入：
  - 格点列表 [(lat, lon, tz_name), ...]
  - 时间范围 [start_date, end_date]
  - 气象源列表 source_list
  - 是否日内 is_intraday

输出：
  - Dict[Tuple[float, float], pd.DataFrame]
  - key: (lat, lon) 格点坐标
  - value: 时间索引的NWP DataFrame
```

**数据表：** `sigen_ai.weather_info_latest_forecast`

### 2.4 NWP数据处理流程（深度模型）

```
1. 从数据库按格点批量加载
   └── 时间范围：历史N天 + 未来2天

2. prepare_nwp_data_batch()
   ├── 时间对齐
   ├── 缺失值处理
   └── 按预测日期切分

3. build_nwp_arrays()
   ├── 构建第一天NWP: nwp_np_d1 [B, 168+35, C_wea]
   │   ├── encoder部分: 过去168小时（7天）
   │   └── decoder部分: 未来35小时
   └── 构建第二天NWP: nwp_np_d2 [B, 168+35, C_wea]
       ├── encoder部分: 过去168小时
       └── decoder部分: 未来35小时（偏移1天）
```

### 2.5 多气象源 vs EC单气象源

**多气象源（主流程）：**
- 使用6个气象源
- 每源取4个关键特征
- C_wea = 6 × 4 = 24

**EC单气象源：**
- 仅使用 ecmwf
- 取6个辐射特征
- C_wea = 6

## 3. PV历史数据处理

### 3.1 数据来源

**数据表：** `sigen_device.station_statistics_min`

**关键字段：**
- `station_id`: 站点ID
- `statistics_time`: 统计时间（5分钟粒度）
- `filtered_pv_total_power`: 过滤后的PV总功率（W）
- `dt`: 日期分区

### 3.2 绿电 vs 非绿电站点

通过 `feature_signature_a` 位标记区分：

```python
def check_station_feature_a(feature_num):
    feature_code = format(int(feature_num) & 0xFFFFFFFF, '032b')
    if feature_code[-3] == '1' or feature_code[-5] == '1':
        return True  # 绿电站点
    return False     # 非绿电站点
```

**绿电站点数据处理：**
```python
# 从 station_statistics_min 表读取
station_his['pv_power_generation'] = filtered_pv_total_power * 5 / 60 * 100
# 单位转换：W → kWh（5分钟功率转换为发电量百分比）
```

**非绿电站点数据处理：**
```python
# 从 IO.load_station_data_for_deep 读取
# 已经是处理好的 pv_power_generation 格式
```

### 3.3 批量数据加载（生成器模式）

`get_latlon_area_station_data_his_generator` 使用生成器模式避免一次性加载所有数据：

```python
def process_stations_in_batches(not_green_list, green_list, batch_size, end_date, past_days_nums):
    # 先处理非绿电站点
    for batch in batch_iterable(not_green_list, batch_size):
        yield get_not_green_station_list_data(batch, end_date, past_days_nums)
    # 再处理绿电站点
    for batch in batch_iterable(green_list, batch_size):
        yield get_green_station_list_data(batch, end_date, past_days_nums)
```

每批返回 `(station_df, filtered_station_list)` 元组。

### 3.4 深度模型数据预处理

**`prepare_data_for_multi_model` 函数：**

```
输入：batch_df (多站点PV数据), station_mode_list, target_points=2304

处理步骤：
1. 按站点分组
2. 时间对齐到 target_points 个点（8天 × 288 = 2304）
3. 构建numpy数组 [B, 2304, 5]:
   ├── dim 0: timestamp (unix seconds)
   ├── dim 1: pv_power_generation
   ├── dim 2: r_pm_am（下午/上午功率比）
   ├── dim 3: r_tail（傍晚/正午功率比）
   └── dim 4: t_cog_norm（功率重心时间）
4. 缺失值填充：同时刻历史均值
5. PV归一化：除以100

输出：batch_np [B, 2304, 5], station_mode_list_filtered
```

### 3.5 TimeXer模型输入准备

```python
# 取最后2016个点（7天）
timexer_input = batch_np[:, -2016:, :]

# PV值再次归一化（除以100）
timexer_input[:, :, 1] = timexer_input[:, :, 1] / 100.0

# NWP取最后 168+35 = 203 个小时级数据点
batch_nwp_np = batch_nwp_np[:, -(168 + 35):, :]
```

## 4. 特征工程

### 4.1 XGBoost特征工程

**`feature_engineering_xgb5fold_simple_v1`（简单特征）：**
- 直接使用NWP原始特征
- 适用于 GridXGBSimpleBayes4fold-dayahead

**`feature_engineering_xgb5fold_simple_v3`（复杂特征）：**
- NWP原始特征
- 衍生统计特征（滑动窗口均值、最大值等）
- 适用于 GridXGBSimplePCA5Fold-dayahead

**`feature_engineering_pca`（PCA降维）：**
```python
pca_df, scaler_para, pca_para = features.feature_engineering_pca(nwp_data.dropna())
feature_data_model3 = pd.concat((feature_data_model3, pca_df), axis=1).interpolate().ffill().bfill()
```

### 4.2 深度学习时间特征

**时间编码方式（timeenc=2，sin/cos编码）：**

```python
# time_features_sin_cos 生成的特征维度
# 输入：datetime序列
# 输出：[T, C_time] 的sin/cos编码矩阵
```

**Dataset_pv_wea_online 数据集：**
```python
def __getitem__(self, index):
    return self.pv_data[index, ...],    # PV序列 [2016]
           self.data_stamp,              # 时间特征 [2016, C_time]
           self.nwp_data[index, ...]     # NWP数据 [203, C_wea]
```

**Dataset_pv_wea_online_feamore 数据集（含形状特征）：**
```python
def __read_data__(self):
    # 时间特征 + 形状特征拼接
    self.x_enc_mark = np.concatenate([self.data_stamp, self.data_np[0, :, 2:]], axis=-1)
    # x_enc_mark: [2016, C_time + 3]  (3 = r_pm_am, r_tail, t_cog_norm)
```

### 4.3 形状特征计算

在 `model_dispatch.py` 中计算：

```python
# r_pm_am: 下午功率总和 / 上午功率总和
# 反映站点朝向（东向/西向偏差）

# r_tail: 傍晚时段功率 / 正午时段功率
# 反映站点是否有下午遮挡

# t_cog_norm: 功率重心时间（归一化到[0,1]）
# t_cog = Σ(t × P(t)) / Σ(P(t))
# 反映发电高峰时段的偏移
```

## 5. 日间Mask计算

### 5.1 `day_mask_reshape7` 算法

```
输入：batch_np [B, 2304, 5]（8天历史数据）

算法：
1. 取最后7天数据（2016个点）
2. reshape为 [B, 7, 288]（7天 × 288点/天）
3. 对每个时间点（0-287）：
   统计7天中有多少天该时间点 PV > threshold
4. 如果 ≥ 3天有值 → mask=1（白天）
5. 否则 → mask=0（夜间）

输出：day_mask [B, 288]
```

### 5.2 应用Mask

```python
first_day_pred = prediction_result[:, :288, :]
day_mask = day_mask_reshape7(batch_np)
first_day_masked = first_day_pred * day_mask[..., None]
```

## 6. 夏令时/冬令时处理

### 6.1 DST检测

```python
# 检测预测日期是否跨越夏令时切换
# 如果是，计算时间跳变量 gap_time
```

### 6.2 时间修正 `apply_dst_wst_correction`

```python
def apply_dst_wst_correction(pred_time_list, pred_values, time_area, pred_date):
    # 检测该日期是否有DST切换
    # 如果有：
    #   - 春季前进（Spring Forward）：删除跳过的时间点
    #   - 秋季后退（Fall Back）：复制重叠的时间点
    return corrected_time_list, corrected_values
```

## 7. 数据库交互

### 7.1 数据库连接池

```python
class DataBase:
    def __init__(self):
        # MySQL连接池（站点配置）
        self.mysql_pool = PooledDB(
            creator=pymysql, maxconnections=10, mincached=1,
            blocking=True, ping=1,
            host=mysql_conf['host'], port=3306,
            user=mysql_conf['user'], password=mysql_conf['password'],
            database=mysql_conf['database']
        )
        # StarRocks连接池（时序数据）
        self.starrocks_pool = PooledDB(
            creator=pymysql, maxconnections=10, mincached=1,
            blocking=True, ping=1,
            host=starrocks_conf['host'], port=9030,
            user=starrocks_conf['user'], password=starrocks_conf['password'],
            database=starrocks_conf['database']
        )
```

### 7.2 关键查询方法

| 方法 | 数据库 | 用途 |
|------|--------|------|
| `get_ai_area()` | MySQL | 获取所有AI时区 |
| `get_ai_lat_lon_area(area)` | MySQL | 获取时区下所有格点 |
| `get_ai_station_from_lat_lon(lat_lon)` | MySQL | 获取格点下站点 |
| `get_station_his_filtered_data_batch()` | StarRocks | 批量获取PV历史 |
| `get_station_prediction_history()` | StarRocks | 获取预测历史 |
| `get_batch_stations_prediction_history()` | StarRocks | 批量获取预测历史 |
| `get_ec_ai_station()` | MySQL | 获取EC模式站点 |
| `get_ec_ai_lat_lon_area()` | MySQL | 获取EC模式格点 |

### 7.3 批量查询规则

- **StarRocks**：一次不超过50个站点
- **MySQL**：一次不超过100个站点
- 使用 `IN (...)` 语法批量查询

### 7.4 sub_working_mode 兼容性

```python
def get_sub_working_mode_condition(self, use_sub_working_mode=True):
    # 1. 未启用时返回空
    # 2. 线上字段未发布时自动降级（只按working_mode=1过滤）
    # 3. 字段存在时返回 "AND sub_working_mode = 1"
```

## 8. 外部存储交互

### 8.1 S3存储

**多区域配置：**
- EU: `eu-central-1` / `sigen-data`
- Asia: `ap-southeast-1` / `sigen-data-ap`
- Australia: `ap-southeast-2` / `sigen-data-aus`
- China: `cn-northwest-1` / `sigen-data-cn`
- US: `us-west-2` / `sigen-data-us`
- Japan: `ap-northeast-1` / `sigen-data-jp`

**用途：**
- XGBoost模型存储/加载
- TimeXer checkpoint下载

### 8.2 Redis缓存

| Key模式 | 用途 | 过期时间 |
|---------|------|---------|
| `sigen_ai_infos:nwp_begin_date:{lat}:{lon}:{tz}` | 格点NWP开始日期 | 永久 |
| `pv-pred:{station_id}:pv_max_last_one_month` | 站点历史最大功率 | 永久 |
| `pv-pred:snow-list:{area}:{date}` | 积雪站点列表 | 7天 |
| `pv-pred:snow-list-advance:{area}:{date}` | 提前积雪站点 | 7天 |
| `pv-pred:snow-list-continue:{area}:{date}` | 延续积雪站点 | 7天 |
| `station:predict:pv:select_best_7:{sid}:{date}` | 最优预测结果 | - |

### 8.3 Kafka输出

**Topic：** `pv_prediction_topic_load`

**消息格式：**
```
{pred_date};{station_id};{predict_type};{model_name};{interval};{record_time};{statistics_time_list};{pred_values_list}
```

**示例：**
```
2026-05-13;12024061505406;1;Timexer-pv-dayahead-mae-mape;0.1;1715500000;[1715558400,1715558700,...];[0.0,0.0,0.12345,...]
```

## 9. 数据流总览

```
┌──────────────────────────────────────────────────────────────┐
│                        数据输入层                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  MySQL                    StarRocks                  Redis   │
│  ├─ station_config_cdc    ├─ station_statistics_min   ├─ NWP开始日期│
│  │  (站点配置)            │  (5分钟PV数据)            ├─ 历史最大功率│
│  └─ 格点/时区信息         ├─ weather_info_latest      └─ 积雪标记  │
│                           │  (NWP天气预报)                    │
│                           └─ pv_prediction_history            │
│                              (预测历史)                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        数据处理层                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  NWP处理                  PV数据处理              特征工程     │
│  ├─ 批量加载格点数据       ├─ 绿电/非绿电分流      ├─ XGB特征   │
│  ├─ 多源合并              ├─ 时间对齐(2304点)     │  ├─ simple_v1│
│  ├─ 时间切分(enc/dec)     ├─ 缺失值填充           │  ├─ simple_v3│
│  └─ 上采样(12x)          ├─ 归一化               │  └─ PCA     │
│                           └─ 形状特征计算          └─ 时间编码   │
│                                                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        模型推理层                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  XGBoost预测              TimeXer预测             规则模型     │
│  ├─ 从S3加载模型          ├─ 加载checkpoint        └─ 基线预测  │
│  ├─ 特征输入              ├─ 构建batch输入                    │
│  └─ K折平均输出           ├─ GPU/CPU推理                      │
│                           └─ 反归一化                         │
│                                                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        后处理层                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ├─ 负值截断 (< 0 → 0)                                       │
│  ├─ 日间mask应用（夜间置0）                                    │
│  ├─ 夏令时/冬令时修正                                         │
│  ├─ 积雪场景处理（功率压制）                                    │
│  ├─ 限电场景处理                                              │
│  └─ 最优模型选择（7天回测）                                    │
│                                                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        数据输出层                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Kafka                              Redis                    │
│  ├─ PV_RESULT_TOPIC                 └─ select_best_7 结果    │
│  │  (各模型预测结果)                    (Protobuf序列化)       │
│  └─ 消息格式:                                                │
│     date;sid;type;model;interval;                            │
│     record_time;time_list;value_list                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 10. 配置管理

### 10.1 静态配置 (`config.conf`)

- S3多区域凭证
- Kafka连接信息
- 模型特征列表
- 线程池大小
- 回测天数
- Redis连接信息

### 10.2 动态配置 (Nacos `global_config`)

- MySQL/StarRocks连接信息
- `deep_model_flag`: 是否启用深度模型
- `pre_pred_flag`: 是否启用预执行模式
- `run_mode.pre_pred_signal`: 预执行信号
- `run_mode.pre_pred_test_flag`: 测试标记
