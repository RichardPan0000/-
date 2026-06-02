# PV预测系统 - 业务流程文档

## 1. 系统概述

本系统是一个面向全球多时区的光伏（PV）发电功率预测平台，服务百万级用户。系统基于 XGBoost 和 TimeXer 深度学习模型的多模型集成方案，实现对光伏电站未来1-2天发电功率的精准预测。

**核心能力：**
- 日前预测（Day-ahead）：预测明天和后天的发电功率
- 日内修正（Intraday）：根据当天实际发电量修正预测
- 积雪场景处理：识别积雪/融雪站点并调整预测
- 限电场景处理（Clipping）：处理限电站点的预测
- 最优模型选择：基于历史回测自动选择最优模型

## 2. 系统入口与调度架构

### 2.1 启动入口

```
main_test_2.py → main_schedule() (model_dispatch.py)
```

### 2.2 调度器架构

系统使用 APScheduler 创建三个独立的 BackgroundScheduler：

| 调度器 | 职责 | 线程池大小 |
|--------|------|-----------|
| main_train_scheduler | 模型训练任务 | 1 |
| main_predict_scheduler | 预测任务 | 2 |
| main_record_scheduler | 数据记录任务 | 2 |

### 2.3 定时任务总览

每个时区（area）独立注册以下定时任务（时间为该时区本地时间）：

| 任务 | 执行时间 | 函数 | 说明 |
|------|---------|------|------|
| 日前训练 | 00:15 | `train_area_parallel_only_xgb5fold` | XGBoost模型训练 |
| 日内训练 | 02:25 | `train_area_intraday_parallel` | 日内XGBoost训练 |
| 日前预测 | 19:15 | `predict_area_multi_model_dayahead` | 深度+XGB多模型预测 |
| 日内预测 | 11:35 | `predict_area_intraday` | XGBoost日内预测 |
| EC日前预测 | 22:15 | `predict_area_ec_dayahead` | EC单气象源日前预测 |
| EC日内修正 | 09:15/12:15 | `predict_area_ec_intraday` | EC模型日内修正 |
| 限电预测 | 22:15 | `predict_area_multi_model_dayahead_clipping` | 限电站点预测 |
| 积雪标记 | 20:45 | `record_pv_snow_flag` | 记录积雪站点 |
| 积雪预测 | 22:55 | `predict_area_snow_on_dayahead` | 积雪站点预测 |
| 融雪预测 | 23:15 | `predict_area_snow_off_dayahead` | 融雪站点预测 |
| PV最大值记录 | 01:15 | `record_pv_max_last_one_mon` | 记录历史最大功率 |
| Bad Case白名单 | 周六 00:05 | `schedule_weekly_pv_bad_case_white_list` | 计算预测效果差的站点 |
| 区域更新 | 00:00 | `update_area` | 动态增删时区 |

### 2.4 动态区域管理

- 系统每天 00:00 从数据库查询最新的时区列表
- 自动为新增时区注册所有定时任务
- 自动移除已下线时区的任务
- 无需重启服务即可扩展新区域

## 3. 日前预测流程（核心流程）

### 3.1 主流程 `predict_area_multi_model_dayahead`

```
┌─────────────────────────────────────────────────────────┐
│                    日前预测主流程                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 获取该时区所有格点列表 (lat_lon_areas)                │
│     └─ 按 batch_size=32 分批处理                         │
│                                                         │
│  2. 每批格点：                                           │
│     ├─ 加载NWP数据（历史N天 + 未来2天）                    │
│     ├─ 获取格点下所有站点信息                              │
│     └─ 收集站点PV历史数据（生成器模式）                     │
│                                                         │
│  3. 深度模型预测 (predict_area_deep_dayahead)             │
│     ├─ 数据预处理（对齐2304点、归一化）                     │
│     ├─ TimeXer模型推理（第一天 + 第二天）                   │
│     ├─ 应用日间mask                                      │
│     └─ 发送结果到Kafka                                   │
│                                                         │
│  4. 等待Kafka写入完成（62秒）                             │
│                                                         │
│  5. 最优模型选择 (select_best_models_for_area)            │
│     ├─ 获取所有候选模型的历史预测                          │
│     ├─ 计算过去7天的回测误差                              │
│     ├─ 选择最优模型                                      │
│     └─ 写入Kafka和Redis                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 深度模型预测详细流程

```
predict_area_deep_dayahead
│
├── Process 1. 准备夏令时信息
│   └── 检测预测日期是否跨越夏令时切换
│
├── Process 2. 收集站点PV历史数据
│   ├── 区分绿电/非绿电站点
│   ├── 批量加载5分钟粒度数据
│   └── 生成器模式逐批返回
│
├── Process 3. 数据预处理
│   ├── 对齐到2304个时间点（8天 × 288点/天）
│   ├── 缺失值填充（同时刻历史均值）
│   ├── 计算形状特征（r_pm_am, r_tail, t_cog_norm）
│   └── PV值归一化（除以100）
│
├── Process 4. NWP数据处理
│   ├── 按站点经纬度匹配格点NWP
│   ├── 构建encoder NWP数组（168小时历史）
│   └── 构建decoder NWP数组（35小时未来）
│
├── Process 5. TimeXer模型推理
│   ├── 第一天预测：batch_np[:, -2016:, :] + nwp_d1
│   ├── 第二天预测：batch_np[:, -2016:, :] + nwp_d2
│   └── 线程锁保护并发访问
│
├── Process 6. 后处理
│   ├── 负值截断为0
│   ├── 应用日间mask（夜间置0）
│   └── 夏令时/冬令时时间修正
│
└── Process 7. 输出
    ├── 发送到Kafka（PV_RESULT_TOPIC）
    └── 模型名：Timexer-pv-dayahead-mae-mape
```

### 3.3 最优模型选择流程

```
select_best_models_for_area
│
├── 获取候选模型列表
│   └── [rule, Timexer-pv-dayahead-mae-mape, GridXGBSimplePCA5Fold-dayahead]
│
├── 获取过去7天各模型的预测历史
│
├── 获取过去7天的实际发电数据
│
├── 逐站点计算各模型误差
│   ├── MAE（平均绝对误差）
│   └── RMSE（均方根误差）
│
├── 选择误差最小的模型作为最优
│   └── 处理夏令时时间偏移
│
└── 输出最优预测结果
    ├── Kafka: model_name='select_best_7'
    └── Redis: station:predict:pv:select_best_7:{station_id}:{date}
```

## 4. EC单气象源预测流程

### 4.1 EC日前预测 `predict_area_ec_dayahead`

针对 `working_mode=1` 且 `sub_working_mode=1` 的站点，仅使用 ECMWF 气象源：

- NWP只用 ecmwf 单一来源（6个特征）
- 使用专用checkpoint: `checkpoints_ec_mae_mape.pth`
- 模型名: `Timexer-pv-dayahead-ec-mae-mape`
- 不执行XGB单模型预测

### 4.2 EC日内修正 `predict_area_ec_intraday`

基于当天实际发电量与EC日前预测的比值进行修正：

```
1. 获取当天0点到当前时刻的真实发电数据
2. 获取EC日前模型今天的预测值
3. 计算累计发电量比例 ratio = sum_real / sum_pred
4. 边界处理：ratio 裁剪到 [0.1, 10.0]
5. 整天预测值 × ratio = 修正后预测值
6. 写入Kafka和Redis
```

## 5. 积雪场景处理流程

### 5.1 积雪标记 `record_pv_snow_flag`

每天 20:45 执行，识别明天可能积雪的站点：

```
1. 获取所有格点
2. 查询今天和明天的气象数据（温度、辐照、天气码）
3. 过滤条件：明天平均温度 ≤ 2°C
4. 判断积雪类型：
   ├── 提前积雪（advance）：今天18:00到明天16:00累计≥6小时雪码
   │   AND 明天最低温<-1°C AND 明天最大辐照<200W
   └── 延续积雪（continue）：
       ├── 严格条件：今天辐照>150W AND 实发<10%历史最大 AND 明天最低温<-5°C
       └── 常规条件：今天辐照>150W AND 实发<10%历史最大
           AND (今天最低温<-1°C OR 有雪码) AND 明天最低温<-1°C AND 明天辐照<250W
5. 写入Redis（7天过期）：
   ├── pv-pred:snow-list-advance:{area}:{date}
   ├── pv-pred:snow-list-continue:{area}:{date}
   └── pv-pred:snow-list:{area}:{date}（总列表）
```

### 5.2 积雪预测 `predict_area_snow_on_dayahead`

对已标记的积雪站点进行功率压制：

```
1. 从Redis读取积雪站点列表（advance + continue）
2. 读取历史最大PV功率
3. 读取现有预测结果（优先级：Redis select_best_7 > DB select_best_7 > rule > Timexer）
4. 放缩预测值：
   ├── 提前积雪站点：放缩到历史最大值的 5%
   └── 延续积雪站点：放缩到历史最大值的 3%
5. 写入Kafka（snow_on_pv_dayahead + select_best_7）
6. 写入Redis
```

### 5.3 融雪预测 `predict_area_snow_off_dayahead`

对从积雪恢复的站点进行功率恢复预测：

```
1. 计算融雪站点 = 过去7天有积雪 - 今天有积雪
2. 查找7天好天气历史数据（日最大功率在历史最大值的15%-90%之间）
3. 用好天气历史数据替换输入，运行TimeXer预测
4. 应用日间mask和夏令时修正
5. 写入Kafka和Redis
```

## 6. 限电预测流程 `predict_area_multi_model_dayahead_clipping`

```
1. 查询近7天出现过 predict_type=4 的站点
2. 获取这些站点的限电预测数据
3. 用限电预测数据替换历史PV数据中对应日期的值
4. 以替换后的数据作为输入运行TimeXer预测
5. 确保288个点（插值处理）
6. 写入Kafka和Redis
```

## 7. 训练流程

### 7.1 日前训练 `train_area_parallel_only_xgb5fold`

```
1. 获取格点列表
2. 逐格点处理：
   ├── 加载NWP数据（过去365天）
   ├── 特征工程：
   │   ├── feature_engineering_xgb5fold_simple_v1（简单特征）
   │   ├── feature_engineering_xgb5fold_simple_v3（复杂特征）
   │   └── feature_engineering_pca（PCA降维）
   ├── 加载站点PV历史数据
   ├── 归一化：pv_power / (max_power / 20)
   └── 训练模型：
       ├── GridXGBSimpleBayes4fold-dayahead（4折CV + 贝叶斯优化）
       └── GridXGBSimplePCA5Fold-dayahead（5折CV + PCA特征）
3. 模型存储到S3
```

### 7.2 日内训练 `train_area_intraday_parallel`

流程与日前训练类似，但使用日内数据和日内模型名称。

## 8. Bad Case白名单

每周六计算预测效果不佳的站点：

```
1. 查询过去30天 Timexer-pv-dayahead-mae-mape 模型的精度数据
2. 筛选 rmse_acc_1hour < 0.4 的日期
3. 统计每站点有多少天 < 0.4
4. 筛选 ≥ 3 天的站点
5. 安全检查：bad_case数量 > 总站点数 × 50% 则拒绝上报
6. 结果用于XGBoost预测时的白名单补充
```

## 9. 数据输出格式

### 9.1 Kafka消息格式

```
{pred_date};{station_id};{predict_type};{model_name};{interval};{record_time};{statistics_time_list};{pred_values_list}
```

- Topic: `pv_prediction_topic_load`
- predict_type: 1=正常预测, 4=限电预测

### 9.2 Redis存储格式

- Key: `station:predict:pv:select_best_7:{station_id}:{date}`
- Value: Protobuf序列化的 `PVSelectBest7 { interval=5min, values=[...288个值...] }`

## 10. 关键业务规则

1. **时间粒度**：所有预测输出为5分钟粒度，每天288个点
2. **日间mask**：夜间时段预测值强制为0，mask基于过去7天历史数据计算
3. **夏令时处理**：自动检测DST切换，调整时间戳和预测值
4. **批量处理**：StarRocks一次不超过50个站点，MySQL一次不超过100个
5. **线程安全**：TimeXer模型调用使用 `timexer_model_lock` 保护
6. **内存管理**：每批处理后执行 `gc.collect()`，避免内存泄漏
