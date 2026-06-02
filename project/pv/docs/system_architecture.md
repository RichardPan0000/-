# PV预测系统 - 系统架构文档

## 1. 部署架构

### 1.1 运行环境

- **容器化部署**：Docker（见 `src/Dockerfile`）
- **集群内服务发现**：Kubernetes Service
- **配置中心**：Nacos
- **多区域部署**：支持EU、US、Asia、Australia、China、Japan

### 1.2 基础设施依赖

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ PV Prediction│  │   MySQL     │  │  StarRocks  │    │
│  │   Pod        │  │   Master    │  │     FE      │    │
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                  │                 │          │
│  ┌──────┴───────┐  ┌──────┴──────┐  ┌──────┴──────┐    │
│  │    Redis     │  │    Kafka    │  │     S3      │    │
│  │   Cluster    │  │   Cluster   │  │  (AWS/多区域)│    │
│  └──────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.3 服务地址（集群内）

| 服务 | 地址 | 端口 |
|------|------|------|
| MySQL | mysql-master.mysql.svc.cluster.local | 3306 |
| StarRocks | fe-domain-search.starrocks.svc.cluster.local | 9030 |
| Redis | redis-hs.redis.svc.cluster.local | 6379 |
| Kafka | kafka-hs.kafka.svc.cluster.local | 9092 |

## 2. 代码结构

```
src/
├── main_test_2.py                    # 程序入口
├── model_dispatch.py                 # 核心调度（~4600行）
│   ├── main_schedule()               # 主调度器
│   ├── train_area_parallel_*()       # 训练流程
│   ├── predict_area_*()              # 预测流程
│   └── select_best_*()              # 最优选择
├── multi_deep_model_dispatch.py      # TimeXer模型管理
├── database.py                       # 数据库连接池
├── config.conf                       # 静态配置
├── get_batch_station_pvdata_handler.py  # 批量数据加载
├── get_pv_white_list_bad_case.py     # Bad Case白名单
│
├── models/                           # ML模型定义
│   ├── xgb_model.py                  # XGBoost模型
│   └── __init__.py
│
├── schedulers/                       # 调度任务模块
│   ├── predict_area_ec_dayahead.py   # EC日前预测
│   ├── predict_area_ec_intraday.py   # EC日内修正
│   ├── predict_snow_on_dayahead.py   # 积雪预测
│   ├── predict_snow_off_dayahead.py  # 融雪预测
│   ├── clipping_predict.py           # 限电预测
│   ├── record_pv_max.py             # PV最大值记录
│   ├── record_pv_snow_flag.py       # 积雪标记
│   └── __init__.py
│
├── tsxer_main/                       # TimeXer深度学习框架
│   ├── plugin/
│   │   └── model.py                  # 模型加载/推理入口
│   ├── models/
│   │   ├── TimeXer_beta.py           # 基础模型
│   │   ├── TimeXer_beta_feamore.py   # 增强特征模型
│   │   └── TimeXer_omega.py          # 形状特征模型
│   ├── layers/                       # 网络层组件
│   │   ├── SelfAttention_Family.py   # 注意力机制
│   │   ├── Embed.py                  # 嵌入层
│   │   ├── revin.py                  # RevIN归一化
│   │   └── ...
│   ├── data_provider/
│   │   ├── data_loader.py           # 数据集定义
│   │   └── data_factory.py          # 数据集工厂
│   ├── exp/
│   │   ├── exp_basic.py             # 实验基类
│   │   └── exp_long_term_forecasting.py  # 长期预测实验
│   ├── utils/
│   │   ├── timefeatures.py          # 时间特征工具
│   │   ├── interp.py                # 插值工具
│   │   └── smooth_tools.py          # 平滑工具
│   └── checkpoints/                  # 模型权重（本地缓存）
│       ├── timexer/
│       └── timexer_ec/
│
├── protos/
│   ├── pv.proto                      # Protobuf定义
│   └── pv_pb2.py                     # 生成的Python代码
│
├── utils/                            # 工具模块（未在树中展示）
│   ├── IO.py                         # S3/Kafka/Redis连接
│   ├── features.py                   # 特征工程
│   ├── functions.py                  # 通用函数
│   ├── nacos_config.py              # Nacos配置加载
│   ├── load_weather.py              # 天气数据加载
│   └── exception_handler.py         # 异常处理
│
└── check/                            # 检查/调试工具
    ├── check_redis.py
    ├── check_snow_off.py
    ├── check_snow_on.py
    └── stat_pv_max_top10_threshold.py
```

## 3. 线程模型

### 3.1 调度器线程池

```python
executors_train = {'default': ThreadPoolExecutor(thread_train_limit)}   # 1线程
executors_predict = {'default': ThreadPoolExecutor(thread_predict_limit)} # 2线程
executors_record = {'default': ThreadPoolExecutor(2)}                    # 2线程
executors_update = {'default': ThreadPoolExecutor(1)}                    # 1线程
```

### 3.2 并发控制

```
┌─────────────────────────────────────────────────────────┐
│                    线程安全机制                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  timexer_model_lock (threading.Lock)                    │
│  └── 保护TimeXer模型的并发访问                            │
│      每次推理前 acquire，推理后 release                    │
│                                                         │
│  模型实例隔离                                            │
│  └── 每次调用创建独立的 Exp_Long_Term_Forecast 实例       │
│      避免多线程共享模型状态                               │
│                                                         │
│  数据库连接池                                            │
│  └── PooledDB 管理连接复用                               │
│      maxconnections=10，blocking=True                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.3 最优模型选择并行度

根据站点数量动态调整：
- \> 3000 站点：12线程
- \> 1000 站点：8线程
- ≤ 1000 站点：4线程

## 4. 内存管理策略

### 4.1 批量处理

- 格点按 batch_size=32 分批
- 站点数据通过生成器逐批加载
- NWP数据按格点批次加载

### 4.2 主动清理

```python
def memory_cleanup():
    gc.collect()
    psutil.Process().memory_info()

# 每批处理后清理
del batch_np, batch_nwp_np
gc.collect()

# 每个格点批次后清理
cleanup_memory_caches(all_nwp_cache, all_station_his_data, {}, time_area)
```

### 4.3 模型实例生命周期

```python
# TimeXer：用完即销毁
args = get_timexer_args(model='TimeXer_omega')
exp = Exp_Long_Term_Forecast(args)
preds = exp.test_online_pv(...)
del exp, timexer_cks, args
gc.collect()
```

## 5. 容错与可靠性

### 5.1 错误隔离

- 单个站点失败不影响整批处理
- 单个格点批次失败不影响其他批次
- 单个时区失败不影响其他时区

### 5.2 数据库重连

```python
def db_reconnect(self):
    # 重建MySQL和StarRocks连接
    # 连接池自动管理（ping=1 自动检测断连）
```

### 5.3 模型下载容错

```python
# 文件不存在或为空时重新下载
if not check_path.exists() or check_path.stat().st_size == 0:
    download_model_checkpoints_from_s3(model_name_list=[model_name])
```

### 5.4 sub_working_mode 向后兼容

```python
# 线上字段未发布时自动降级
if not self.has_station_config_cdc_column('sub_working_mode'):
    return ""  # 不加过滤条件，降级为仅按 working_mode=1
```

## 6. 监控与可观测性

### 6.1 日志

- 使用 Python `logging` 模块
- 级别：INFO（正常流程）、WARNING（可恢复异常）、ERROR（需关注）
- 关键节点打印内存使用：`functions.show_memory_info()`

### 6.2 关键日志点

```
[EC日前] area=xxx, grids=N, stations=M
[EC日内修正] 站点xxx kafka写入成功
积雪站点日前预测处理完成，提前积雪: X, 延续积雪: Y
```

## 7. 配置热更新

### 7.1 Nacos动态配置

通过 `utils/nacos_config.py` 加载，支持运行时更新：
- 数据库连接信息
- 功能开关（deep_model_flag, pre_pred_flag）
- 运行模式参数

### 7.2 区域动态管理

```python
def update_area():
    # 每天00:00执行
    ai_area_new = db_conn_global.get_ai_area()
    current_areas = get_registered_areas()
    
    # 移除已下线区域
    for area in current_areas - ai_area_new_set:
        remove_area_jobs(area)
    
    # 添加新区域
    for area in ai_area_new:
        if area not in current_areas:
            ensure_area_jobs(area)
```

## 8. 扩展性设计

### 8.1 新增时区

无需代码修改，只需在数据库 `station_config_cdc` 中添加新时区的站点记录，系统会在下次 `update_area()` 时自动发现并注册任务。

### 8.2 新增模型

1. 实现模型训练/预测逻辑
2. 将模型名添加到 `model_name_list` 配置
3. 确保预测结果写入Kafka（统一格式）
4. 最优选择模块会自动纳入新模型进行比较

### 8.3 新增气象源

1. 在 `config.conf` 的 `weather_source` 列表中添加
2. 确保数据库中有对应气象源的数据
3. 特征工程模块会自动处理新气象源

## 9. 关键依赖

### 9.1 Python依赖

| 包 | 用途 |
|----|------|
| xgboost | XGBoost模型 |
| torch | PyTorch深度学习 |
| pandas/numpy | 数据处理 |
| apscheduler | 定时任务调度 |
| pymysql | MySQL/StarRocks连接 |
| dbutils | 数据库连接池 |
| kafka-python | Kafka消息 |
| redis | Redis缓存 |
| boto3 | AWS S3 |
| bayes_opt | 贝叶斯优化 |
| scikit-learn | PCA/StandardScaler |
| scipy | 插值/统计 |
| pytz | 时区处理 |
| protobuf | 数据序列化 |
| psutil | 内存监控 |

### 9.2 Protobuf定义

```protobuf
message PVSelectBest7 {
  int32 interval = 1;        // 时间间隔（分钟）
  repeated double values = 2; // 预测值列表（288个点）
}
```
