# 诊断并发变慢分析

## 1. 背景

本文用于总结当前异步逆变器诊断 API 在并发执行时变慢的根因分析。

分析基于最近一次 20 并发压测结果以及当前仓库中的实现代码。本文会刻意区分两类结论：

- `Confirmed`：可以直接从代码或压测产物中得到支持的事实
- `Inferred`：有较强证据支持，但还不能当作唯一最终结论的推断

本文档只做分析，不在这一轮直接推动代码改造。

## 2. 证据来源

主要运行产物：

- `scripts/output/load_test/diagnosis_api_load_test_20260424_204708.json`
- `scripts/output/load_test/diagnosis_api_load_test_20260424_204708_tasks.csv`
- `scripts/output/load_test/diagnosis_api_load_test_20260424_204708_resources.csv`

重点审查的代码路径：

- `src/diagnosis/flows/healthy_curve_flow.py`
- `src/data_utils/string_portrait_manager.py`
- `src/data_utils/ecmwf_temperature_reader.py`
- `src/database.py`
- `src/diagnosis/runtime_task_runner.py`

## 3. 主要现象

从压测汇总结果看：

- `queue_seconds` 基本可以忽略，平均约 `0.0077s`
- `fetch_iv_data_seconds` 是明显耗时项，平均约 `5.17s`
- `diagnose_seconds` 是最大耗时项，平均约 `21.84s`
- `save_results_seconds` 不是主因，但也不小，平均约 `3.25s`
- `running_count` 可以达到 `20`
- 整个压测过程中，进程级 CPU 基本长期在 `0.0` 到 `0.05` 个逻辑核之间

从单任务 profile 看：

- 慢任务的热点 stage 主要集中在：
  - `flow.portrait_cache.prefetch`
  - `flow.group.external_lookup.total`
  - `flow.group.fetch_ecmwf_temperature`
  - `flow.group.collect_group_statistics`
  - `flow.group.voc_slot_calibration`
  - 多次重复出现的 `flow.string.total` / `flow.string.rule_predictor_predict`
- 快任务的 stage 结构基本一致，只是外部查找和 group 级别阶段明显更短

这说明：变慢不是因为 API 队列排队，而是任务进入 `running` 之后，在诊断内部发生了显著放大。

## 4. 代码映射

### 4.1 IV 数据获取

`src/diagnosis/runtime_task_runner.py` 在诊断开始前单独记录了 `fetch_iv_data_seconds`。

`src/database.py:get_iv_data_via_task_id()` 每个任务会执行两次 StarRocks 查询：

- 一次查 IV 原始数据
- 一次查 `information_schema.COLUMNS`

这和压测里 `fetch_iv_data_seconds` 较高是对应得上的，因此这一段必须被算进整体变慢原因里。

### 4.2 画像预加载

`src/diagnosis/flows/healthy_curve_flow.py` 在每个任务开始时会做：

- `flow.portrait_cache.prefetch`
- `self.portrait_manager.load_portrait_cache_by_inverter(...)`

`src/data_utils/string_portrait_manager.py` 显示：

- 每个 `HealthyCurveDiagnosisFlow` 都会创建自己的 `StringPortraitManager`
- 每个任务都会把同一台逆变器的画像 preload 到任务私有内存中
- 任务结束后又会清空这份 cache

这个点很重要：当前并不存在“跨任务共享的内存画像缓存”。因此问题不是共享缓存锁竞争，而是**多个并发任务重复去加载同一批下游画像数据**。

### 4.3 外部查找路径

在 `src/diagnosis/flows/healthy_curve_flow.py` 中，`flow.group.external_lookup.total` 当前包含：

- 参考画像解析
- 参考 `n_components` 解析
- `db_conn_global.get_station_info(station_id)`
- UTC 时间转换
- `fetch_ecmwf_temperature(...)`

这个 stage 在慢任务中确定很热，但它是一个聚合 stage，本身不能直接证明到底是哪一个子步骤占了最大头。

### 4.4 ECMWF 查询路径

`src/data_utils/ecmwf_temperature_reader.py` 并没有复用 `src/database.py` 中的连接池。

它每次调用都会：

- 新建一个 MySQL 连接去查站点位置
- 新建一个 StarRocks 连接去查 ECMWF 天气数据

并且这里还存在一次重复站点查询，因为 `healthy_curve_flow.py` 在调用 ECMWF reader 之前已经先调用过一次 `db_conn_global.get_station_info(station_id)`。

### 4.5 group 级重复计算

`src/diagnosis/flows/healthy_curve_flow.py:_collect_group_statistics()` 会串行遍历 eligible string，并重复执行：

- `resolve_n_components`
- 预处理 / 插值 / 平滑
- 归一化网格特征提取
- `FivePointSolver.solve_from_iv_curve(...)`

`flow.group.voc_slot_calibration` 又会做一轮 group 级别的逐串处理。

最后的逐串诊断路径里，还会再次做预处理和预测。

这说明并发变慢并不只是“外部等待”，内部也存在非常真实的重复串行计算。

### 4.6 结果落盘

`src/diagnosis/runtime_task_runner.py` 对 `save_results_seconds` 做了单独计时。

`src/run_diagnosis_with_config.py:save_results()` 每个任务会写出多类文件：

- 文本结果
- JSON 结果
- report 目录
- diagnosis details CSV
- processed current IV CSV
- processed healthy IV CSV
- 可选的 rulemap summary CSV
- summary JSON

这部分持久化 I/O 不是当前主瓶颈，但它是**已确认的次级贡献项**，不能在完整分析里完全忽略。

## 5. 已确认结论

- 对这次测试而言，API queue 不是主瓶颈。
- 任务确实并发执行，`running_count` 能达到 `20`。
- 进程级 CPU 在整个压测过程中并没有被打满。
- `diagnose` 是主 wall-time 耗时阶段。
- `fetch_iv_data_seconds` 是实质性耗时来源之一，不能从根因分析中排除。
- 每个任务都会重复 preload 同一台逆变器的画像，并在任务结束时清空私有 cache。
- group 外部查找路径里存在重复远程工作，尤其是站点信息查询在 ECMWF 之前被重复执行。
- ECMWF reader 每次都会新建 DB 连接，而不是复用 `database.py` 的池。
- `collect_group_statistics`、`voc_slot_calibration` 和最终逐串诊断路径都存在重复的逐串预处理 / 特征提取工作。

## 6. 推断结论

- 并发变慢中，有相当一部分很可能来自并发下重复的下游 DB / 外部查询工作，尤其是画像 preload 和 ECMWF 相关查询。
- 另一部分很可能来自 group 统计、校准以及重复逐串预处理造成的串行本地计算放大。
- 当前证据支持“复合型根因”，而不支持“某一个单独子阶段就是唯一已确认瓶颈”。

## 7. 不应过度下结论的内容

以下说法当前都不应写成 confirmed：

- `database.py` 连接池就是已确认根因
- `flow.group.external_lookup.total` 主要就是画像 DB 等待
- 低 CPU 可以直接证明整个诊断路径都是 I/O bound
- `collect_group_statistics` 只是等待，并没有真实重复计算

这些说法都超过了当前证据能支撑的范围。

## 8. 次要问题：profile 输出路径不一致

当前还有一个与性能主因无关、但会影响排查体验的问题：

- 正常诊断输出通过 `run_diagnosis_with_config.py:_resolve_output_dir()`，相对路径会按项目根目录解析
- profiler 输出在 `src/diagnosis/runtime_task_runner.py` 中通过 `profiler.write_outputs(Path(output_dir), ...)` 写出

因此 profile 文件可能实际落在 `src/output/...`，但任务状态里展示的是 `output/...` 相对路径。

这是一个产物路径基准不一致问题，不是本次并发变慢的根因。

## 9. 建议的后续方向

以下方向按当前置信度排序。

### 9.1 先补更细的 DB / 连接等待埋点

这是当前最稳妥、置信度最高的下一步动作。

现有 stage profiling 已经证明外部查找 / 数据获取相关路径很热，但还没有拆开：

- 取连接慢
- SQL 执行慢
- 查询本身重复
- 还是聚合 stage 内部的本地计算慢

因此下一轮更细埋点应该先把这些拆开，再决定是否把某一个外部子路径当作“主优化入口”。

### 9.2 减少重复外部查找

这是当前置信度最高的优化目标。

优先考虑：

- 对 `(station_id, sn_code)` 级别的逆变器画像 preload 做跨任务缓存或 singleflight
- 对 `station_id` 级站点信息做缓存
- 对 `(station_id, rounded_hour)` 级 ECMWF 结果做缓存
- 把 ECMWF 相关 DB 访问统一收敛到共享查询层，避免每次直接创建新连接

### 9.3 压缩 group 级重复计算

这条方向有价值，但当前置信度低于外部查找路径。

可考虑：

- 复用已解析过的 `n_components`
- 在 group statistics、voc slot calibration 和最终逐串预测之间复用预处理结果
- 对不变输入避免重复特征提取

## 10. 后续讨论推荐表述

当前最安全的总结句是：

`Confirmed：并发变慢主要表现为任务进入 running 之后的低 CPU wall-time 放大，最热的测量 stage 出现在 diagnose 内部，同时结果落盘 I/O 也有次级贡献。`

`Inferred：主要机制更像是重复外部 DB/查询工作与重复串行本地计算叠加，而不是单一孤立瓶颈。`

## 11. 审核说明

本文已经吸收了两轮独立 agent 审核意见：

- 第一轮：避免把 external lookup 怀疑写成单一 confirmed 根因
- 第二轮：明确把重复串行 group 计算和 `fetch_iv_data_seconds` 纳入整体根因分析
- 第三轮文档审校：把“先优化”调整为“先补埋点，再优化”，补入 `save_results_seconds` 的次级贡献，并收紧结尾总结措辞

后续若进入实现阶段，建议继续保留这种区分：

- 哪些是已经被代码和压测直接证明的
- 哪些只是高概率推断，仍需下一轮埋点确认
