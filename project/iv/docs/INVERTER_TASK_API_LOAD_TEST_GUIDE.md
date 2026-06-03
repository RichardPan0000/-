# Inverter Task API Load Test Guide

本文说明如何对逆变器级异步诊断 API 做并发压测，目标是评估真实算法在不同并发度下的耗时、CPU、内存和失败情况。

## 1. 测试目标

当前异步接口是：

```text
POST /diagnosis/tasks
GET  /diagnosis/tasks/{station_id}/{sn_code}/{task_id}
GET  /diagnosis/tasks/submissions/{submission_id}
GET  /health
```

生产默认去重 key 是：

```text
station_id:sn_code:task_id
```

因此同一个 `station_id + sn_code + task_id` 重复提交时，默认只会入队一次。压测同一份真实 IV 数据时，需要在 `src/config/diagnosis_config.yaml` 中临时打开重复提交模式：

```yaml
api_service:
  load_test:
    allow_duplicates: true
```

打开后，服务端会为每次提交生成一个临时 `submission_id`，实际任务 key 变成：

```text
station_id:sn_code:task_id:submission_id
```

这样可以用同一个真实 `task_id` 并发跑多次算法。

## 2. 启动 API 服务

建议压测时先关闭报告、Kafka 和 callback，避免把文件写入、外部系统和网络回调算进算法并发能力里。

在 `src/config/diagnosis_config.yaml` 中调整：

```yaml
api_service:
  server:
    host: "0.0.0.0"
    port: 3602
    reload: true

  task_manager:
    max_concurrency: 20
    queue_size: 100

  load_test:
    allow_duplicates: true

  runtime:
    output_dir: "output"
    use_mppt_level: null
    generate_csv: true
    generate_word_report: false
    write_kafka: false
    model_version: "v1.0.0"

  callback:
    enabled: false
    url: ""
    token: ""
    retries: 3
    timeout_seconds: 10
```

然后启动：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 src\start_api.py
```

如果要使用另一份配置文件：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 src\start_api.py --config src\config\diagnosis_config.yaml
```

关键参数：

| 配置项 | 含义 | 建议值 |
|---|---|---|
| `api_service.load_test.allow_duplicates` | 是否允许同一真实 task 重复提交 | 压测时 `true`，生产不要开 |
| `api_service.task_manager.max_concurrency` | 同时运行的诊断 worker 数 | 先测 4，再测 8/12/20 |
| `api_service.task_manager.queue_size` | 等待队列容量 | 100 |
| `api_service.runtime.generate_csv` | 是否生成诊断明细 CSV、处理后曲线 CSV 和 summary JSON | 如果要保留 CSV，设为 `true` |
| `api_service.runtime.generate_word_report` | 是否生成 Word 报告和 images | 压测算法时建议 `false` |
| `api_service.runtime.write_kafka` | 是否写 Kafka | 压测算法时 `false` |
| `api_service.callback.enabled` | 是否回调后端 | 压测算法时 `false` |

## 3. 用 Python 脚本提交并发请求

推荐优先使用仓库内的纯 Python 脚本：

```text
scripts/load_test_diagnosis_api.py
```

这个脚本会做三件事：

1. 并发提交 `POST /diagnosis/tasks`
2. 收集每次返回的 `submission_id`
3. 轮询 `GET /diagnosis/tasks/submissions/{submission_id}`，直到任务成功、失败或超时

默认会使用这组测试参数：

```text
BASE_URL        = http://127.0.0.1:3602
STATION_ID      = 82026011900040
SN_CODE         = 120A13AT0220
TASK_ID         = 1928
REQUESTS        = 20
SUBMIT_WORKERS  = 20
POLL_INTERVAL   = 2
TIMEOUT         = 1800
```

如果就用这组数据，直接运行：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 scripts\load_test_diagnosis_api.py
```

如果要覆盖默认值，例如仍然运行 20 个并发任务：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 scripts\load_test_diagnosis_api.py `
  --base-url http://127.0.0.1:3602 `
  --station-id 82026011900040 `
  --sn-code 120A13AT0220 `
  --task-id 1928 `
  --requests 20 `
  --submit-workers 20 `
  --poll-interval 2 `
  --timeout 1800
```

参数含义：

| 参数 | 含义 | 建议 |
|---|---|---|
| `--base-url` | API 服务地址 | 本机测试用 `http://127.0.0.1:3602` |
| `--station-id` | 站点 ID | 默认 `82026011900040` |
| `--sn-code` | 逆变器 SN | 默认 `120A13AT0220` |
| `--task-id` | IV 扫描任务 ID | 默认 `1928` |
| `--requests` | 总共提交多少个任务 | 例如 20 |
| `--submit-workers` | 同时发起多少个提交请求 | 通常和 `--requests` 一致 |
| `--poll-interval` | 轮询状态间隔，单位秒 | 2 秒 |
| `--timeout` | 等待所有任务完成的总超时，单位秒 | 真实算法建议 1800 秒起 |
| `--output-json` | 结果 JSON 输出路径 | 不传则写到 `output/load_test/` |
| `--resources-csv` | CPU、内存、队列采样 CSV 输出路径 | 不传则和 JSON 写到同一目录 |
| `--tasks-csv` | 单任务阶段耗时 CSV 输出路径 | 不传则和 JSON 写到同一目录 |

脚本结束时会打印提交成功数、失败数、最终任务状态计数、提交耗时、总耗时，并写出三类文件：

```text
output/load_test/diagnosis_api_load_test_时间戳.json
output/load_test/diagnosis_api_load_test_时间戳_resources.csv
output/load_test/diagnosis_api_load_test_时间戳_tasks.csv
```

JSON 里包含完整任务状态、`health_samples` 和 `task_timings_summary`。`resources.csv` 是扁平资源采样表，方便直接用 Excel 或 pandas 看压测过程中的资源变化。`tasks.csv` 是每个任务一行的阶段耗时表，用来定位慢在哪一步。

`resources.csv` 重点列：

| 列 | 含义 |
|---|---|
| `elapsed_seconds` | 从压测开始到本次采样的秒数 |
| `running_count` | 当前正在跑算法的任务数 |
| `queue_size` | 当前排队任务数 |
| `process_cpu_cores_used` | API 进程平均使用的逻辑核数 |
| `process_cpu_percent_of_machine` | API 进程占整机 CPU 百分比 |
| `process_memory_rss_mb` | API 进程实际内存占用 MB |
| `system_memory_percent` | 整机内存使用率 |
| `process_thread_count` | API 进程线程数 |

`tasks.csv` 重点列：

| 列 | 含义 |
|---|---|
| `queue_seconds` | 任务从创建到开始 running 的时间 |
| `run_seconds` | 任务从 running 到 finished 的时间 |
| `fetch_iv_data_seconds` | 从数据库获取 IV 数据耗时 |
| `parse_data_seconds` | 解析 IV 数据耗时 |
| `orchestrator_init_seconds` | 初始化诊断编排器耗时 |
| `diagnose_seconds` | 核心诊断算法耗时 |
| `rulemap_seconds` | rulemap 补充处理耗时 |
| `save_results_seconds` | 写 txt/json/csv 等结果文件耗时 |
| `build_records_seconds` | 构造入库/Kafka 记录耗时 |
| `kafka_seconds` | 写 Kafka 耗时，关闭时为 0 |
| `output_dir` | 本次任务实际输出目录 |

判断方式：

| 现象 | 优先怀疑 |
|---|---|
| `queue_seconds` 高 | API worker 并发不足或队列积压 |
| `fetch_iv_data_seconds` 高 | 数据库查询慢、连接池等待或 StarRocks 压力 |
| `diagnose_seconds` 高且 CPU 高 | 算法 CPU 计算瓶颈 |
| `diagnose_seconds` 高但 CPU 低 | 算法内部等待外部资源或共享锁 |
| `save_results_seconds` 高 | 文件写入慢、CSV 生成慢或输出目录冲突 |

如果输出 warning：

```text
accepted responses without submission_id
```

说明服务端没有打开：

```yaml
api_service:
  load_test:
    allow_duplicates: true
```

这种情况下，同一个真实 `task_id` 会被服务端按重复任务处理，不能用于测试“同一份数据重复并发跑算法”的能力。

压测重复提交模式下，每个 `submission_id` 会写到独立输出目录，避免 20 个并发任务同时覆盖同一批 CSV/JSON：

```text
output/load_test_runs/{station_id}_{sn_code}_{task_id}_{submission_id}/
```

目录内部仍然沿用原诊断产物命名，例如：

```text
diagnosis_82026011900040_1928.txt
diagnosis_82026011900040_1928.json
diagnosis_82026011900040_1928_report/
```

生产模式或非重复提交任务仍然写到原来的 `api_service.runtime.output_dir` 下。

### PyCharm 运行方式

在 PyCharm 新建一个 Python Run Configuration：

| 配置项 | 值 |
|---|---|
| Script path | `D:\sigen-project2\sigen-iv-diagnosis\scripts\load_test_diagnosis_api.py` |
| Working directory | `D:\sigen-project2\sigen-iv-diagnosis` |
| Python interpreter | `D:\software\miniforge\install\envs\python310\python.exe` |
| Parameters | 可以留空 |

```text
--requests 20 --submit-workers 20
```

这样不需要在 PyCharm 里配置环境变量。如果使用默认值，`Parameters` 可以完全留空；如果只想改并发数，填 `--requests 50 --submit-workers 50` 即可。

## 4. 可选：用 k6 只测并发提交

仓库内也保留了 k6 脚本：

```text
scripts/k6_submit_diagnosis.js
```

直接运行：

```powershell
k6 run --vus 20 --iterations 20 scripts\k6_submit_diagnosis.js
```

k6 脚本只测“并发提交”。因为接口是异步的，`POST /diagnosis/tasks` 返回 `202` 只代表任务入队成功，不代表算法已经完成。如果要统计算法真正完成耗时，优先用上面的 Python 脚本。

## 5. 观察任务并发

压测提交后，用 `/health` 查看服务内部状态：

```powershell
Invoke-RestMethod "http://127.0.0.1:3602/health"
```

重点看：

```json
{
  "resources": {
    "available": true,
    "process_id": 12345,
    "logical_cpu_count": 16,
    "physical_cpu_count": 8,
    "process_cpu_cores_used": 3.75,
    "process_cpu_percent_of_machine": 23.44,
    "process_memory_rss_mb": 1850.5,
    "system_memory_percent": 62.1,
    "process_thread_count": 42
  },
  "task_manager": {
    "worker_count": 20,
    "queue_size": 0,
    "queue_capacity": 100,
    "running_count": 20,
    "status_counts": {
      "running": 20
    },
    "load_test_allow_duplicates": true
  }
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `resources.process_cpu_cores_used` | 两次 `/health` 调用之间，API 进程平均用了多少个逻辑核。第一次调用通常是 `null`，第二次开始有效 |
| `resources.process_cpu_percent_of_machine` | API 进程占整台机器 CPU 的百分比 |
| `resources.process_memory_rss_mb` | API 进程当前实际占用内存，单位 MB |
| `resources.system_memory_percent` | 整台机器内存使用率 |
| `resources.process_thread_count` | 当前 API 进程线程数 |
| `worker_count` | 当前服务启动的 worker 数，也就是并发上限 |
| `running_count` | 正在执行算法的任务数 |
| `queue_size` | 还在等待执行的任务数 |
| `status_counts` | 各状态任务数量 |
| `load_test_allow_duplicates` | 是否处于重复提交压测模式 |

如果 `api_service.task_manager.max_concurrency=20`，并且提交了 20 个任务，理想情况下 `running_count` 会接近或达到 20。  
如果一直只有 1 或 2，说明瓶颈可能在启动参数、worker 初始化、数据库连接或算法内部共享资源。

注意：`process_cpu_cores_used` 是相邻两次 `/health` 之间的平均值。例如返回 `3.75`，表示这段时间平均消耗约 3.75 个逻辑核；如果机器有 16 个逻辑核，对应整机 CPU 约 `23.44%`。

## 6. 查询单个 submission

压测模式下，`POST /diagnosis/tasks` 返回里会包含 `submission_id`：

```json
{
  "success": true,
  "message": "task queued",
  "data": {
    "station_id": 82026011900040,
    "sn_code": "120A13AT0220",
    "task_id": 1928,
    "submission_id": "f5a7...",
    "status": "queued",
    "task_key": "82026011900040:120A13AT0220:1928:f5a7..."
  }
}
```

可以按 `submission_id` 查询：

```powershell
Invoke-RestMethod "http://127.0.0.1:3602/diagnosis/tasks/submissions/f5a7..."
```

状态可能是：

| 状态 | 含义 |
|---|---|
| `queued` | 已入队，等待 worker |
| `running` | 正在运行算法 |
| `success` | 算法完成 |
| `failed` | 算法失败 |
| `callback_failed` | 算法完成或失败后，回调后端失败 |

## 7. 记录资源和耗时

建议每个并发梯度都记录一次：

| 并发 | 总任务数 | 总完成时间 | 成功数 | 失败数 | CPU 峰值 | 内存峰值 | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 20 |  |  |  |  |  |  |
| 8 | 20 |  |  |  |  |  |  |
| 12 | 20 |  |  |  |  |  |  |
| 20 | 20 |  |  |  |  |  |  |

Windows 上可以先用任务管理器观察 Python 进程，也可以用 PowerShell 采样：

```powershell
Get-Process python | Select-Object Id,CPU,WorkingSet64,StartTime
```

压测过程建议每隔几秒查一次：

```powershell
Invoke-RestMethod "http://127.0.0.1:3602/health"
```

判断并发能力时重点看：

- `running_count` 是否能达到设置的并发上限
- 总完成时间是否随着并发增加而下降
- CPU 是否打满
- 内存是否持续上涨
- 是否出现数据库连接、画像查询、ECMWF 查询或算法异常

## 8. 队列满测试

可以把并发和队列调小，验证保护逻辑：

```yaml
api_service:
  task_manager:
    max_concurrency: 1
    queue_size: 2
  load_test:
    allow_duplicates: true
```

然后提交超过 3 个任务。超过容量时，接口应返回：

```json
{
  "detail": "Diagnosis task queue is full"
}
```

HTTP 状态码是 `429`。

## 9. 压测结束后恢复

压测结束后必须关闭重复提交模式：

```yaml
api_service:
  load_test:
    allow_duplicates: false
```

修改配置后重新启动 API 服务。

生产环境不要打开：

```text
api_service.load_test.allow_duplicates: true
```

否则同一个真实 `task_id` 可以被重复执行，会造成重复计算、重复结果和额外资源消耗。

## 10. 推荐压测顺序

1. `api_service.task_manager.max_concurrency=4`，提交 20 个任务。
2. 记录总耗时、CPU、内存、成功/失败数。
3. 改成 `8`，重复测试。
4. 改成 `12`，重复测试。
5. 改成 `20`，重复测试。
6. 如果 `running_count` 到不了上限，先看队列、数据库连接、日志异常。
7. 如果 CPU 已经打满，再提高并发通常不会提升吞吐。
