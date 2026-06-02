# 节假日负荷预测服务 API 文档

## 📖 目录

- [1. 服务概述](#1-服务概述)
- [2. 快速开始](#2-快速开始)
- [3. 接口说明](#3-接口说明)
  - [3.1 健康检查](#31-健康检查)
  - [3.2 节假日负荷预测](#32-节假日负荷预测)
- [4. 调用示例](#4-调用示例)
- [5. 错误码说明](#5-错误码说明)
- [6. 注意事项](#6-注意事项)
- [7. FAQ](#7-faq)

---

## 1. 服务概述

### 1.1 服务信息

| 项目 | 说明 |
|------|------|
| 服务名称 | 节假日负荷预测服务 |
| 服务版本 | v1.0.0 |
| 协议 | HTTP |
| 端口 | 3401 |
| 基础URL | `http://sigen-load-prediction-model-service.data-platform.svc.cluster.local:3401` |
| 响应格式 | JSON |

### 1.2 功能说明

本服务提供节假日场景下的电站负荷预测功能，采用DLinear深度学习模型，基于历史低负荷天数据进行预测。

**核心特性**：
- ✅ 异步任务处理，立即响应
- ✅ 多线程并发执行（最多5个任务并发）
- ✅ 支持GET和POST两种请求方式
- ✅ 预测结果直接写入Kafka消息队列
- ✅ 自动进行夏令时/冬令时校正

**预测策略**：
- 从过去60天历史数据中选择日均负荷最低的7天
- 使用这7天数据（2016个时间点）训练预测模型
- 输出未来2天的负荷预测（每天288个5分钟点）

---

## 2. 快速开始

### 2.1 环境要求

- Python 3.8+
- 网络可访问Kafka集群
- 网络可访问MySQL数据库

### 2.2 启动服务

**修改配置**（首次部署）：

编辑 `load_holiday_intra_schedule.py` 文件第702行：

```python
TEST_MODE = False  # 设置为False启动服务模式
```

**启动命令**：

```bash
# 进入项目目录
cd D:\sigen_ai\sigen-load-prediction\src

# 直接启动
python load_holiday_intra_on_call.py

# 后台启动（Linux）
nohup python load_holiday_intra_on_call.py > holiday_service.log 2>&1 &
```

**验证服务**：

```bash
curl http://localhost:3401/health
```

预期响应：
```json
{
  "status": "running",
  "db_connected": true,
  "model_loaded": true,
  "timestamp": "2025-12-19T10:30:00.123456"
}
```

---

## 3. 接口说明

### 3.1 健康检查

#### 基本信息

```
GET /health
```

#### 功能说明

检查服务运行状态、数据库连接状态和模型加载状态。

#### 请求参数

无需参数。

#### 响应示例

**成功响应（HTTP 200）**：

```json
{
  "status": "running",
  "db_connected": true,
  "model_loaded": true,
  "timestamp": "2025-12-19T10:30:00.123456"
}
```

**失败响应（HTTP 500）**：

```json
{
  "status": "error",
  "db_connected": false,
  "model_loaded": false,
  "error": "Database connection failed",
  "timestamp": "2025-12-19T10:30:00.123456"
}
```

---

### 3.2 节假日负荷预测

#### 基本信息

```
GET  /load_prediction_holiday
POST /load_prediction_holiday
```

#### 功能说明

提交节假日负荷预测任务，服务将在后台异步执行预测，并将结果写入Kafka消息队列。

#### 请求参数

| 参数名 | 类型 | 必填 | 说明 | 示例 |
|--------|------|------|------|------|
| station_id | string/int | 是 | 电站ID | `"12024061505406"` |
| holiday_mode | string/int | 是 | 节假日模式，**必须为1** | `"1"` |
| update_time | string/int | 是 | 更新时间，支持多种格式（见下方说明） | `"1766067235"` |

**update_time 支持的格式**：

| 格式类型 | 示例 | 说明 |
|---------|------|------|
| Unix时间戳（整数） | `1766067235` | 秒级时间戳 |
| 时间戳字符串 | `"1766067235"` | 秒级时间戳的字符串形式 |
| 日期时间字符串 | `"2025-12-18 10:50:00"` | 格式：`YYYY-MM-DD HH:MM:SS` |
| 日期字符串 | `"2025-12-18"` | 格式：`YYYY-MM-DD` |

**⚠️ 重要限制**：
- `holiday_mode` 必须为 `1`，否则任务将被拒绝
- `update_time` 的日期必须是**当前日期的前一天**，否则任务将被拒绝

#### 请求示例

**GET 请求**：

```bash
# 使用时间戳
curl "http://your-server-ip:3401/load_prediction_holiday?station_id=12024061505406&holiday_mode=1&update_time=1766067235"

# 使用日期字符串（需要URL编码）
curl "http://your-server-ip:3401/load_prediction_holiday?station_id=12024061505406&holiday_mode=1&update_time=2025-12-18"
```

**POST 请求**：

```bash
curl -X POST http://your-server-ip:3401/load_prediction_holiday \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": "12024061505406",
    "holiday_mode": "1",
    "update_time": "1766067235"
  }'
```

#### 响应示例

**成功响应（HTTP 202 Accepted）**：

```json
{
  "code": 0,
  "msg": "任务已接受，正在后台执行",
  "data": {
    "task_id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    "station_id": "12024061505406",
    "holiday_mode": "1",
    "update_time": "1766067235",
    "status": "accepted"
  }
}
```

**响应字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 状态码，0表示成功，非0表示失败 |
| msg | string | 响应消息 |
| data.task_id | string | 任务唯一标识符（UUID） |
| data.station_id | string | 电站ID |
| data.holiday_mode | string | 节假日模式 |
| data.update_time | string | 更新时间 |
| data.status | string | 任务状态，`accepted`表示已接受 |

**错误响应示例**：

```json
// 缺少必填参数
{
  "code": 400,
  "msg": "缺少必填参数 station_id",
  "data": null
}

// 数据库连接失败
{
  "code": 500,
  "msg": "数据库连接未初始化",
  "data": null
}

// 业务逻辑错误
{
  "code": 500,
  "msg": "节假日负荷预测执行失败: update_time日期与当前日期不一致",
  "data": null
}
```

#### 预测结果获取

**⚠️ 重要提示**：预测结果**不通过HTTP接口返回**，而是直接写入Kafka消息队列。

**Kafka消息格式**：

```
pred_date;station_id;predict_type;model;version;record_time;statistics_time;pred_values
```

**字段说明**：

| 字段 | 说明 | 示例 |
|------|------|------|
| pred_date | 预测日期 | `"2025-12-19"` |
| station_id | 电站ID | `12024061505406` |
| predict_type | 预测类型，2表示负荷 | `2` |
| model | 模型名称 | `"DeepHoliday-load-intraday"` |
| version | 模型版本 | `0.1` |
| record_time | 记录时间戳（秒） | `1766067890` |
| statistics_time | 时间戳列表（288个点） | `[1766016000, 1766016300, ...]` |
| pred_values | 预测值列表（288个点） | `[85.12345, 87.23456, ...]` |

**生成的模型消息**：

每次预测会生成**4条Kafka消息**：

| 预测天 | 模型名称 | 说明 |
|--------|---------|------|
| Day1 | `DeepHoliday-load-intraday` | 当日日内预测 |
| Day1 | `select_best_7` | 当日假期选优预测 |
| Day2 | `DeepHoliday-load-dayahead` | 次日日前预测 |
| Day2 | `select_best_7` | 次日假期选优预测 |

**消费Kafka示例**（Python）：

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'pv_prediction_topic_load',  # 从配置文件获取实际topic名称
    bootstrap_servers=['your-kafka-server:9092'],
    value_deserializer=lambda m: m.decode('utf-8')
)

for message in consumer:
    data = message.value.split(';')
    pred_date = data[0]
    station_id = data[1]
    predict_type = data[2]
    model = data[3]
    version = data[4]
    record_time = data[5]
    statistics_time = eval(data[6])  # 转换为列表
    pred_values = eval(data[7])      # 转换为列表

    print(f"收到预测结果: {pred_date} - {station_id} - {model}")
    print(f"预测点数: {len(pred_values)}")
```

---

## 4. 调用示例

### 4.1 Python

```python
import requests
import time

BASE_URL = "http://your-server-ip:3401"

def check_health():
    """检查服务健康状态"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Status: {response.json()}")
    return response.status_code == 200

def submit_prediction(station_id, holiday_mode=1, update_time=None):
    """提交预测任务"""
    if update_time is None:
        # 使用昨天的时间戳
        update_time = int(time.time()) - 86400

    # 方式1: GET请求
    url = f"{BASE_URL}/load_prediction_holiday"
    params = {
        "station_id": station_id,
        "holiday_mode": holiday_mode,
        "update_time": update_time
    }
    response = requests.get(url, params=params)

    # 方式2: POST请求
    # response = requests.post(url, json=params)

    result = response.json()

    if result['code'] == 0:
        task_id = result['data']['task_id']
        print(f"✅ 任务提交成功")
        print(f"   Task ID: {task_id}")
        print(f"   Station ID: {result['data']['station_id']}")
        return task_id
    else:
        print(f"❌ 任务提交失败: {result['msg']}")
        return None

# 使用示例
if __name__ == "__main__":
    # 1. 检查服务状态
    if not check_health():
        print("服务未就绪")
        exit(1)

    # 2. 提交预测任务
    task_id = submit_prediction(
        station_id=12024061505406,
        holiday_mode=1,
        update_time=1766067235
    )

    if task_id:
        print(f"请在Kafka中监听预测结果，task_id: {task_id}")
```

### 4.2 Java

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

public class HolidayPredictionClient {
    private static final String BASE_URL = "http://your-server-ip:3401";
    private static final HttpClient client = HttpClient.newHttpClient();
    private static final ObjectMapper mapper = new ObjectMapper();

    /**
     * 检查健康状态
     */
    public static boolean checkHealth() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(BASE_URL + "/health"))
            .GET()
            .build();

        HttpResponse<String> response = client.send(request,
            HttpResponse.BodyHandlers.ofString());

        System.out.println("Health: " + response.body());
        return response.statusCode() == 200;
    }

    /**
     * 提交预测任务（GET方式）
     */
    public static String submitPrediction(String stationId, int holidayMode, long updateTime)
            throws Exception {
        String url = String.format(
            "%s/load_prediction_holiday?station_id=%s&holiday_mode=%d&update_time=%d",
            BASE_URL, stationId, holidayMode, updateTime
        );

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .GET()
            .build();

        HttpResponse<String> response = client.send(request,
            HttpResponse.BodyHandlers.ofString());

        JsonNode json = mapper.readTree(response.body());

        if (json.get("code").asInt() == 0) {
            String taskId = json.get("data").get("task_id").asText();
            System.out.println("✅ 任务提交成功，Task ID: " + taskId);
            return taskId;
        } else {
            System.err.println("❌ 任务提交失败: " + json.get("msg").asText());
            return null;
        }
    }

    /**
     * 提交预测任务（POST方式）
     */
    public static String submitPredictionPost(String stationId, int holidayMode, long updateTime)
            throws Exception {
        String jsonBody = String.format(
            "{\"station_id\":\"%s\",\"holiday_mode\":\"%d\",\"update_time\":\"%d\"}",
            stationId, holidayMode, updateTime
        );

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(BASE_URL + "/load_prediction_holiday"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .build();

        HttpResponse<String> response = client.send(request,
            HttpResponse.BodyHandlers.ofString());

        JsonNode json = mapper.readTree(response.body());

        if (json.get("code").asInt() == 0) {
            String taskId = json.get("data").get("task_id").asText();
            System.out.println("✅ 任务提交成功，Task ID: " + taskId);
            return taskId;
        } else {
            System.err.println("❌ 任务提交失败: " + json.get("msg").asText());
            return null;
        }
    }

    public static void main(String[] args) {
        try {
            // 检查健康状态
            if (!checkHealth()) {
                System.err.println("服务未就绪");
                return;
            }

            // 提交预测任务
            String taskId = submitPrediction("12024061505406", 1, 1766067235L);

            if (taskId != null) {
                System.out.println("请在Kafka中监听预测结果");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 4.3 JavaScript/Node.js

```javascript
const axios = require('axios');

const BASE_URL = 'http://your-server-ip:3401';

/**
 * 检查健康状态
 */
async function checkHealth() {
    try {
        const response = await axios.get(`${BASE_URL}/health`);
        console.log('Health Status:', response.data);
        return response.status === 200;
    } catch (error) {
        console.error('Health check failed:', error.message);
        return false;
    }
}

/**
 * 提交预测任务
 */
async function submitPrediction(stationId, holidayMode = 1, updateTime = null) {
    try {
        if (!updateTime) {
            // 使用昨天的时间戳
            updateTime = Math.floor(Date.now() / 1000) - 86400;
        }

        // 方式1: GET请求
        const response = await axios.get(`${BASE_URL}/load_prediction_holiday`, {
            params: {
                station_id: stationId,
                holiday_mode: holidayMode,
                update_time: updateTime
            }
        });

        // 方式2: POST请求
        // const response = await axios.post(`${BASE_URL}/load_prediction_holiday`, {
        //     station_id: stationId,
        //     holiday_mode: holidayMode,
        //     update_time: updateTime
        // });

        const result = response.data;

        if (result.code === 0) {
            console.log('✅ 任务提交成功');
            console.log('   Task ID:', result.data.task_id);
            console.log('   Station ID:', result.data.station_id);
            return result.data.task_id;
        } else {
            console.error('❌ 任务提交失败:', result.msg);
            return null;
        }
    } catch (error) {
        console.error('请求失败:', error.message);
        if (error.response) {
            console.error('错误详情:', error.response.data);
        }
        return null;
    }
}

// 使用示例
(async () => {
    // 1. 检查服务状态
    const isHealthy = await checkHealth();
    if (!isHealthy) {
        console.error('服务未就绪');
        return;
    }

    // 2. 提交预测任务
    const taskId = await submitPrediction('12024061505406', 1, 1766067235);

    if (taskId) {
        console.log('请在Kafka中监听预测结果，task_id:', taskId);
    }
})();
```

### 4.4 cURL

```bash
#!/bin/bash

BASE_URL="http://your-server-ip:3401"

# 1. 检查健康状态
echo "检查服务健康状态..."
curl -s "${BASE_URL}/health" | jq .

# 2. 提交预测任务（GET方式）
echo -e "\n提交预测任务（GET方式）..."
curl -s "${BASE_URL}/load_prediction_holiday?station_id=12024061505406&holiday_mode=1&update_time=1766067235" | jq .

# 3. 提交预测任务（POST方式）
echo -e "\n提交预测任务（POST方式）..."
curl -s -X POST "${BASE_URL}/load_prediction_holiday" \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": "12024061505406",
    "holiday_mode": "1",
    "update_time": "1766067235"
  }' | jq .
```

---

## 5. 错误码说明

### 5.1 HTTP状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功（健康检查） |
| 202 | 任务已接受，正在后台执行 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

### 5.2 业务错误码

| code | msg | 说明 | 解决方案 |
|------|-----|------|---------|
| 0 | 任务已接受，正在后台执行 | 成功 | - |
| 400 | 缺少必填参数 station_id | 参数缺失 | 检查请求参数 |
| 400 | 缺少必填参数 holiday_mode | 参数缺失 | 检查请求参数 |
| 400 | 缺少必填参数 update_time | 参数缺失 | 检查请求参数 |
| 500 | 数据库连接未初始化 | 数据库连接失败 | 检查数据库配置 |
| 500 | 节假日负荷预测执行失败: xxx | 业务逻辑错误 | 查看错误详情 |

### 5.3 常见错误原因

#### holiday_mode 验证失败

```json
{
  "code": 500,
  "msg": "节假日负荷预测执行失败: holiday_mode为 0，不等于1，停止预测",
  "data": null
}
```

**原因**：`holiday_mode` 参数不等于1。
**解决**：确保 `holiday_mode` 值为 `1`。

#### update_time 日期不匹配

```json
{
  "code": 500,
  "msg": "节假日负荷预测执行失败: update_time日期 2025-12-17 与当前日期 2025-12-18 不一致，停止预测",
  "data": null
}
```

**原因**：`update_time` 的日期不是当前日期的前一天。
**解决**：确保 `update_time` 是昨天的日期。

#### 无法获取站点时区信息

```json
{
  "code": 500,
  "msg": "节假日负荷预测执行失败: 无法获取站点 12024061505406 的时区信息",
  "data": null
}
```

**原因**：数据库中找不到该站点的时区配置。
**解决**：检查站点ID是否正确，或联系管理员配置站点时区。

---

## 6. 注意事项

### 6.1 异步执行机制

⚠️ **重要提示**：本接口采用**异步任务处理**机制。

- 接口调用后**立即返回** HTTP 202 状态码和 `task_id`
- 实际预测任务在**后台线程池**中执行
- **不会阻塞**HTTP请求，可立即处理下一个请求
- 预测结果**不通过HTTP返回**，而是写入Kafka

### 6.2 并发限制

- 全局线程池最大工作线程数：**5个**
- 超出限制的任务将进入**队列等待**
- 建议控制调用频率，避免大量任务堆积

### 6.3 时间校验逻辑

```python
# 伪代码说明
current_date = 当前日期（根据站点时区）
update_date = update_time参数的日期部分

if update_date != current_date - 1天:
    拒绝任务
```

**示例**：
- 当前日期：2025-12-19
- 有效的 `update_time`：2025-12-18 的任意时刻
- 无效的 `update_time`：2025-12-17、2025-12-19、2025-12-20 等

### 6.4 时区处理

服务会自动处理以下时区问题：
- ✅ 夏令时（DST）自动校正
- ✅ 冬令时（WST）自动校正
- ✅ 时间戳转换到站点本地时区

### 6.5 预测结果说明

**预测天数**：2天（Day1和Day2）
- Day1：当天（`pred_date` 对应的日期）
- Day2：次日（`pred_date` + 1天）

**预测粒度**：5分钟
- 每天288个点（24小时 × 12个点/小时）

**模型输出**：
- 每个站点生成**4条**Kafka消息
- 2个模型名称：`DeepHoliday-load-*` 和 `select_best_7`
- 2个预测天：Day1和Day2

### 6.6 安全建议

- 🔒 建议在生产环境配置**API网关**进行访问控制
- 🔒 建议配置**IP白名单**限制访问来源
- 🔒 建议配置**请求频率限制**防止滥用
- 🔒 建议启用**HTTPS**加密传输（需配置反向代理）

---

## 7. FAQ

### Q1: 为什么接口返回202而不是200？

**A**: HTTP 202 (Accepted) 表示"请求已接受，但尚未处理完成"，这是异步API的标准响应状态码。任务在后台执行，不阻塞HTTP响应。

---

### Q2: 如何获取预测结果？

**A**: 预测结果不通过HTTP返回，而是直接写入Kafka消息队列。您需要：
1. 订阅Kafka Topic（topic名称从配置文件获取）
2. 消费消息并解析字段
3. 根据 `station_id` 和 `model` 筛选所需结果

---

### Q3: task_id 有什么用？

**A**: `task_id` 是任务的唯一标识符（UUID），可用于：
- 日志追踪和问题排查
- 任务状态监控（需要额外实现）
- 与Kafka消息关联（需要在消息中包含task_id，当前未实现）

---

### Q4: 为什么 holiday_mode 必须为1？

**A**: 这是业务逻辑要求。`holiday_mode=1` 表示启用节假日预测模式。其他值将被拒绝，以确保只在节假日场景使用该服务。

---

### Q5: update_time 为什么必须是昨天？

**A**: 这是业务规则：
- 系统需要使用"昨天"作为基准日期
- 预测"今天"和"明天"的负荷
- 确保有足够的历史数据进行预测

---

### Q6: 支持批量提交多个站点吗？

**A**: 当前版本不支持批量接口。如需预测多个站点，请：
1. 循环调用接口，每次传入一个站点ID
2. 控制调用频率，避免超过并发限制（5个）
3. 建议间隔100-200ms调用

---

### Q7: 如何判断任务是否执行成功？

**A**: 有以下几种方式：
1. **Kafka消息**：检查是否收到对应站点的预测结果
2. **日志文件**：查看服务日志，搜索 `task_id` 或 `station_id`
3. **数据库**（如果服务写入数据库）：查询预测记录表

---

### Q8: 任务提交后多久能得到结果？

**A**: 取决于多个因素：
- 站点数据量大小
- 当前并发任务数量
- 数据库查询性能
- 模型推理速度

**参考时间**：
- 单个站点：30秒 - 2分钟
- 多个站点（批次20个）：每批次1-3分钟

---

### Q9: 服务重启会丢失正在执行的任务吗？

**A**: 是的。当前版本不支持任务持久化，服务重启后：
- 线程池中的任务会丢失
- 需要重新提交预测请求

**建议**：
- 在服务重启前等待所有任务完成
- 或在重启后重新提交未完成的任务

---

### Q10: 如何监控服务运行状态？

**A**: 建议配置以下监控：
1. **健康检查**：定期调用 `/health` 接口
2. **日志监控**：监控日志中的ERROR和WARNING级别
3. **资源监控**：监控CPU、内存、网络使用率
4. **Kafka监控**：监控消息生产速率和延迟

---

## 📞 联系方式

如有问题或需要技术支持，请联系：

- **技术支持邮箱**: support@example.com
- **开发团队**: dev-team@example.com
- **紧急联系**: +86-xxx-xxxx-xxxx

---

**文档版本**: v1.0.0
**最后更新**: 2025-12-19
**维护者**: AI团队