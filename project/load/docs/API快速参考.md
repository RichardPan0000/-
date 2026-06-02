# 节假日负荷预测服务 - 快速参考

## 🚀 5分钟快速接入

### 服务地址

```
http://sigen-load-prediction-model-service.data-platform.svc.cluster.local:3401
```

---

## 📌 核心接口

### 1. 健康检查

```bash
GET /health
```

**响应**：
```json
{
  "status": "running",
  "db_connected": true,
  "model_loaded": true
}
```

---

### 2. 提交预测任务

```bash
GET /load_prediction_holiday?station_id=12024061505406&holiday_mode=1&update_time=1766067235
```

**必填参数**：
- `station_id`: 电站ID
- `holiday_mode`: **必须为1**
- `update_time`: 昨天的时间戳或日期

**响应**：
```json
{
  "code": 0,
  "msg": "任务已接受，正在后台执行",
  "data": {
    "task_id": "xxx...",
    "status": "accepted"
  }
}
```

---

## 💡 调用示例

### Python

```python
import requests

response = requests.get(
    "http://sigen-load-prediction-model-service.data-platform.svc.cluster.local:3401/load_prediction_holiday",
    params={
        "station_id": "12024061505406",
        "holiday_mode": "1",
        "update_time": "1766067235"
    }
)

result = response.json()
if result['code'] == 0:
    print(f"成功！task_id: {result['data']['task_id']}")
```

### cURL

```bash
curl "http://sigen-load-prediction-model-service.data-platform.svc.cluster.local:3401/load_prediction_holiday?station_id=12024061505406&holiday_mode=1&update_time=1766067235"
```

### Java

```java
String url = "http://sigen-load-prediction-model-service.data-platform.svc.cluster.local:3401/load_prediction_holiday" +
    "?station_id=12024061505406&holiday_mode=1&update_time=1766067235";

HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create(url))
    .GET()
    .build();

HttpResponse<String> response = client.send(request,
    HttpResponse.BodyHandlers.ofString());
```

---

## ⚠️ 重要注意事项

### ❌ 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `holiday_mode为0，不等于1` | holiday_mode不是1 | 设置为1 |
| `update_time日期不一致` | update_time不是昨天 | 使用昨天的日期 |
| `缺少必填参数 xxx` | 参数缺失 | 检查所有必填参数 |

### ✅ 关键要点

1. **异步执行**：接口立即返回，预测在后台执行
2. **结果获取**：预测结果写入Kafka，**不在HTTP响应中**
3. **并发限制**：最多5个任务同时执行，超出则排队
4. **时间校验**：update_time 必须是昨天的日期
5. **holiday_mode**：必须为1，否则拒绝

---

## 📊 Kafka消息格式

**Topic**: 从配置文件读取 `PV_RESULT_TOPIC`

**消息格式**：
```
pred_date;station_id;predict_type;model;version;record_time;statistics_time;pred_values
```

**示例**：
```
2025-12-19;12024061505406;2;DeepHoliday-load-intraday;0.1;1766067890;[1766016000,1766016300,...];[85.12,87.23,...]
```

**每次预测生成4条消息**：
- Day1: `DeepHoliday-load-intraday`
- Day1: `select_best_7`
- Day2: `DeepHoliday-load-dayahead`
- Day2: `select_best_7`

---

## 🔧 故障排查

### 检查服务状态
```bash
curl http://localhost:3401/health
```

### 查看日志
```bash
tail -f holiday_service.log | grep ERROR
```

### 测试连通性
```bash
telnet your-server-ip 3401
```

---

## 📞 快速联系

- 详细文档：查看 `API文档-节假日负荷预测服务.md`
- 技术支持：support@example.com
- 紧急问题：查看服务日志 `/path/to/logs/`

---

**提示**：这是快速参考文档，完整说明请参阅详细API文档。