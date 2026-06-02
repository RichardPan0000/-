# Anthropic《Writing Effective Tools for Agents》精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://www.anthropic.com/engineering/writing-tools-for-agents

## 1. 核心问题

这篇文章回答的是：如何设计让 Agent 真正会用、用得对、用得稳定的工具？

很多 Agent 效果差，不是模型不会推理，而是工具设计得太差。

## 2. 工具不是 API 的简单包装

底层 API 往往面向人类开发者，不适合直接暴露给模型。LLM 使用工具时，需要更明确的工具名、使用场景、不适用场景、参数含义、输入限制、输出格式、错误说明和示例。

## 3. Agent-Computer Interface

原文强调 ACI：Agent-Computer Interface。给 Agent 用的工具接口，也应该像给人用的人机界面一样认真设计。

## 4. 好工具的特征

### 名称具体

不好：

```text
query()
process()
run()
```

好：

```text
search_project_docs(query)
read_file_by_absolute_path(path)
get_iv_curve_by_station_and_date(station_id, date)
```

### 描述清晰

工具描述应该回答什么时候用、什么时候不要用、输入是什么、输出是什么、失败怎么办。

### 参数不容易误用

例如文件路径最好要求绝对路径，减少模型误解当前目录导致的问题。

### 返回结构稳定

建议返回结构：

```json
{
  "success": true,
  "result": {},
  "warnings": [],
  "error": null
}
```

### 工具边界不重叠

如果两个工具都能“查询信息”，模型会困惑。应该让工具职责明确。

## 5. 防呆设计

工具应该让错误更难发生。例如修改文件工具可以要求 old_snippet 精确匹配，而不是直接覆盖整个文件。

## 6. 工具权限与安全

| 类型 | 例子 | 控制 |
|---|---|---|
| 只读工具 | 查询文档、读数据库 | 可自动调用 |
| 低风险写工具 | 保存草稿 | 可自动或轻量确认 |
| 高风险写工具 | 发邮件、建工单、改配置 | 人工审批 |
| 危险工具 | 删除数据、生产发布 | 强审批或禁止 |

## 7. 业务工具设计示例

### IV 曲线查询工具

```json
{
  "name": "get_iv_curve",
  "description": "根据电站、设备、日期查询 IV 扫描曲线。只用于读取历史诊断数据，不做诊断结论。",
  "input_schema": {
    "station_id": "string",
    "sn_code": "string",
    "date": "YYYY-MM-DD"
  }
}
```

### BMS 告警查询工具

```json
{
  "name": "query_bms_alarms",
  "description": "查询指定设备在时间范围内的 BMS 告警记录，用于辅助故障分析。",
  "input_schema": {
    "device_sn": "string",
    "start_time": "ISO datetime",
    "end_time": "ISO datetime"
  }
}
```

## 8. 练习

设计 5 个你自己业务中的 Agent 工具，并标注风险等级、是否需要人审。
