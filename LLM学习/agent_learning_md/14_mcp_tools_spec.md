# MCP Tools Spec：工具规范

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://modelcontextprotocol.io/specification/2025-06-18/server/tools

## 一句话理解

MCP Tools 规范定义了 Server 如何向 AI 应用暴露可调用工具，以及客户端如何发现和调用这些工具。

## 工具为什么需要规范

没有规范时，每个工具接口都可能长得不一样：

- 参数格式不同。
- 返回格式不同。
- 错误表达不同。
- 描述不一致。
- 权限不可控。

MCP Tools 规范让工具变成可以被发现、理解、调用和验证的标准对象。

## 工具元数据

一个工具通常应该包含：

- name：工具唯一名称。
- title：人类可读名称。
- description：工具用途说明。
- inputSchema：输入参数 JSON Schema。
- annotations：可选描述，比如是否只读、是否危险等。

## 调用流程

```text
1. Client 请求 tools/list
2. Server 返回工具列表
3. LLM 根据工具描述选择工具
4. Client 发起 tools/call
5. Server 执行工具
6. Client 把结果返回给 LLM
```

## 工具设计建议

### 1. 名称要明确

不好：

```text
query
```

好：

```text
query_iv_curve_by_station_and_date
```

### 2. 输入 schema 要严格

参数越明确，模型越不容易犯错。

### 3. 危险工具要标记

例如：

- 删除数据。
- 写数据库。
- 发邮件。
- 修改生产配置。

这些都应该配合 human-in-the-loop。

### 4. 输出要结构化

不要只返回自然语言，最好返回：

```json
{
  "success": true,
  "data": {},
  "warnings": [],
  "error": null
}
```

## 对你的实践建议

你可以先做一个简单 MCP Server：

- `search_docs(query)`
- `get_iv_curve(station_id, date)`
- `get_bms_alarm(sn, start_time, end_time)`
- `save_report(title, markdown)`

再让不同 Agent 都通过 MCP 使用这些能力。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
