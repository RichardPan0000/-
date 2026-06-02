# MCP Architecture 与 Tools Spec 精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


MCP Architecture：  
https://modelcontextprotocol.io/docs/learn/architecture

MCP Intro：  
https://modelcontextprotocol.io/docs/getting-started/intro

MCP Tools Spec：  
https://modelcontextprotocol.io/specification/2025-06-18/server/tools

## 1. MCP 解决什么问题？

MCP 的核心问题是：如何让 AI 应用以统一方式连接外部工具、数据源和工作流？

过去每个 AI 应用都要单独接文件系统、数据库、Git、企业知识库、内部 API、浏览器、业务系统。MCP 希望提供统一协议。

## 2. MCP 架构角色

| 角色 | 说明 |
|---|---|
| MCP Host | 宿主应用，例如 IDE、Chat 客户端、Agent 平台 |
| MCP Client | Host 里面负责连接 MCP Server 的组件 |
| MCP Server | 对外暴露能力的服务 |

## 3. MCP Server 可以暴露什么？

### Tools

可执行动作，例如 search_docs、read_file、query_database、create_ticket、run_test。

### Resources

可读取资源，例如文件内容、数据库记录、项目配置、知识库文档。

### Prompts

可复用提示模板，例如代码审查模板、专利交底书模板、故障诊断模板。

## 4. Data Layer 与 Transport Layer

Data Layer 规定协议语义，例如初始化、能力协商、tools/list、tools/call、resources/read、prompts/get。

Transport Layer 规定通信方式，例如 stdio 和 Streamable HTTP。

## 5. MCP Tools 的基本结构

```json
{
  "name": "search_docs",
  "title": "Search Documents",
  "description": "Search internal documents by query.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"}
    },
    "required": ["query"]
  }
}
```

## 6. 为什么 MCP 对定制 Agent 很重要？

如果你做定制开发，真正的价值不只是写 prompt，而是把企业能力接入 Agent。

例如你的场景：

```text
PV IV 数据库
BMS 告警系统
设备台账
气象数据
代码仓库
标准规范文档
专利模板库
```

这些都可以封装成 MCP Server。

## 7. 业务 MCP Server 设计示例

### PV 诊断 MCP Server

Tools：

```text
get_station_info
get_iv_curve
get_weather_at_scan_time
get_string_history
search_fault_knowledge
generate_diagnosis_report
```

### BMS MCP Server

Tools：

```text
query_cell_voltage
query_bms_alarms
get_soc_history
get_temperature_history
detect_voltage_inconsistency
```

### Code MCP Server

Tools：

```text
search_repo
read_file
apply_patch
run_tests
create_pull_request
```

## 8. 工具安全

| 工具 | 风险 | 控制 |
|---|---|---|
| read_file | 低 | 自动 |
| search_docs | 低 | 自动 |
| query_database | 中 | 权限控制 |
| write_file | 中 | 审批或沙箱 |
| send_email | 高 | 人审 |
| delete_data | 极高 | 禁止或强审批 |
| deploy_production | 极高 | 强审批 |

## 9. MCP 和 LangGraph 的关系

```text
LangGraph Node → MCP Tool → Business System
```

MCP 负责把外部能力标准化暴露出来，LangGraph 负责在流程中决定什么时候调用这些能力。

## 10. 练习

设计一个你的 MCP Server，列出 Tool、用途、输入、输出、风险、是否需要审批。
