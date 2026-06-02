# MCP Architecture：架构与核心概念

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://modelcontextprotocol.io/docs/learn/architecture

## 一句话理解

MCP 是一个 client-server 架构，负责在 AI 应用和外部上下文/工具提供方之间传递能力。

## 参与者

| 角色 | 说明 |
|---|---|
| MCP Host | AI 应用，例如 IDE、桌面客户端、Agent 平台 |
| MCP Client | Host 内部维护连接的组件 |
| MCP Server | 对外提供工具、资源、prompt 的服务 |

一个 Host 可以连接多个 MCP Server。

## 两层架构

### Data Layer

负责协议语义，基于 JSON-RPC 2.0，包括：

- 初始化。
- 能力协商。
- 工具列表。
- 资源读取。
- prompt 获取。
- 工具调用。
- 通知。

### Transport Layer

负责通信通道，包括：

- stdio：本地进程通信。
- Streamable HTTP：远程通信。

## 核心 primitives

| Primitive | 作用 |
|---|---|
| Tools | 可执行动作 |
| Resources | 提供上下文数据 |
| Prompts | 提供提示模板 |

## Tool Discovery

客户端不是预先知道所有工具，而是通过 `tools/list` 获取工具列表，再通过 `tools/call` 调用工具。

这意味着工具能力可以动态发现、动态更新。

## 对定制开发的启发

真正可复用的企业 Agent 体系，不应该把业务能力写死在某个 Agent 里，而应该封装为 MCP Server：

```text
业务系统 → MCP Server → Agent / IDE / Coding Tool
```

这样你的工具生态才能复用和扩展。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
