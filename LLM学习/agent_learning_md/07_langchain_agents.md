# LangChain Agents：单 Agent 抽象

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://docs.langchain.com/oss/python/langchain/agents

## 一句话理解

LangChain Agents 适合快速构建“一个 LLM + 工具 + 循环”的 Agent。

## Agent 的基本组成

一个典型 Agent 包含：

- Model：语言模型。
- Instructions：系统指令。
- Tools：可调用工具。
- Middleware：中间件和控制逻辑。
- Structured Output：结构化输出。
- Memory：短期或长期记忆。

## 什么时候用 LangChain Agent

适合：

- 快速做单 Agent。
- 做工具调用原型。
- 做 RAG + Tool Calling。
- 做结构化输出。
- 做一个不太复杂的业务助手。

不适合：

- 复杂多阶段审批流。
- 长时间运行任务。
- 复杂多 Agent 状态机。
- 需要细粒度恢复的任务。

这些更适合 LangGraph。

## 学习重点

### 1. Tool Calling

Agent 不是只聊天，而是根据任务选择工具。

### 2. Structured Output

生产里不要只要自然语言，很多时候要 JSON / Pydantic 格式。

### 3. Middleware

中间件适合加：

- 日志。
- 权限。
- 人审。
- 输出过滤。
- 成本统计。

## 实战理解

可以把 LangChain Agent 当作 LangGraph 中的一个 node：

```text
LangGraph Node = 一个 LangChain Agent 或普通函数
```

这样组合非常常见。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
