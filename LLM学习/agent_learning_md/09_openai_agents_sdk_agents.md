# OpenAI Agents SDK：Agent 抽象

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://openai.github.io/openai-agents-python/agents/

## 一句话理解

OpenAI Agents SDK 把 Agent 定义成：**LLM + instructions + tools + 可选运行行为**，包括 handoffs、guardrails、structured outputs 等。

## 核心组成

一个 Agent 通常包含：

- name：名称。
- instructions：系统指令。
- model：模型。
- tools：工具。
- handoffs：可交接的其他 Agent。
- guardrails：输入输出保护。
- output_type：结构化输出类型。
- hooks：生命周期钩子。
- context：运行时上下文。

## 思想价值

这套抽象很适合你理解生产 Agent：

```text
Agent 不只是 prompt
Agent 是一个具备工具、边界、上下文和运行行为的对象
```

## Manager 与 Handoff

文档中提到两个典型多 Agent 模式：

### Manager / Agents as Tools

中心 Agent 调用专业 Agent 工具。

优点：

- 控制集中。
- 容易审计。
- 适合生产系统。

### Handoffs

一个 Agent 把控制权交给另一个 Agent。

优点：

- 专业 Agent 可以持续和用户交互。
- 更自然地支持多轮场景。

## 学习重点

1. Agent 的属性设计。
2. Tool 和 MCP server 的接入。
3. Structured output。
4. Guardrails。
5. Hooks 和 tracing。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
