# OpenAI Agents SDK Handoffs：Agent 交接机制

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://openai.github.io/openai-agents-python/handoffs/

## 一句话理解

Handoff 是一种控制权转移机制：一个 Agent 判断自己不适合继续处理时，把任务交给另一个更专业的 Agent。

## Handoff 不是简单函数调用

函数调用一般是：

```text
Agent 调用工具 → 工具返回结果 → 原 Agent 继续控制
```

Handoff 是：

```text
Agent 判断需要转交 → 专业 Agent 接管后续对话/任务
```

所以它更像“转人工客服”或“转专业部门”。

## 什么时候适合 Handoff

- 用户问题进入另一个专业领域。
- 后续对话应该由专业 Agent 处理。
- 当前 Agent 不应该继续持有控制权。
- 某个子任务具有持续上下文。

## 什么时候不适合

- 只是临时查一下资料。
- 只是执行一个工具。
- 子任务完成后仍要回到主控。
- 需要中心化强审计。

这些更适合 Manager / Subagent 模式。

## 设计问题

使用 Handoff 前要回答：

1. 转交条件是什么？
2. 转交时带哪些上下文？
3. 转交后谁负责最终输出？
4. 是否允许再转交？
5. 如何记录和追踪转交流程？

## 与 LangGraph 的关系

LangGraph 可以把 handoff 显式做成条件边：

```text
triage_node → billing_agent
triage_node → tech_agent
triage_node → general_agent
```

这样更容易观测和控制。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
