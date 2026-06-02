# LangGraph Overview：低层编排框架的定位

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://docs.langchain.com/oss/python/langgraph/overview

## 一句话理解

LangGraph 是面向长流程、带状态、可恢复、可人工介入的 Agent 编排框架。

## LangGraph 解决什么问题

它不是为了替代模型调用，而是解决这些生产问题：

- 复杂任务需要多个步骤。
- 步骤之间要共享状态。
- 中途可能失败，需要恢复。
- 高风险动作需要人工审批。
- 执行过程需要 tracing 和调试。
- 多 Agent 需要明确路由。

## 与 LangChain 的关系

可以这样理解：

```text
LangChain：更偏模型、工具、单 Agent 抽象
LangGraph：更偏状态机、流程编排、持久化运行
LangSmith：更偏观测、评估、调试
```

## LangGraph 的核心能力

| 能力 | 含义 |
|---|---|
| State | 所有节点共享的状态 |
| Node | 一个离散执行步骤 |
| Edge | 节点之间的连接 |
| Conditional Edge | 根据状态决定下一步 |
| Checkpoint | 保存执行状态 |
| Human-in-the-loop | 中断等待人类输入 |
| Durable Execution | 失败后恢复继续执行 |

## 对生产系统的意义

生产级 Agent 不是一个 while-loop，而是一个有状态、有边界、有恢复点的系统。LangGraph 强迫你显式设计这些东西：

```text
输入是什么？
状态存什么？
节点做什么？
失败怎么处理？
什么时候结束？
什么时候人审？
```


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
