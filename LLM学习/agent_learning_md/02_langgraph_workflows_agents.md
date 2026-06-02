# LangGraph Workflows and Agents：Workflow 与 Agent 的边界

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://docs.langchain.com/oss/python/langgraph/workflows-agents

## 一句话理解

这篇文档适合用来建立一个判断标准：**什么时候用固定流程，什么时候让 LLM 动态决策。**

## 核心思想

### Workflow

Workflow 是预先定义好的路径。比如：

```text
输入 → 分类 → 检索 → 生成 → 审核 → 输出
```

这种方式更稳定、可测试、可控，适合生产系统中的确定性流程。

### Agent

Agent 的路径不完全固定，它可以根据中间结果动态选择下一步。比如：

```text
输入 → 思考 → 选择工具 → 观察结果 → 再决定下一步
```

它适合那些子任务数量不确定、需要连续决策的问题。

## 关键判断

| 问题 | 更适合 |
|---|---|
| 任务路径清楚吗？ | Workflow |
| 每一步输入输出稳定吗？ | Workflow |
| 需要模型自己决定下一步吗？ | Agent |
| 需要反复尝试和根据反馈调整吗？ | Agent |
| 需要人工审批吗？ | Workflow + HITL |
| 需要长期运行与恢复吗？ | LangGraph |

## 实战建议

生产环境里常见的最佳结构不是纯 Agent，而是：

```text
固定 workflow 负责主流程
LLM/Agent 负责局部判断
工具负责外部动作
Reviewer 负责质量检查
Human-in-the-loop 负责高风险动作
```

## 你应该学到什么

1. 不要把所有逻辑都交给 Agent。
2. 先把流程图画出来，再决定哪些节点需要 LLM。
3. Agent 自由度越大，越需要 tracing、评估和 guardrails。
4. Workflow 是生产系统的骨架，Agent 是其中的智能决策部件。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
