# LangGraph Overview 精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://docs.langchain.com/oss/python/langgraph/overview

## 1. LangGraph 的定位

LangGraph 是面向 Agent 和复杂 workflow 的编排框架。关键词是 stateful、durable、controllable、human-in-the-loop、observable。

可以把它理解为给 Agent 系统用的状态机 / 图执行引擎。

## 2. 为什么需要 LangGraph？

普通 LLM 应用可能只需要：

```text
prompt → model → answer
```

但生产级 Agent 往往需要：

```text
输入 → 分类 → 检索 → 规划 → 工具调用 → 评审 → 返工 → 人工审批 → 输出
```

这个过程涉及多步骤、分支、循环、失败恢复、状态保存、用户中断、人工审批、多 Agent 分工。

## 3. 核心概念

### State

所有节点共享的状态，例如 user_input、plan、retrieved_docs、draft、review、next_step。

### Node

一个执行步骤，可以是普通 Python 函数、LLM 调用、LangChain Agent、工具调用、审批节点。

### Edge

节点之间的连接。

### Conditional Edge

根据状态决定下一步。

### Checkpoint

保存执行状态，支持恢复。

### Interrupt

暂停流程，等待人类输入。

## 4. 为什么 State 很关键？

没有显式 State 的 Agent 系统通常会靠聊天历史硬传递状态，这会导致难以调试、难以恢复、难以测试、中间结果不清晰、多 Agent 互相污染上下文。

显式 State 可以清楚知道当前任务处在哪一步、已经产生什么结果、下一步该去哪、有没有错误、是否需要人审。

## 5. Durable Execution

生产任务可能会中断：服务重启、工具超时、人工审批等待、用户稍后恢复、外部系统失败。有 checkpoint 后，可以从已保存状态继续，而不是从头开始。

## 6. Human-in-the-loop

很多动作不能自动执行，比如发邮件、删除数据、修改生产配置、提交代码、发布版本、对客户做承诺。LangGraph 可以在某个节点 interrupt，然后等待人类批准或修改。

## 7. LangGraph 和 LangChain 的关系

| 组件 | 主要作用 |
|---|---|
| LangChain | 模型调用、工具、Agent 抽象 |
| LangGraph | 流程图、状态、路由、持久化 |
| LangSmith | 观测、评估、调试 |

## 8. 典型生产图

```text
START
  ↓
input_guard
  ↓
router
  ↓
planner
  ↓
researcher
  ↓
worker
  ↓
reviewer
  ↓
[passed?]
  ├─ yes → final
  └─ no  → worker
  ↓
END
```

## 9. 练习

把你现在一个业务流程拆成节点，写出每个节点的输入 State、输出 State、是否 LLM、是否工具、是否人审。
