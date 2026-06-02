# Thinking in LangGraph：如何用图思维设计 Agent

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

## 一句话理解

这篇文档最重要的思想是：**先从流程出发，再设计状态和节点，最后才写 prompt 与代码。**

## 五步思考法

### 第 1 步：把业务流程拆成离散步骤

例如客服邮件 Agent：

```text
Read Email → Classify Intent → Search Docs → Draft Reply → Human Review → Send Reply
```

每个步骤未来都可以成为一个 node。

### 第 2 步：识别节点类型

常见节点类型：

| 类型 | 说明 |
|---|---|
| LLM Step | 理解、分类、生成、推理 |
| Data Step | 检索文档、查数据库、查 API |
| Action Step | 发邮件、建工单、写文件 |
| User Input Step | 人工审批、补充信息 |

### 第 3 步：设计 State

State 是 Agent 的共享记忆。应该存：

- 原始输入。
- 分类结果。
- 检索结果。
- 中间草稿。
- 审批结果。
- 错误和调试信息。

不要在 State 里存已经格式化好的 prompt。状态应该尽量保存原始数据，需要时在节点内格式化。

### 第 4 步：构建节点

节点应该职责单一：

```text
一个节点只做一件事
有清晰输入
有清晰输出
有错误处理策略
```

### 第 5 步：连接边

边表示流程走向。条件边表示根据状态做路由：

```text
if classification == "bug":
    go_to("bug_track")
else:
    go_to("draft_reply")
```

## 对你的启发

你以后做任何 Agent 系统，都可以先画这张表：

| 节点 | 类型 | 输入 State | 输出 State | 失败策略 | 是否需要人审 |
|---|---|---|---|---|---|

这比直接写 prompt 稳得多。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
