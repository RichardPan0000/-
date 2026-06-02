# LangChain Multi-Agent：多 Agent 模式

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://docs.langchain.com/oss/python/langchain/multi-agent

## 一句话理解

多 Agent 的价值不是数量多，而是**职责拆分、上下文隔离和路由清晰**。

## 常见多 Agent 模式

### 1. Subagents

主 Agent 保持控制权，把子 Agent 当作工具调用。

适合：

- 需要中心化控制。
- 需要强上下文隔离。
- 子任务可以相互独立。

### 2. Handoffs

当前 Agent 把控制权交给另一个 Agent。

适合：

- 多轮对话中角色切换。
- 用户需要直接和专业 Agent 交互。
- 状态需要在专业 Agent 中持续。

### 3. Skills

一个 Agent 按需加载技能和上下文。

适合：

- 单 Agent 但需要多个领域能力。
- 不希望引入太多 Agent。
- 上下文可以动态加载。

### 4. Router

先路由，再交给不同 Agent。

适合：

- 输入类型明显不同。
- 任务类别稳定。
- 希望降低上下文干扰。

## 性能思考

多 Agent 会增加：

- 模型调用次数。
- token 成本。
- 延迟。
- 调试难度。

但是它可以减少：

- 单 Agent 上下文混乱。
- 一个 prompt 承担过多职责。
- 不同任务之间互相干扰。

## 选型建议

| 目标 | 推荐 |
|---|---|
| 强控制 | Subagents |
| 多轮专业对话 | Handoffs |
| 简单任务动态加载能力 | Skills |
| 明确分类分发 | Router |
| 复杂生产流程 | LangGraph custom workflow |

## 关键学习点

多 Agent 的核心问题不是“怎么创建多个 Agent”，而是：

```text
谁拥有控制权？
谁能看到哪些上下文？
谁负责最终答案？
失败后谁来修正？
成本是否可控？
```


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
