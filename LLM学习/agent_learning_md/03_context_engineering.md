# Effective Context Engineering：上下文工程

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

## 一句话理解

这篇文章的核心是：**Agent 的效果很大程度上取决于你给它什么上下文，而不是只取决于模型能力。**

## Prompt Engineering 与 Context Engineering

- Prompt Engineering 更像是“怎么写指令”。
- Context Engineering 更像是“怎么组织任务所需的信息环境”。

对 Agent 来说，真正重要的是：

```text
模型当前能看到什么？
不能看到什么？
什么时候加载新上下文？
什么时候丢弃旧上下文？
```

## 核心思想

### 1. 上下文不是越多越好

长上下文会带来：

- 干扰信息增加。
- 重要信息被稀释。
- 成本增加。
- 模型注意力分散。

### 2. 每个 Agent 应该只看到自己需要的信息

例如多 Agent 系统：

- Planner：看用户目标和约束。
- Researcher：看问题和检索需求。
- Coder：看实现目标、接口、相关代码片段。
- Reviewer：看产出和评审标准。

### 3. 上下文应该动态加载

不要把所有文档一次性塞进去，而是按任务阶段加载：

```text
任务阶段 → 查询需求 → 检索相关上下文 → 压缩 → 注入 Agent
```

## 对生产系统的启发

1. 需要设计上下文窗口预算。
2. 需要做检索结果压缩。
3. 需要把状态和 prompt 分开。
4. 需要区分长期记忆、短期工作记忆、临时工具结果。
5. 多 Agent 的本质之一就是上下文隔离。

## 你的实践练习

设计一个“代码审查 Agent”的上下文策略：

- Planner 看：需求、仓库结构。
- Code Reader 看：相关文件。
- Reviewer 看：diff、测试结果、规范。
- Final Writer 看：评审结论和修复建议。


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
