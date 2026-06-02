# Anthropic《Building Effective Agents》精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://www.anthropic.com/engineering/building-effective-agents

## 1. 这篇文章解决什么问题？

这篇文章不是在教某个具体框架，而是在回答一个非常根本的问题：

> 当我们要构建 LLM Agent 系统时，应该优先采用什么样的架构思想？

Anthropic 的核心结论是：成功的 Agent 系统通常不是最复杂的系统，而是由简单、可组合、可调试的模式构成。

这对于学习 Agent 非常重要，因为很多人一开始就会陷入“我要不要多 Agent、我要不要 AutoGPT、我要不要复杂规划器”的问题。但原文的基本态度是：先简单，只有当简单方案不够时，再逐步增加复杂度。

## 2. 原文结构理解

这篇文章大致分成这些部分：

1. 什么是 agents。
2. 什么时候该用 agent，什么时候不该用。
3. 框架应该怎么用。
4. 基础构件：增强型 LLM。
5. 几种 workflow 模式。
6. 更自主的 agent。
7. 如何组合这些模式。
8. 生产中的实际案例。
9. 工具设计附录。

你可以把它理解成一套从简单到复杂的 Agent 架构阶梯：

```text
单次 LLM 调用
  ↓
增强型 LLM：LLM + Retrieval + Tools + Memory
  ↓
Prompt Chaining
  ↓
Routing
  ↓
Parallelization
  ↓
Orchestrator-Workers
  ↓
Evaluator-Optimizer
  ↓
Autonomous Agent
```

## 3. Agent 和 Workflow 的关键区别

### Workflow

Workflow 是指 LLM 和工具通过预定义代码路径被编排。流程基本由开发者控制。

例如：

```text
用户输入 → 意图分类 → 检索文档 → 生成答案 → 格式校验 → 输出
```

这里 LLM 可以参与分类、生成，但流程路径主要由代码控制。

### Agent

Agent 是指 LLM 动态决定自己的流程和工具使用方式。

例如：

```text
用户给一个目标
  ↓
LLM 自己判断是否需要搜索
  ↓
搜索后再判断是否需要读文件
  ↓
读完后再判断是否需要写代码
  ↓
运行测试
  ↓
根据测试结果继续修改
```

这就是更强的自主性。

## 4. 什么时候不要用 Agent？

很多任务根本不需要 agentic system。比如：

- 简单问答。
- 简单摘要。
- 固定格式抽取。
- 单次 RAG 就能回答的问题。
- 规则路径非常明确的问题。

这些场景用：

```text
LLM call + retrieval + examples + output schema
```

往往就够了。

## 5. 为什么不要过早使用复杂框架？

原文并不是反对框架，而是提醒：

1. 框架可以快速启动。
2. 但框架也可能掩盖底层 prompt、tool call、message 传递。
3. 复杂抽象会让调试变困难。
4. 初学者可能以为“用了复杂框架 = 系统更高级”。

工程上，真正要避免的是：

```text
你不知道框架底层实际发给模型了什么。
```

所以学习时建议：

- 初期可以用框架。
- 但要能还原到底层是哪些 messages、tools、states。
- 生产系统中，要能 trace 每一步。

## 6. 增强型 LLM：Agent 的最小积木

原文把 augmented LLM 作为基础构件：

```text
LLM + Retrieval + Tools + Memory
```

这说明一件事：不是多 Agent 才叫 Agent，先把单个 LLM 增强好，才是基础。

### Retrieval

让模型能查知识，而不是只依赖参数记忆。

### Tools

让模型能执行动作，例如查数据库、读文件、运行代码、调用 API。

### Memory

让模型能保留任务中的重要状态或长期偏好。

### 对工程的启发

你做任何 Agent 项目，第一步不是“几个 Agent”，而是先问：

```text
这个任务需要哪些外部知识？
这个任务需要哪些工具？
需要保留哪些状态？
```

![Augmented LLM](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fd3083d3f40bb2b6f477901cc9a240738d3dd1371-2401x1000.png&w=3840)

## 7. Prompt Chaining：提示链模式

Prompt Chaining 是把复杂任务拆成一串较简单的 LLM 调用，每一步处理前一步的输出。

例如：

```text
写提纲 → 检查提纲 → 写正文 → 检查正文 → 润色
```

适合任务可以明确拆解成固定步骤的时候。

### 工程价值

- 每一步 prompt 更简单。
- 中间结果可检查。
- 失败点容易定位。
- 可以加入 gate，比如 JSON 校验、字段完整性校验。

### 常见误区

不要把所有任务都拆得过细。拆得太细会增加延迟、成本、中间错误传播和工程复杂度。

![Prompt Chaining](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840)

## 8. Routing：路由模式

Routing 是先对输入分类，再把任务送到不同的处理链路。

例如客服系统：

```text
退款问题 → refund_agent
技术问题 → tech_agent
普通咨询 → faq_agent
投诉问题 → escalation_agent
```

### 工程价值

- 降低单个 prompt 的复杂度。
- 每个分支可以独立优化。
- 可以把简单问题交给便宜模型，把复杂问题交给强模型。
- 降低上下文干扰。

### 常见误区

路由分类本身也可能出错，所以需要置信度、fallback、人工介入、多标签或二级分类机制。

![Routing](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840)

## 9. Parallelization：并行模式

原文提到两种并行方式：

### Sectioning

把任务拆成不同部分并行处理。例如代码审查：

```text
安全性检查
性能检查
可读性检查
测试覆盖检查
```

### Voting

同一个问题让多个模型或多个 prompt 独立判断，再投票或聚合。

### 工程启发

这适合做 reviewer 系统。比如你做代码 Agent，可以让不同 reviewer 分别评估 bug 风险、安全风险、风格规范、业务逻辑完整性。

![Parallelization](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840)

## 10. Orchestrator-Workers：编排器-工人模式

一个中心 LLM 动态拆解任务，然后分配给多个 worker，再综合结果。

这和并行模式不同：

- 并行模式的子任务往往预先定义。
- Orchestrator-workers 的子任务由中心模型动态决定。

特别适合代码修改、多文件分析、复杂搜索、需求拆解、不知道具体子任务数量的工作。

![Orchestrator Workers](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840)

## 11. Evaluator-Optimizer：评估器-优化器模式

一个模型生成结果，另一个模型评估结果并给反馈，然后循环改进。

```text
Generator → Evaluator → Feedback → Generator
```

### 什么时候用？

需要满足两个条件：

1. 评估标准比较明确。
2. 根据反馈确实能改进结果。

适合翻译润色、代码审查、文档生成、专利交底书完善、复杂检索结果验证。

![Evaluator Optimizer](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840)

## 12. Autonomous Agent：自主 Agent

自主 Agent 会基于环境反馈循环行动：

```text
目标 → 计划 → 工具调用 → 观察结果 → 更新计划 → 继续行动
```

它适合难以预先写死路径的问题，例如修复复杂代码问题、自动分析大量资料、多轮搜索和验证、操作计算机完成任务。

### 风险

自主性越强，成本和错误累积风险越高。所以必须有 sandbox、最大迭代次数、停止条件、human checkpoint、工具权限控制、tracing。

![Autonomous Agent](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840)

## 13. 原文最后的三条原则

### 1. 保持设计简单

复杂度必须服务于结果，而不是服务于炫技。

### 2. 提高透明度

显式展示 Agent 的计划、步骤、工具调用、中间结果。

### 3. 认真设计 Agent-Computer Interface

给 Agent 用的工具接口，也要像给人用的界面一样认真设计。

## 14. 和你的学习关系

你以后看到一个需求，要先问：

```text
1. 单次 LLM 能不能解决？
2. 需要 retrieval 吗？
3. 需要 tools 吗？
4. 固定 workflow 能不能解决？
5. 是否需要 routing？
6. 是否需要 reviewer？
7. 是否真的需要自主 Agent？
```

## 15. 精读练习

1. 你现在手上一个任务，判断它是 workflow 还是 agent。
2. 设计一个 prompt chaining 流程。
3. 设计一个 routing 流程。
4. 设计一个 evaluator-optimizer 流程。
5. 写出哪些动作必须 human approval。
