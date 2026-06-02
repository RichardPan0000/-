# Building Effective Agents：有效 Agent 的设计思想

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://www.anthropic.com/engineering/building-effective-agents

## 一句话理解

这篇文章的核心思想是：**不要为了 Agent 而 Agent；先用最简单、可组合、可观察的模式解决问题，只有当简单方案不够时，再增加复杂度。**

## 核心概念

### 1. Workflow 和 Agent 的区别

- **Workflow**：流程路径基本由代码预先定义，LLM 只是其中一个节点。
- **Agent**：LLM 在执行过程中动态决定下一步、选择工具、调整路线。

生产系统里往往不是二选一，而是二者混合：核心流程固定，局部决策交给模型。

### 2. Augmented LLM 是基础

Agent 的基础不是“多 Agent”，而是一个增强型 LLM：

- Retrieval：能查知识。
- Tools：能调用外部能力。
- Memory：能保存上下文和中间状态。
- Feedback：能基于环境结果继续行动。

### 3. 常见模式

| 模式 | 适合场景 | 关键思想 |
|---|---|---|
| Prompt Chaining | 可拆成固定步骤的任务 | 每步变简单，中间加 Gate |
| Routing | 输入类型差异明显 | 先分类，再分发 |
| Parallelization | 多角度检查或独立子任务 | 并行执行再聚合 |
| Orchestrator-Workers | 子任务不确定 | 中心模型动态拆分 |
| Evaluator-Optimizer | 有明确评审标准 | 生成-评审-改进 |
| Autonomous Agent | 开放式复杂任务 | 模型循环计划、行动、反馈 |

## 对定制开发的启发

1. 先设计最小可用 workflow，不要直接堆多 Agent。
2. 评估每增加一个 LLM 调用是否真的提升结果。
3. Tool 文档和接口设计要像写 API 文档一样认真。
4. 必须有停止条件、最大循环次数和人工介入点。
5. Agent 不是为了“看起来智能”，而是为了完成难以固定编码的任务。

## 适合你的落地方向

你可以把这套思想用到：

- 代码分析 Agent：Plan → Search Repo → Modify → Test → Review。
- 专利文档 Agent：Extract → Analyze → Draft → Review。
- 光伏/储能诊断 Agent：Data Check → Knowledge Retrieval → Diagnosis → Explanation → Human Review。


## 图示 / 图片

![Augmented LLM](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fd3083d3f40bb2b6f477901cc9a240738d3dd1371-2401x1000.png&w=3840)

> 增强型 LLM：LLM + Retrieval + Tools + Memory 是 Agent 系统的基础能力。

![Prompt Chaining](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840)

> 提示链：把任务拆成多个确定步骤，中间可加入 Gate 检查。

![Routing](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840)

> 路由：先分类，再交给不同专家链路处理。

![Parallelization](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840)

> 并行化：多个 LLM 调用并行执行，再汇总。

![Orchestrator Workers](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840)

> 编排器-工人：中心模型动态拆分任务，分配给 worker，再综合结果。

![Evaluator Optimizer](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840)

> 评估器-优化器：生成器产出方案，评审器给反馈，循环优化。

![Autonomous Agent](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840)

> 自主 Agent：模型根据环境反馈循环行动，直到任务完成或触发停止条件。

![Coding Agent Flow](https://www.anthropic.com/_next/image?q=75&url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F4b9a1f4eb63d5962a6e1746ac26bbc857cf3474f-2400x1666.png&w=3840)

> 编码 Agent 高层流程：用户澄清任务，Agent 搜索文件、写代码、测试、返回结果。



## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
