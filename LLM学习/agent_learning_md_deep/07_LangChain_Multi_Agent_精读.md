# LangChain Multi-Agent 精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://docs.langchain.com/oss/python/langchain/multi-agent

## 1. 核心问题

什么时候需要多个 Agent？多个 Agent 应该如何协作？

重点不是“创建很多 Agent”，而是职责拆分、上下文隔离和控制权设计。

## 2. 为什么需要多 Agent？

单 Agent 的问题：prompt 越来越长、工具越来越多、角色混乱、上下文噪声大、不同任务互相干扰、难以优化局部能力。

多 Agent 可以带来专家化、上下文隔离、工具隔离、分工更清晰、更容易独立测试。

但代价是成本上升、延迟增加、调试复杂、状态同步复杂、失败链路更长。

## 3. Subagents 模式

主 Agent 把子 Agent 当作工具调用：

```text
Supervisor
  ├─ Research Agent
  ├─ Code Agent
  └─ Review Agent
```

适合强控制、中心化审计、子任务明确、最终答案由 supervisor 汇总。

## 4. Handoffs 模式

一个 Agent 把控制权交给另一个 Agent：

```text
Triage Agent → Billing Agent
Triage Agent → Technical Agent
```

适合多轮对话、专业领域转接、用户需要直接和专业 Agent 交互。

## 5. Router 模式

先分类，再分发：

```text
if input_type == "code":
    code_agent
elif input_type == "paper":
    paper_agent
elif input_type == "patent":
    patent_agent
```

适合输入类别稳定、每类处理路径差异明显的场景。

## 6. Skills 模式

不是创建多个长期 Agent，而是让一个 Agent 按需加载能力。适合能力模块多，但不希望多 Agent 通信复杂的场景。

## 7. 什么时候不要多 Agent？

- 单 Agent + 工具就能解决。
- 任务上下文很短。
- 工具数量少。
- 没有明显角色边界。
- 结果不需要多个视角。
- 成本和延迟更重要。

## 8. 多 Agent 设计检查表

1. 为什么单 Agent 不够？
2. 每个 Agent 的职责是什么？
3. 每个 Agent 的输入输出是什么？
4. 谁拥有最终控制权？
5. 谁和用户对话？
6. 哪些上下文共享？
7. 哪些上下文隔离？
8. 失败后谁负责修正？
9. 是否需要 reviewer？
10. 是否需要 tracing？

## 9. 针对你的推荐结构

```text
Supervisor
  ├─ Standard Researcher：查标准/论文/资料
  ├─ Domain Analyst：做光伏储能技术分析
  ├─ Code Agent：写代码/脚本
  ├─ Patent Writer：写交底书
  └─ Reviewer：检查逻辑与完整性
```
