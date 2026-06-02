# Anthropic《Effective Context Engineering for AI Agents》精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

## 1. 核心问题

这篇文章讲的是如何给 Agent 构造合适的上下文环境。很多 Agent 效果不好，不是模型不够强，而是上下文给错、给多、给乱。

## 2. Prompt Engineering 与 Context Engineering

Prompt Engineering 更关注“指令怎么写”；Context Engineering 更关注“模型运行时能看到什么信息，这些信息如何选择、排序、压缩、隔离和更新”。

也就是说，context engineering 是系统层面的 prompt engineering。

## 3. 为什么上下文不是越多越好？

长上下文会带来：

- 注意力稀释。
- 成本上升。
- 冲突信息增加。
- 模型更难判断重点。
- 旧信息污染新任务。

## 4. 上下文工程的核心任务

1. 选择：哪些信息应该进入上下文？
2. 排序：最重要的信息放在哪里？
3. 压缩：长文档、工具结果、历史对话如何摘要？
4. 隔离：不同 Agent 是否应该看到不同上下文？
5. 更新：什么时候加入新信息，什么时候删除旧信息？
6. 持久化：哪些信息进入长期记忆，哪些只保留当前任务？

## 5. 多 Agent 中的上下文隔离

多 Agent 的一个重要价值就是上下文隔离。

例如代码修复系统：

| Agent | 应该看到 | 不一定需要看到 |
|---|---|---|
| Planner | 用户需求、项目结构、约束 | 所有源码 |
| Searcher | 查询目标、文件索引、相关代码片段 | 最终文档格式 |
| Coder | 需求、相关代码、修改目标 | 检索噪声 |
| Reviewer | diff、需求、测试结果、评审标准 | 大量原始历史 |

## 6. Context Budget

生产系统里应该为每类上下文设置预算：

```text
System Prompt：1000 tokens
User Goal：500 tokens
Relevant Docs：4000 tokens
Tool Results：2000 tokens
Memory：1000 tokens
Output Schema：500 tokens
```

这能防止上下文无限膨胀。

## 7. State 和 Context 的区别

State 是系统内部保存的原始状态；Context 是发给模型的最终消息内容。

好的系统会把二者分开：

```text
State 存原始信息
Node 根据需要格式化成 Context
```

## 8. 工程落地模式

### 阶段性上下文

```text
Plan 阶段：目标 + 约束
Research 阶段：目标 + 查询
Write 阶段：目标 + 研究摘要
Review 阶段：目标 + 草稿 + 标准
```

### 角色化上下文

每个 Agent 有自己的信息边界。

### 工具结果压缩

工具返回不要原样塞入，而是压缩成关键发现、证据来源、不确定点、下一步建议。

## 9. 业务应用

### 专利 Agent

上下文分层：技术方案、已有背景、现有方案缺陷、发明点、权利要求限制、审查风险。

### 光伏 IV 诊断 Agent

上下文分层：电站信息、IV 曲线特征、历史数据、气象条件、组件参数、故障知识库、诊断规则。

### Coding Agent

上下文分层：需求、repo map、相关文件、依赖关系、测试结果、代码规范。

## 10. 练习

给你的“代码审查 Agent”设计上下文策略，明确每个 Agent 应该看到什么，不应该看到什么。
