# OpenAI Agents SDK：Handoffs 与 Tracing 精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


Handoffs 原文：  
https://openai.github.io/openai-agents-python/handoffs/

Tracing 原文：  
https://openai.github.io/openai-agents-python/tracing/

## 1. 为什么一起学？

多 Agent 系统真正难的不是转交，而是转交为什么发生、转交了什么上下文、转交后谁负责、哪里出错了。

Handoff 解决控制权转移，Tracing 解决可观测性。

## 2. Handoff 的本质

Handoff 是 Agent 把任务交给另一个 Agent。类似客服前台转技术支持、退款部门、账单部门。

### Tool Call 与 Handoff 区别

Tool Call：Agent 仍然掌握控制权，工具只是执行动作，返回后原 Agent 继续处理。

Handoff：控制权交给另一个 Agent，后续任务由新 Agent 处理。

## 3. 什么时候用 Handoff？

适合多轮对话角色切换、专业领域明显不同、新 Agent 需要持续维护上下文、用户需要直接和专业 Agent 对话。

不适合只是临时查资料、执行一个动作、需要主控统一汇总、子任务短小明确的场景。

## 4. Handoff 设计要点

1. 转交条件。
2. 上下文裁剪。
3. 输入过滤。
4. 返回机制。
5. 防止转交循环。

## 5. Tracing 的本质

Tracing 是记录 Agent 执行轨迹，解决为什么 Agent 做了这个决策、它用了哪些工具、在哪里失败、成本和延迟在哪里。

## 6. Tracing 应该记录什么？

| 内容 | 目的 |
|---|---|
| trace_id | 串联完整请求 |
| input | 复盘用户输入 |
| model_call | 查看模型调用 |
| prompt/context | 检查上下文 |
| tool_call | 查看工具选择 |
| tool_args | 检查参数 |
| tool_result | 检查外部结果 |
| handoff | 查看控制权转移 |
| guardrail | 查看安全拦截 |
| token_usage | 成本分析 |
| latency | 性能分析 |

## 7. 生产调试流程

当 Agent 输出不好时，不要直接改 prompt。按下面顺序查：

```text
输入是否被正确理解？
路由是否正确？
上下文是否足够？
上下文是否太多？
工具是否被正确选择？
工具参数是否正确？
工具返回是否可用？
是否需要 reviewer？
reviewer 是否有明确标准？
最终输出是否通过 schema？
```

## 8. 练习

为你的多 Agent 系统设计 tracing schema，包含 trace_id、node_name、input、output、tool_calls、latency、error。
