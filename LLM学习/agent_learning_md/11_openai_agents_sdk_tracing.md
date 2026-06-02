# OpenAI Agents SDK Tracing：可观测性

> 说明：这是“中文学习版”笔记，不是网页全文逐字翻译。内容以原文思想、关键概念、工程实践和学习练习为主。  
> 原文：https://openai.github.io/openai-agents-python/tracing/

## 一句话理解

Tracing 解决的是生产 Agent 的核心问题：**它为什么这么做？哪里出错了？成本和延迟在哪里？**

## 为什么 Tracing 重要

没有 tracing，你只能看到最终回答，看不到：

- 调用了哪个模型。
- 调了几次工具。
- 每次工具输入输出是什么。
- 是否发生 handoff。
- guardrail 有没有触发。
- 哪一步耗时最长。
- 哪一步产生了错误。

## 典型追踪对象

- LLM generation。
- Tool call。
- Handoff。
- Guardrail。
- 自定义 span。
- 上下文传递。
- 最终输出。

## 对生产系统的启发

Agent 系统上线前，至少应该记录：

| 字段 | 用途 |
|---|---|
| trace_id | 串起完整请求 |
| user_input | 复盘输入 |
| selected_route | 观察路由是否正确 |
| tool_calls | 观察工具使用 |
| intermediate_outputs | 观察中间结果 |
| token_usage | 成本统计 |
| latency | 性能分析 |
| final_output | 结果评估 |

## 调试方式

当 Agent 出错时，不要只改 prompt，应该按顺序看：

```text
输入是否清楚？
路由是否正确？
上下文是否过多或过少？
工具是否被正确调用？
工具返回是否可用？
评审节点是否有效？
最终输出是否符合结构？
```


## 建议练习

1. 用自己的业务场景复述本文核心思想。
2. 写出一个可以落地的小例子，不要只停留在概念。
3. 总结：哪些部分适合固定 workflow，哪些部分适合交给 LLM/Agent 动态决策。
