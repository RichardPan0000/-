# OpenAI Agents SDK：Agents 抽象精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://openai.github.io/openai-agents-python/agents/

## 1. 核心价值

OpenAI Agents SDK 的 Agent 抽象适合帮助你理解生产级 Agent 到底由哪些元素组成。

它不是只有 prompt，而是一个结构化对象。

## 2. Agent 的组成

一个 Agent 通常包含：

- name。
- instructions。
- model。
- tools。
- handoffs。
- guardrails。
- output type。
- context。
- hooks。
- tracing。

这说明 Agent 是一个运行单元，而不是简单 prompt。

## 3. Instructions

instructions 类似系统提示词，但在生产系统里要更结构化：

```text
角色
目标
输入解释
输出格式
工具使用规则
不该做什么
风险边界
失败时怎么办
```

## 4. Tools

工具让 Agent 执行外部动作，例如查知识库、读文件、写报告、创建工单、调数据库、运行代码。

## 5. Handoffs

handoffs 表示这个 Agent 可以把任务转交给其他 Agent。这不是简单工具调用，而是控制权转移。

## 6. Guardrails

guardrails 是生产安全层，可用于输入检查、输出检查、风险动作拦截、敏感信息过滤、结构校验、合规校验。

## 7. Structured Outputs

生产系统中，很多 Agent 不应该只输出自然语言。

例如 reviewer 应该输出：

```json
{
  "passed": true,
  "risk_level": "low",
  "missing_items": [],
  "suggestions": []
}
```

结构化输出有利于后续路由、自动评估、数据入库、流程控制。

## 8. 和 LangGraph 的关系

```text
OpenAI Agents SDK：定义 Agent 运行单元
LangGraph：定义多个运行单元之间的流程和状态
```

## 9. 练习

定义三个 Agent：code_reviewer、patent_writer、iv_diagnosis_explainer。每个写 name、instructions、tools、handoffs、guardrails、output_type。
