# Agent 开发思想学习资料包

> 说明：本资料包不是网页全文翻译，而是基于官方文档和官方工程文章整理的中文学习版笔记。  
> 目标：帮助你系统学习 Agent 开发思想，而不是只学框架 API。

## 推荐阅读顺序

1. `01_building_effective_agents.md`
2. `02_langgraph_workflows_agents.md`
3. `03_context_engineering.md`
4. `04_writing_effective_tools.md`
5. `05_langgraph_overview.md`
6. `06_thinking_in_langgraph.md`
7. `07_langchain_agents.md`
8. `08_langchain_multi_agent.md`
9. `09_openai_agents_sdk_agents.md`
10. `10_openai_agents_sdk_handoffs.md`
11. `11_openai_agents_sdk_tracing.md`
12. `12_mcp_intro.md`
13. `13_mcp_architecture.md`
14. `14_mcp_tools_spec.md`

## 学习主线

```text
Agent 设计思想
  ↓
Workflow / Agent 边界
  ↓
上下文工程
  ↓
工具工程
  ↓
LangGraph 状态编排
  ↓
多 Agent 模式
  ↓
Tracing / Handoff / Guardrails
  ↓
MCP 标准化接入
```

## 每篇学习时建议回答的问题

1. 这篇解决什么设计问题？
2. 它反对什么常见误区？
3. 它给出的抽象边界是什么？
4. 我自己的业务系统里，哪里能套用这个思想？
5. 如果做成项目，应该拆成哪些节点、工具和状态？

## 版权说明

这些文档是学习笔记和中文讲解，不是原网页逐字翻译。请结合每篇开头的原文链接阅读官方资料。
