# Thinking in LangGraph 精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

## 1. 核心价值

这篇文档最重要的思想是：先从流程出发，再设计状态和节点，最后才写 prompt 与代码。

## 2. 第一步：从业务流程开始

不要一开始就问我要几个 Agent、我要用哪个框架、我要写什么 prompt。

应该先问：

```text
真实业务流程是什么？
人类专家是怎么做这件事的？
这件事有哪些步骤？
哪些步骤有决策？
哪些步骤会失败？
哪些步骤需要人工确认？
```

## 3. 第二步：识别节点类型

| 类型 | 适合 |
|---|---|
| LLM 节点 | 理解、分类、总结、推理、生成、评审 |
| 数据节点 | 查数据库、查向量库、调接口、读取文件 |
| 动作节点 | 写文件、发邮件、创建工单、提交代码 |
| 用户输入节点 | 人工审批、补充信息、高风险动作确认 |

## 4. 第三步：设计 State

State 不应该只是聊天历史。它应该是结构化的任务状态。

```python
class State(TypedDict):
    user_input: str
    task_type: str
    plan: str
    retrieved_context: list
    draft: str
    review_result: str
    risk_level: str
    approved: bool
    errors: list
```

## 5. State 设计原则

1. 保存原始数据。
2. 区分业务数据和控制字段。
3. 避免把所有内容塞到 messages。
4. 为恢复和调试设计。

## 6. 第四步：设计节点边界

节点边界决定观测边界、恢复边界、测试边界、责任边界。

好节点的特征：

```text
输入清楚
输出清楚
只做一件事
可单独测试
错误可处理
```

差节点：一个节点里完成分类、检索、生成、评审、保存。

## 7. 第五步：设计边和条件路由

条件边根据 state 决定：

```text
if task_type == "code":
    coder
elif task_type == "patent":
    patent_writer
else:
    general_writer
```

Reviewer 后：

```text
if passed:
    final
elif iteration_count < max:
    revise
else:
    human_review
```

## 8. 示例：代码分析 Agent

```text
START
  ↓
understand_request
  ↓
search_repo
  ↓
plan_change
  ↓
edit_code
  ↓
run_tests
  ↓
review
  ↓
END / revise
```

## 9. 常见误区

- 先写 prompt 后想流程。
- State 只存 messages。
- 节点越少越简单。
- 所有失败都靠 LLM 自我修复。

## 10. 练习

把“论文精读 Agent”拆成 LangGraph，设计 State 和节点。
