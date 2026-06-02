# LangGraph《Workflows and Agents》精读

> 版权与使用说明：本文档是面向学习的中文精读笔记，不是原网页全文逐字翻译。  
> 处理方式：保留原文链接、结构化复述原文思想、补充中文解释、工程化理解、练习题与个人项目迁移建议。  
> 图片处理：图片采用官方网页公开图片 URL 进行 Markdown 引用，不重新分发图片文件。  


原文链接：  
https://docs.langchain.com/oss/python/langgraph/workflows-agents

## 1. 核心价值

这篇文档最重要的价值是帮你区分 workflow 和 agent。

```text
Workflow：流程由代码预先定义
Agent：流程由模型动态决定
```

生产系统更推荐：能确定的用 workflow 固定，不确定的局部用 agent 决策。

## 2. Workflow 的本质

Workflow 是确定性更强的流程，例如：

```text
输入 → 分类 → 检索 → 生成 → 校验 → 输出
```

优点是稳定、容易测试、容易调试、成本可控、结果一致性高。缺点是灵活性差，无法很好处理开放式任务。

## 3. Agent 的本质

Agent 是让 LLM 根据当前状态动态决定下一步：

```text
目标 → 判断需要搜索 → 搜索 → 判断是否读文件 → 读文件 → 写代码 → 测试 → 继续修复
```

优点是灵活，适合开放任务和未知步骤数。缺点是成本高、延迟高、调试难、错误可能累积。

## 4. LangGraph 为什么适合混合架构？

LangGraph 的图结构可以表达：

- 固定 workflow：普通边。
- 动态决策：条件边。
- 循环改进：回边。
- 人工审批：interrupt。
- 状态持久化：checkpoint。
- 多 Agent：不同节点是不同 agent。

所以它适合：

```text
固定主流程 + 局部动态决策 + 人工审批 + 持久化状态
```

## 5. 典型模式

### Prompt Chaining

```text
生成提纲 → 检查提纲 → 写正文 → 润色
```

适合可明确拆解的任务。

### Routing

```text
if task_type == "code":
    coder
elif task_type == "doc":
    writer
else:
    general
```

适合输入类别稳定的场景。

### Parallelization

并行执行多个子任务：

```text
安全审查
性能审查
风格审查
```

最后汇总。

### Orchestrator-Worker

中心节点动态拆解任务，再分配 worker。

### Evaluator-Optimizer

生成器生成，评审器反馈，然后循环。

## 6. 生产落地建议

1. 路径越关键，越要固定。
2. 判断越模糊，越适合 LLM。
3. 动作越危险，越需要审批。
4. 循环一定要有停止条件。
5. 不要把所有节点都做成 Agent。

## 7. 常见误区

- Workflow 不智能：错，workflow 中也可以有很多 LLM 节点。
- Agent 越自主越高级：错，越自主越不可控。
- 多 Agent 就是生产级：错，没有状态、审计、评估、权限控制的多 Agent 只是复杂 demo。

## 8. 练习

设计一个“专利交底书 Agent”的 workflow：

```text
技术输入
  ↓
背景技术抽取
  ↓
现有方案缺陷分析
  ↓
发明点生成
  ↓
技术效果总结
  ↓
审查员视角评审
  ↓
最终文档
```

思考哪些节点固定，哪些节点用 LLM，哪些节点需要 reviewer。
