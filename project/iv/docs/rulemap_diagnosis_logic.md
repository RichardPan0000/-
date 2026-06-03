# Rulemap 诊断逻辑说明

## 1. 文档目的

这份文档专门说明 `rulemap` 如何根据 IV 表象指标推断根因。

它回答三个问题：

- `rulemap` 吃什么输入
- 它如何从表象推断根因
- 当前 `run_jianheng` 接入后，最终三个根因是怎么来的

相关代码：

- `src/rulemap/diagnose.py`
- `src/rulemap/knowledge_graph.yaml`
- `src/analysis/jianheng_rulemap_bridge.py`

## 2. `rulemap` 的定位

`rulemap` 不是简单的标签映射表，也不是把 `diagnosis` 字段硬转换成另一个标签。

它是一套基于知识图谱的推理引擎：

1. 先从输入指标里检测异常表象
2. 再用知识图谱里的矛盾规则修正表象集合
3. 然后把异常表象按权重投票到候选根因
4. 最后用根因期望表象做反向验证，并支持多故障拆解

因此它输出的是“表象、故障模式、候选根因、置信度、推理链路”，而不是单一分类器标签。

## 3. 输入字段

`rulemap.diagnose(...)` 的输入是一个字典，字段来自 IV 曲线表象。

当前字段包括：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `dIsc` | numeric | 短路电流偏差 |
| `dVoc` | numeric | 开路电压偏差 |
| `dImp` | numeric | MPP 电流偏差 |
| `dVmp` | numeric | MPP 电压偏差 |
| `dPmp` | numeric | 最大功率偏差 |
| `dFF` | numeric | 填充因子偏差 |
| `Rs` | bool | 串联电阻异常 |
| `Rsh` | bool | 并联电阻异常 |
| `prestep` | bool | MPP 前台阶 |
| `poststep` | bool | MPP 后台阶 |
| `Isc_near_zero` | special | 由 `dIsc` 接近 -1 间接触发，用于开路型判断 |

在 `run_jianheng` 中，这些输入由 `src/analysis/jianheng_rulemap_bridge.py` 生成。

映射关系是：

| Jianheng 输出字段 | rulemap 输入字段 |
| --- | --- |
| `dIsc_pair` | `dIsc` |
| `dVoc_pair` | `dVoc` |
| `dImp_pair` | `dImp` |
| `dVmp_pair` | `dVmp` |
| `dPmp_pair` | `dPmp` |
| `dFF_pair` | `dFF` |
| `is_high_rs_slope_anomaly` | `Rs` |
| `is_low_rsh_slope_anomaly` | `Rsh` |
| `pre_step` | `prestep` |
| `post_step` | `poststep` |

也就是说，Jianheng 场景里数值表象来自同 MPPT 参考串 pairwise 对比，而不是主诊断流里的健康参考曲线差值。

### 3.1 Runtime `run_diagnosis_with_config` 输入

`run_diagnosis_with_config.py` 接入 rulemap 时，输入来源不同于 Jianheng。

Runtime 主流程使用 `healthy_curve` 已经生成的健康参考曲线对比结果：

```text
DiagnosisResult.details["rule_comparison"]
```

其中的 `dIsc/dVoc/dImp/dVmp/dPmp/dFF` 表示当前组串相对健康曲线参考的偏差。为了方便后续沿用 Jianheng CSV 分析口径，runtime 导出的 CSV 也会提供：

| Runtime CSV 字段 | rulemap 输入字段 | 含义 |
| --- | --- | --- |
| `dIsc_pair` | `dIsc` | 相对健康曲线参考的短路电流偏差 |
| `dVoc_pair` | `dVoc` | 相对健康曲线参考的开路电压偏差 |
| `dImp_pair` | `dImp` | 相对健康曲线参考的 MPP 电流偏差 |
| `dVmp_pair` | `dVmp` | 相对健康曲线参考的 MPP 电压偏差 |
| `dPmp_pair` | `dPmp` | 相对健康曲线参考的最大功率偏差 |
| `dFF_pair` | `dFF` | 相对健康曲线参考的填充因子偏差 |

这里的 `*_pair` 只是为了 CSV 字段兼容，不表示 Jianheng 的参考串 pairwise 对比。

Runtime 的台阶输入来自 `prediction_details.pattern_analysis.steps.pre_step/post_step`。`Rs/Rsh` 输入来自 `prediction_details.slope_anomalies`，并沿用台阶抑制逻辑：

- `post_step=True` 时抑制 `Rs`
- `pre_step=True` 时抑制 `Rsh`

Runtime 复用 `src/rulemap/knowledge_graph.yaml` 的图谱关系，但阈值可通过 `src/rulemap/runtime_healthy_curve_thresholds.yaml` 单独覆盖。该文件初始值复制当前图谱阈值，不代表新调参。

当前 runtime 第一版为了兼容下游，`result.diagnosis` 仍输出旧标签体系，例如 `局部遮挡/隐裂/热斑效应 -> break_shadow`。中文 rulemap 根因、表象和推理链路会保存在 `details["rulemap"]`、`diagnosis_details.csv` 和 `rulemap_final_summary.csv` 中。

## 4. 第一层：表象检测

表象检测由 `DiagnosisEngine.detect_phenomena(...)` 完成。

数值型字段的规则是：

```text
仅负值表示性能下降，才参与异常检测
abs(value) >= threshold 时判定为 abnormal
deviation = (abs(value) - threshold) / threshold
```

例如：

```text
dVoc = -0.08
threshold = 0.05
abs(-0.08) >= 0.05
deviation = (0.08 - 0.05) / 0.05 = 0.6
```

此时 `dVoc` 会被记录为异常表象。

如果 `dVoc = 0.04` 或 `dVoc = +0.08`，都不会被判为异常：

- `0.04` 没超过阈值
- `+0.08` 表示优于参考或没有性能下降

布尔型字段的规则更直接：

```text
True  -> abnormal, deviation = 1.0
False -> normal,   deviation = 0.0
```

特殊字段 `Isc_near_zero` 由 `dIsc` 触发：

```text
dIsc < 0 且 abs(dIsc) >= Isc_near_zero.threshold
```

当前 `knowledge_graph.yaml` 中 `Isc_near_zero.threshold` 默认为 `0.90`。

## 5. 第二层：矛盾规则修正

表象检测完成后，`DiagnosisEngine.apply_contradiction_rules(...)` 会应用知识图谱中的矛盾规则。

这些规则定义在 `knowledge_graph.yaml` 的 `contradiction_rules` 中。

当前主要有三类：

### 5.1 排除型规则

排除型规则用于避免一个更强表象把其它派生表象重复解释。

当前典型规则：

- `Isc_near_zero` 触发时，排除 `dIsc/dImp/dPmp`
- `prestep` 触发时，排除 `dPmp`
- `poststep` 触发时，排除 `dPmp`

含义是：

- 电流接近 0 时，普通电流和功率下降已经没有独立诊断意义
- 台阶本身就会导致功率异常，因此 `dPmp` 不再单独作为根因证据

### 5.2 优先级规则

当前规则：

```text
Rs 和 Rsh 同时异常时，保留 Rsh
```

这表示并联电阻下降在当前知识图谱中被视为更根本的表象。

### 5.3 警告规则

警告规则不会删除表象，只会生成 `warnings`。

例如：

- `dIsc` 异常但 `dPmp` 正常
- `prestep` 异常但 `dImp` 正常

这些情况表示表象之间存在弱矛盾，后续人工分析时需要注意。

## 6. 第三层：表象到根因的加权投票

表象修正后，`DiagnosisEngine.forward_inference(...)` 会把异常表象映射到候选根因。

映射关系来自 `knowledge_graph.yaml` 的 `edges`。

每条边表示：

```text
某个异常表象 -> 某个可能根因
```

并带有一个 `weight`。

投票公式是：

```text
cause_score += edge.weight * phenomenon.deviation
```

例如，如果 `dVoc` 异常，知识图谱里可能有这些边：

- `dVoc -> 旁路二极管短路`
- `dVoc -> PID效应`
- `dVoc -> 温度测量偏差`
- `dVoc -> 少接组件`
- `dVoc -> 热斑效应`

如果 `Rsh` 也异常，又会继续给这些根因投票：

- `Rsh -> PID效应`
- `Rsh -> 局部漏电`
- `Rsh -> 灰尘污垢导电`

最终某个根因得分越高，说明它被更多、更强、更高权重的表象支持。

## 7. 第四层：反向验证

正向投票只说明“哪些根因被当前表象支持”，还不能说明这个根因是否完整解释了现象。

因此 `DiagnosisEngine.backward_verification(...)` 会继续检查候选根因的期望表象。

期望表象定义在 `knowledge_graph.yaml` 的 `expected_phenomena` 中。

每个根因可以定义：

- `required`：必须出现的表象
- `expected`：应该出现的表象
- `optional`：可选表象

反向验证会计算：

```text
coverage = 命中的 required+expected 数量 / required+expected 总数量
```

如果 required 表象缺失，置信度会被打折：

```text
confidence = coverage
如果 missed_required 非空，则 confidence *= 0.5
```

这一步的作用是给候选根因做合理性校验。

注意：当前排序主要仍由前面的加权得分驱动，`confidence` 是合理性校验，不等同于严格概率。

### 7.1 遮挡、热斑、隐裂的当前图谱口径

当前图谱把遮挡相关故障分成几类不同机制，不再把 `dIsc` 作为所有遮挡类故障的共同硬条件。

| 根因 | 主证据 | 辅助证据 | 当前推理口径 |
| --- | --- | --- | --- |
| `均匀遮挡/灰尘` | `dIsc + dImp + dPmp` | `dFF` | 典型电流型问题，要求整体短路电流、MPP 电流和功率一起下降。没有 `step` 也可以成立。 |
| `局部遮挡` | `step + dPmp` | `dImp`, `dVmp`, `dIsc` | 典型失配/旁路问题，`dIsc` 只是可选增强证据，不再是 required。 |
| `不均匀脏污` | `step + dPmp` | `dImp`, `dIsc`, `dVmp` | 更偏不规则局部瓶颈，和局部遮挡类似，但权重略低。 |
| `隐裂` | `dPmp` | `dImp`, `step`, `dIsc`, `dVmp` | 可以表现得像局部遮挡，但 `step` 只是中等支持证据，不要求一定有台阶。 |
| `热斑效应` | `dPmp + dVmp` | `dImp`, `step`, `dVoc` | I-V 单独不够特异；`step` 会加分，但不会因单纯台阶强触发热斑。 |
| `玻璃碎裂` | `dPmp` | `step`, `dIsc`, `dImp`, `dVmp` | 不是稳定单一 I-V 形态，主要作为低特异候选。 |

几个典型输入下的排序含义：

- `dIsc + dImp + dPmp`，没有 `step`：优先解释为 `均匀遮挡/灰尘`。
- `step + dPmp + dImp`，即使 `dIsc = 0`：可以优先解释为 `局部遮挡` 或 `不均匀脏污`。
- 只有 `step + dIsc`，没有 `dPmp/dImp`：`局部遮挡` 可能出现，但置信度低，并会伴随弱矛盾 warning。
- 只有 `step`：不会强判热斑或隐裂；这类结果应视为弱台阶证据。
- `step + dPmp + dVmp + dImp`：`热斑效应` 会成为高置信候选，但通常仍排在更明确的局部/不均匀遮挡之后。
- `dPmp + dVmp + dImp`，没有 `step`：`热斑效应` 也可能靠前，因为当前图谱把 `dPmp + dVmp` 设为热斑的 required 证据，`step` 是 optional。

因此，当前图谱下：

- 热斑不容易因为单纯 `step` 被误触发。
- 隐裂在 `step + dPmp + dImp` 场景中比热斑更容易作为候选出现，但通常不是第一主因。
- 如果需要更严格地区分热斑，后续应引入红外、温度或外部巡检证据，而不是只依赖 I-V 形态。

## 8. 多故障推断

`rulemap` 支持多故障推断，逻辑在 `DiagnosisEngine.detect_multi_faults(...)`。

它的做法是循环解释剩余异常表象：

1. 收集所有尚未解释的异常表象
2. 对这些表象重新做正向推理
3. 选择当前最高分的候选根因
4. 找出这个根因可以解释的表象
5. 把已解释表象从剩余集合中移除
6. 继续推断下一组故障

最多循环 5 次。

如果最后仍然有表象无法解释，会生成一个：

```text
pattern = 未解释
pattern_desc = 存在未能归因的异常表象
```

因此 `faults` 是天然支持多故障的结构化结果。

## 9. 最终根因如何取值

`rulemap` 原生输出不是单一最终标签，而是一个 `faults` 列表。

每个 fault 大致包含：

```json
{
  "fault_id": 1,
  "pattern": "电压型",
  "pattern_desc": "...",
  "phenomena": ["dVoc", "dVmp"],
  "causes": [
    {
      "cause": "旁路二极管短路",
      "score": 1.2,
      "source_phenomena": ["dVoc", "dVmp"],
      "confidence": 0.8
    }
  ],
  "confidence": 0.8
}
```

`src/rulemap/run_diagnosis.py` 对最终根因的处理方式是：

```python
causes = []
for fault in result.get("faults", []):
    for c in fault.get("causes", []):
        causes.append(c["cause"])

diagnosed_causes = causes
最终展示 = diagnosed_causes[:3]
```

也就是说，最终三个根因不是“每个故障各三个根因”，也不是只取 `faults[0].causes[0]`。

正确逻辑是：

1. 按 `faults` 的输出顺序遍历。
2. 对每个 fault，按其 `causes` 的既有顺序遍历。
3. 将所有 causes 扁平化成一个全局根因列表。
4. 取全局前三个作为最终根因。

每个 cause 的 `score` 来自 `DiagnosisEngine.forward_inference(...)`：

```text
cause_score += edge.weight * phenomenon.deviation
```

其中：

- `edge.weight` 来自 `knowledge_graph.yaml` 的表象到根因映射边
- `phenomenon.deviation` 来自表象超过阈值后的偏离程度
- 候选根因会按 `score` 从高到低排序

因此最终三个根因需要保留 `score`，否则 CSV 里会丢掉 rulemap 判断根因强弱的关键信息。

## 10. Jianheng CSV 中的 rulemap 输出字段

在 Jianheng bridge 中，为了方便 CSV 筛选，额外提取了最终字段：

- `rulemap_final_phenomena`
  - 最终异常表象的人可读列
  - 示例：`dVmp=-0.2193, dFF=-0.2462, Rs=+1.0000, poststep=+1.0000`
- `rulemap_final_causes`
  - 最终三个根因的人可读列
  - 参考 `src/rulemap/run_diagnosis.py`，由 `faults[*].causes` 扁平化后取前三个
  - 示例：`串联电阻增大(score=14.0181,conf=1.00,src=dFF|dVmp); 连接器接触不良(score=1.0000,conf=1.00,src=Rs); 线缆老化(score=0.8000,conf=1.00,src=Rs)`
- `rulemap_final_causes_json`
  - 最终三个根因的结构化 JSON
  - 每项保留 `cause`、`score`、`confidence`、`source_phenomena`、`fault_id`、`pattern`、`pattern_desc`

同时保留了三个兼容摘要字段：

- `rulemap_primary_pattern`
- `rulemap_primary_cause`
- `rulemap_primary_confidence`

这三个 `primary` 字段只表示第一故障的第一根因摘要，取值规则是：

```text
rulemap_primary_pattern = faults[0].pattern
rulemap_primary_cause = faults[0].causes[0].cause
rulemap_primary_confidence = faults[0].confidence
```

如果 `faults[0].confidence` 为空，则尝试使用 `faults[0].causes[0].confidence`。

完整多故障结果仍然保存在：

- `rulemap_faults_json`
- `rulemap_pattern_groups_json`
- `rulemap_phenomena_json`

后续分析最终根因时，应优先使用 `rulemap_final_causes` 或 `rulemap_final_causes_json`，不要只看 `rulemap_primary_cause`。

## 11. Open Circuit 的特殊情况

当前 Jianheng 接入层对开路做了特殊处理。

如果一行满足：

```text
scan_status in {2, 3}
或 diagnosis == Open_Circuit
```

则不会走普通 rulemap 推理，而是由 `src/analysis/jianheng_rulemap_bridge.py` 直接生成 override 结果：

- `rulemap_status = overridden_open_circuit`
- `rulemap_primary_pattern = open_circuit_override`
- `rulemap_primary_cause = Open_Circuit`
- `rulemap_primary_confidence = 1.0`
- `rulemap_final_phenomena = Isc_near_zero=-1.0000`
- `rulemap_final_causes = Open_Circuit(score=1.0000,conf=1.00,src=Isc_near_zero)`

这样做的原因是：

- Jianheng 流程里开路已经有明确业务规则
- `scan_status` 当前还不是 `knowledge_graph.yaml` 里的原生节点
- 直接走普通图谱推理可能因为 pairwise 指标缺失或异常而得到不稳定结论

## 12. 当前局限

当前 `rulemap` 推理需要注意这些限制：

- 它不是概率模型，分数主要来自 `edge.weight * deviation`。
- `confidence` 是期望表象覆盖率，不代表严格概率。
- `rulemap_primary_cause` 只是 `faults[0].causes[0].cause` 的摘要，不代表最终只有一个根因。
- `rulemap_reasoning` 是人类可读文本，不适合机器消费。
- 最终三个根因应优先读取 `rulemap_final_causes_json`。
- 多故障完整结构应读取 `rulemap_faults_json`。
- 当前 Jianheng 接入没有把 `voc_total_outlier`、`dVoc_total`、`step_abnormal`、`deform_abnormal`、`scan_status` 建成知识图谱原生输入。
- 当前 rulemap 阈值来自 `knowledge_graph.yaml`，不一定和 `RulePredictor4` 阈值完全一致。

## 13. 阈值分析与调参策略

### 13.1 当前阈值全景

rulemap 在不同场景下存在三套阈值配置：

| 指标 | 基准 (`knowledge_graph.yaml`) | Runtime 覆盖 (`runtime_healthy_curve_thresholds.yaml`) | RulePredictor4 (`rule_predictor4.py`) |
|------|------|------|------|
| dIsc | 0.05 | 0.15 | -0.10 |
| dVoc | 0.05 | 0.05 | -0.06 |
| dImp | 0.05 | 0.15 | -0.08 |
| dVmp | 0.05 | 0.05 | -0.04 |
| dPmp | 0.05 | 0.15 | -0.10 |
| dFF | 0.05 | 0.06 | -0.05 |
| Isc_near_zero | 0.90 | 0.90 | — |

注意：rulemap 使用 `|value| >= threshold` 且仅负值判定异常；RulePredictor4 使用 `value <= drop_limit`（signed）。两者逻辑不完全等价，但可以做方向性对比。

**配置文件生效说明**：代码中实际加载阈值的唯一文件是 `src/rulemap/knowledge_graph.yaml`（通过 `load_graph()` → `KnowledgeGraph.get_threshold()`）。同目录下的 `thresholds.yaml` 和 `phenomenon_detection.yaml` 是旧版配置，仅供参考，没有被任何代码加载。Runtime 场景通过 `runtime_healthy_curve_thresholds.yaml` 以 `threshold_overrides` 方式覆盖基准值。

### 13.2 已知问题

**问题一：基准阈值偏敏感，runtime 覆盖差距过大**

`dIsc/dImp/dPmp` 基准为 0.05，runtime 直接拉到 0.15（3 倍差距）。这说明实际数据中 0.05 会导致大量误触发，runtime 被迫大幅放宽。但基准未同步修改，意味着 Jianheng 场景（仍用基准 0.05）可能存在同样的误触发风险。

**问题二：dFF 基准已调整，但仍偏敏感**

FF 的正常波动范围在 2%~3% 是常见的（测量噪声、温度波动、插值误差）。基准已从 0.02 调整到 0.05，和 RulePredictor4 的 -0.05 基本对齐。Runtime 覆盖为 0.06。认证场景对精度要求更高，后续可能需要进一步验证 0.05 是否在认证数据上仍存在误触发。

**问题三：dVoc/dVmp 未同步放宽**

Runtime 覆盖中 `dVoc=0.05`、`dVmp=0.05` 保持不变，而 RulePredictor4 对应阈值为 -0.06 和 -0.04。这意味着在健康曲线对比场景下，电压类表象比电流类表象更容易被检测为异常，可能存在不对称性。

**问题四：两套配置维护成本**

当前 `knowledge_graph.yaml` 同时定义图谱结构和基准阈值，`runtime_healthy_curve_thresholds.yaml` 只覆盖阈值。如果后续场景增多（如 Jianheng pairwise、Runtime healthy curve、未来可能的其他场景），维护多份阈值覆盖会变得复杂。

### 13.3 调参建议

#### 短期：稳定现有行为

1. 将 `dVoc` 纳入 runtime 覆盖，考虑从 0.05 放宽到 0.08（对齐 RulePredictor4 的 -0.06 + 余量）。
2. 将 `dVmp` 纳入 runtime 覆盖，考虑从 0.05 放宽到 0.06~0.08（对齐 RulePredictor4 的 -0.04 偏保守）。
3. ~~将基准 `dFF` 从 0.02 调到 0.04~0.05~~，已完成，当前 Jianheng 基准为 0.05。
4. 认证场景精度要求高于通用诊断，后续需在认证数据集上单独验证当前阈值的误报/漏报率。

#### 中期：数据驱动批量校准

代码中已有 `Calibrator` 类（`diagnose.py`），支持：

- `calibrate_single(iv_data, known_fault)` — 单样本反推最优阈值
- `calibrate_batch(samples)` — 批量校准，取最宽松阈值
- `export_site_override(result, path)` — 导出 YAML 覆盖配置

建议流程：

1. 从增强数据集构造已知标签样本：`[{"data": {...}, "fault": "已知标签"}, ...]`
2. 跑 `Calibrator().calibrate_batch(samples)`，获取每个指标的建议阈值
3. 和当前阈值对比，找出系统性偏差
4. 导出为 `site_override.yaml` 做 A/B 对比

#### 长期：多场景阈值 Profile

建议将阈值体系改为 profile 制：

```yaml
# knowledge_graph.yaml 只定义图谱结构（节点、边、期望表象、矛盾规则）
# 阈值按场景 profile 独立管理

profiles:
  jianheng_pairwise:
    description: "鉴衡认证 pairwise 对比场景"
    dIsc: 0.05    # pairwise 对比，偏差天然小
    dVoc: 0.05
    dImp: 0.05
    dVmp: 0.05
    dPmp: 0.05
    dFF: 0.05    # 认证精度要求高，已从 0.02 调整到 0.05
    Isc_near_zero: 0.90
  runtime_healthy_curve:
    description: "Runtime 健康曲线对比场景"
    dIsc: 0.15    # 健康曲线对比，偏差天然大
    dVoc: 0.08
    dImp: 0.15
    dVmp: 0.08
    dPmp: 0.15
    dFF: 0.06
    Isc_near_zero: 0.90
```

这样不需要维护多份 `knowledge_graph.yaml`，只需在加载时指定 profile 即可。

## 14. 后续建议

如果后续要让 rulemap 成为认证结果的主视角，建议优先做这些增强：

1. 把 `scan_status`、`voc_total_outlier`、`dVoc_total` 纳入知识图谱输入。
2. 把 `step_abnormal/deform_abnormal` 和 `prestep/poststep` 的关系明确建模。
3. 为多故障输出增加更易消费的结构化摘要字段或单独明细表。
4. 按 13.3 节方案分阶段调参：短期手动修正 → 中期 Calibrator 校准 → 长期 Profile 化。
5. 将 rulemap 摘要写入 Word 报告，避免只看报告时漏掉知识图谱结论。
