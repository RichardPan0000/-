# Jianheng IV Certification Performance Profiling

本文档说明如何查看 `run_jianheng_iv_certification.py` 的阶段耗时、CPU 与内存占用，并进一步定位 `healthy_curve_flow` 内部慢点。

## 开启方式

默认开启 profiling：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 src\run_jianheng_iv_certification.py live --station-id 82026011900040 --sn-code 120A13AT0220 --task-id 26321
```

关闭 profiling：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 src\run_jianheng_iv_certification.py live --no-profile
```

调整内存采样和终端 Top 输出：

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 src\run_jianheng_iv_certification.py live --profile-sample-interval 0.2 --profile-print-top 20
```

## 输出文件

profiling 开启后，会在本次鉴衡输出目录下生成：

| 文件 | 内容 |
|---|---|
| `performance_profile.json` | 完整性能 profile，包含环境、总耗时、慢阶段摘要和所有 stage |
| `performance_stages.csv` | 每个 stage 一行，适合用 Excel / pandas 分析 |
| `performance_top_stages.csv` | 按 `wall_seconds` 倒排的慢阶段 |
| `summary.json` | 增加 performance 文件路径、总耗时、CPU、峰值 RSS 和慢阶段摘要 |

终端也会打印：

| 字段 | 含义 |
|---|---|
| `performance_profile_path` | 完整 JSON 路径 |
| `performance_stages_csv_path` | 明细 CSV 路径 |
| `performance_top_stages_csv_path` | 慢阶段 CSV 路径 |
| `perf_top` | 终端 Top 慢阶段摘要 |

## CSV 字段说明

| 字段 | 含义 |
|---|---|
| `stage` | 阶段名 |
| `parent_stage` | 父阶段名 |
| `depth` | 嵌套深度 |
| `wall_seconds` | 实际耗时 |
| `cpu_seconds` | 当前进程 CPU 时间 |
| `cpu_percent_one_core` | 相对单核 CPU 的占比 |
| `cpu_percent_all_cores` | 按机器 CPU 核数归一后的占比 |
| `rss_start_mb` | 阶段开始 RSS 内存 |
| `rss_end_mb` | 阶段结束 RSS 内存 |
| `rss_delta_mb` | 阶段 RSS 变化 |
| `rss_peak_mb` | 阶段内采样到的 RSS 峰值 |
| `metadata` | JSON 字符串，包含 task、station、MPPT、string 等上下文 |

## 外层阶段

| stage | 含义 |
|---|---|
| `jianheng.cli.total` | CLI 总耗时 |
| `jianheng.csv.read_processed` | 读取 processed CSV |
| `jianheng.csv.read_labels` | 读取 labels CSV |
| `jianheng.csv.filter_inputs` | task / inverter / timestamp 筛选 |
| `jianheng.csv.build_batches` | 构造诊断 batch |
| `jianheng.task.total` | 单个 task 总耗时 |
| `jianheng.batch.analyze_station` | 单个 batch 诊断 |
| `jianheng.analyzer.merge_labels` | 合并标签元数据 |
| `jianheng.analyzer.orchestrator_diagnosis` | 调用 diagnosis orchestrator |
| `jianheng.analyzer.prepare_pairs` | 构造双串对比 pair |
| `jianheng.analyzer.select_reference` | 选择参考组串 |
| `jianheng.analyzer.write_processed_curves` | 写 processed IV 曲线 |
| `jianheng.analyzer.plot_figures` | 生成 MPPT 报告图 |
| `jianheng.analyzer.build_rows` | 构造结果行 |
| `jianheng.output.rulemap_enrich` | rulemap 增强输出 |
| `jianheng.output.write_results_csv` | 写结果 CSV |
| `jianheng.output.generate_word_report` | 生成 Word 报告 |
| `jianheng.output.write_summary` | 写 summary |
| `jianheng.live.load_snapshot` | live 模式查询并解析 IV 数据 |

## Flow 内部阶段

`healthy_curve_flow` 内部会输出这些阶段：

| stage | 含义 |
|---|---|
| `orchestrator.diagnose_station` | orchestrator 调用 flow 的总耗时 |
| `flow.healthy_curve.total` | healthy curve flow 总耗时 |
| `flow.mppt.total` | 单个 MPPT 总耗时 |
| `flow.mppt.filter_normal_strings` | 过滤正常扫描组串 |
| `flow.mppt.group_strings` | 组串分组 |
| `flow.group.total` | 单个 module group 总耗时 |
| `flow.group.get_reference_portrait` | 读取参考串画像 |
| `flow.group.resolve_reference_n_components` | 解析 / 估算参考串组件数 |
| `flow.group.fetch_ecmwf_temperature` | 查询 ECMWF 温度 |
| `flow.group.estimate_environment` | 估算辐照度和温度 |
| `flow.group.resolve_voc_reference` | 计算修正后的单组件 Voc |
| `flow.group.generate_healthy_curve` | 生成健康 STC 曲线 |
| `flow.group.map_healthy_curve` | 映射健康曲线到当前工况 |
| `flow.group.calculate_healthy_reference` | 计算健康参考特征 |
| `flow.group.collect_group_statistics` | 组内统计 |
| `flow.string.total` | 单条组串诊断总耗时 |
| `flow.string.get_portrait` | 读取单串画像 |
| `flow.string.resolve_n_components` | 解析 / 估算单串组件数 |
| `flow.string.interpolate_curve` | IV 曲线插值 |
| `flow.string.extract_features` | 特征提取 |
| `flow.string.step_aware_slope_features` | 台阶相关 slope 特征 |
| `flow.string.build_rule_comparison` | 构造 rule comparison metrics |
| `flow.string.rule_predictor_predict` | RulePredictor 预测 |
| `flow.string.build_result` | 构造诊断结果 |
| `flow.n_components.update_portrait_writeback` | 组件数估算后的画像写回 |

## 分析建议

| 现象 | 优先查看 |
|---|---|
| 总体很慢 | `performance_top_stages.csv` 的 `wall_seconds` 倒排 |
| CPU 高 | `cpu_percent_one_core` 和 `cpu_seconds` |
| 内存高 | `rss_peak_mb` 和 `rss_delta_mb` |
| 某个 MPPT 慢 | 过滤 `metadata` 中的 `mppt_id` |
| 某条组串慢 | 过滤 `metadata` 中的 `string_id` |
| rulemap 慢 | 查看 `jianheng.output.rulemap_enrich` |
| Word 报告慢 | 查看 `jianheng.output.generate_word_report` |
| diagnosis flow 慢 | 查看 `flow.*` 阶段 |
