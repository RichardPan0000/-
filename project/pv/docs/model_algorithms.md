# PV预测系统 - 模型算法文档

## 1. 模型体系概览

系统采用多模型集成策略，包含传统机器学习模型和深度学习模型：

```
模型体系
├── XGBoost系列（传统ML）
│   ├── GridXGBSimpleBayes4fold-dayahead（4折CV + 贝叶斯优化）
│   ├── GridXGBSimplePCA5Fold-dayahead（5折CV + PCA特征）
│   └── GridXGBSimplePCA5Fold-intraday（日内版本）
│
├── TimeXer系列（深度学习）
│   ├── Timexer-pv-dayahead-mae-mape（主力模型，多气象源）
│   ├── Timexer-pv-dayahead-ec-mae-mape（EC单气象源版本）
│   └── Timexer-pv-dayahead_mae（MAE优化版本）
│
└── 规则模型
    └── rule（基于规则的基线模型）
```

## 2. XGBoost模型

### 2.1 模型架构 (`src/models/xgb_model.py`)

**核心类：`XGB5fold_v2`**

采用K折交叉验证 + 贝叶斯超参数优化的XGBoost回归模型。

**训练策略：**
- 按日期轮转划分K折（非随机划分，保证时间连续性）
- 每折使用时间衰减权重：`weight = 0.996^(距今天数)`
- 早停机制：`early_stopping_rounds=25`，最大轮数500

**超参数搜索空间（贝叶斯优化）：**

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| max_depth | [4, 13] | 树最大深度 |
| subsample | [0.71, 0.95] | 样本采样比例 |
| eta | [0.01, 1] | 学习率 |
| gamma | [10, 60] | 最小分裂增益 |
| colsample_bytree | [0.71, 0.94] | 特征采样比例 |
| min_child_weight | [20, 300] | 最小叶子权重 |
| reg_lambda | [10, 100] | L2正则化 |
| reg_alpha | [10, 100] | L1正则化 |

贝叶斯优化：`init_points=12, n_iter=30`

**稳定模式默认参数：**
```python
paras = {
    'colsample_bytree': 0.8256,
    'eta': 0.2079,
    'gamma': 33.49,
    'max_depth': 9,
    'min_child_weight': 100.5,
    'reg_alpha': 50.07,
    'reg_lambda': 54.37,
    'subsample': 0.8223
}
```

### 2.2 模型变体

| 模型名 | 折数 | 优化方式 | 特征集 |
|--------|------|---------|--------|
| GridXGBSimpleBayes4fold-dayahead | 4 | 贝叶斯优化 | simple_v1 |
| GridXGBSimplePCA5Fold-dayahead | 5 | 稳定参数 | simple_v3 + PCA |
| GridXGBSimplePCA5Fold-intraday | 5 | 稳定参数 | simple_v3 + PCA |

### 2.3 数据预处理

**功率归一化：**
```python
pv_scale = station_his['pv_power_generation'].max() / 20
station_his['pv_power_generation'] = station_his['pv_power_generation'] / pv_scale
```

**数据过滤：**
1. 日数据量过滤：每天数据点 > 288 × 0.6 才保留
2. 非零PV日过滤：日均PV > 0.001 才保留
3. 非零辐照日过滤：日累计辐照 > 10 才保留

### 2.4 预测输出

```python
pred_df['model_pred'][pred_df['model_pred'] < 0.2] = 0  # 低值截断
output = pred_mean * pv_scale  # 反归一化
```

## 3. TimeXer深度学习模型

### 3.1 模型架构概述

TimeXer 是一个基于 Transformer 的时序预测模型，专为光伏功率预测设计。核心创新点：

1. **Patch-based Encoder**：将长序列切分为patch进行编码
2. **Multi-scale CNN Fusion**：多尺度卷积融合PV、天气、形状特征
3. **TimeFiLM**：时间特征条件调制（Feature-wise Linear Modulation）
4. **Cross-Attention**：PV序列与外生变量（天气）的交叉注意力
5. **Decoder-side CNN Residual**：解码端天气信息残差修正

### 3.2 模型输入输出

**输入：**
- `x_enc`: PV历史数据 `[B, 2016, 1]`（7天 × 288点/天 = 2016点）
- `x_mark_enc`: 时间特征 `[B, 2016, C_time]`（month, day, hour, weekday 的 sin/cos 编码）
- `wea_enc`: 历史天气数据 `[B, 168, C_wea]`（7天 × 24小时 = 168小时）
- `wea_dec`: 未来天气数据 `[B, 35, C_wea]`（约1.5天）

**输出：**
- `dec_out`: 预测功率 `[B, 288, 1]`（未来1天，5分钟粒度）

### 3.3 模型变体对比

| 模型 | 文件 | C_wea | 特殊特征 | Checkpoint |
|------|------|-------|---------|-----------|
| TimeXer_beta | TimeXer_beta.py | 24（6源×4特征） | 无 | checkpoints_mae.pth |
| TimeXer_omega | TimeXer_omega.py | 24 | r_pm_am, r_tail, t_cog_norm | checkpoints_mae_mape_91.pth |
| TimeXer_beta (EC) | TimeXer_beta.py | 6（1源×6特征） | 无 | checkpoints_ec_mae_mape.pth |

### 3.4 TimeXer_omega 详细架构

```
输入
├── x_enc [B, 2016, 1]          # PV历史
├── x_mark_enc [B, 2016, C_time+3]  # 时间特征 + 形状特征
├── wea_enc [B, 168, C_wea]     # 历史天气（小时级）
└── wea_dec [B, 35, C_wea]      # 未来天气（小时级）

Step 0: 分离形状特征
├── shape_scalars_seq = x_mark_enc[:, :, -3:]   # [B, L, 3]
└── x_mark_enc_pure = x_mark_enc[:, :, :-3]     # [B, L, C_time]

Step 1: 天气上采样
├── wea_enc: [B, 168, Cw] → [B, 2016, Cw]  (12x线性插值)
└── wea_dec: [B, 35, Cw] → [B, 420, Cw]    (12x线性插值)

Step 2: RevIN归一化
├── x_enc: (x - mean) / (std + 1e-5)
└── wea: (wea - mean) / std

Step 3: Encoder-side Multi-scale CNN Fusion
├── 输入拼接: [PV(1) + Weather(Cw) + Shape(3)] → [B, 1+Cw+3, L]
├── 4个并行卷积分支:
│   ├── Conv1d(kernel=2, dilation=1) → GELU
│   ├── Conv1d(kernel=3, dilation=2) → GELU
│   ├── Conv1d(kernel=5, dilation=3) → GELU
│   └── Conv1d(kernel=7, dilation=4) → GELU
├── 拼接 → 1x1 Conv投影到1通道
└── 门控残差: x_enc = x_enc + sigmoid(gate) * delta

Step 4: Patch Embedding
├── patch_len = 48（可配置）
├── 切分为 2016/48 = 42 个patch
├── Linear(48 → d_model) + PositionalEmbedding
└── 添加全局token (glb_token)

Step 5: TimeFiLM 时间条件调制
├── 时间特征按patch聚合: [B, 42, C_time] → mean → [B, 42, C_time]
├── 投影: Linear(C_time → d_model)
├── 拼接全局时间token
└── FiLM调制: x = x * (1 + gamma) + beta

Step 6: Transformer Encoder
├── Self-Attention (patch序列内)
├── Cross-Attention (全局token ↔ 外生变量)
└── FFN (Conv1d)

Step 7: Flatten Head
└── Linear(d_model × (patch_num+1) → pred_len)

Step 8: Decoder-side CNN Residual
├── 输入拼接: [pred(1) + dec_wea(Cw)] → [B, 1+Cw, pred_len]
├── 4个并行卷积分支（同encoder结构）
├── 1x1 Conv投影
└── 门控残差: dec_out = dec_out + sigmoid(gate) * delta

Step 9: 反归一化
└── dec_out = dec_out * std + mean
```

### 3.5 形状特征（Shape Features）

TimeXer_omega 模型引入了3个描述PV发电曲线形状的统计特征：

| 特征 | 含义 | 计算方式 |
|------|------|---------|
| `r_pm_am` | 下午/上午功率比 | PM总功率 / AM总功率 |
| `r_tail` | 傍晚/正午功率比 | 傍晚时段功率 / 正午时段功率 |
| `t_cog_norm` | 功率重心时间（归一化） | Σ(t × P(t)) / Σ(P(t))，归一化到[0,1] |

这些特征帮助模型理解不同站点的发电曲线形态差异（如朝向、遮挡等）。

### 3.6 TimeFiLM 机制

Feature-wise Linear Modulation（FiLM）用时间信息调制序列表示：

```python
class TimeFiLM(nn.Module):
    def forward(self, x_main, t_emb):
        gamma_beta = self.net(t_emb)           # [B, S, 2D]
        gamma, beta = gamma_beta.chunk(2, -1)  # [B, S, D] each
        return x_main * (1.0 + gamma) + beta
```

作用：让模型根据时间上下文（月份、小时等）动态调整特征表示的缩放和偏移。

### 3.7 模型加载与推理

```python
# 每次调用创建独立实例（线程安全）
args = get_timexer_args(model='TimeXer_omega')
exp = Exp_Long_Term_Forecast(args)

# 从S3下载checkpoint（首次或文件不存在时）
timexer_cks = load_model_checkpoints(is_from_s3=True, model_name='...')

# 推理
preds = exp.test_online_pv(data_np=data_np, nwp_np=nwp_np, timexer_cks=timexer_cks)

# 清理
del exp, timexer_cks
gc.collect()
```

### 3.8 日间Mask机制

基于过去7天历史数据计算日间mask（`day_mask_reshape7`）：

```
对每个5分钟时间点：
  如果过去7天中 ≥ 3天 该时间点有功率 > 阈值：
    mask = 1（白天）
  否则：
    mask = 0（夜间）
```

预测结果 = 原始预测 × day_mask，确保夜间预测为0。

## 4. 最优模型选择算法

### 4.1 选择策略 `select_best_model_v2_dst_wst`

```
1. 获取过去7天各候选模型的预测历史
2. 获取过去7天的实际发电数据
3. 对每天每个模型计算误差指标
4. 综合7天表现选择最优模型
5. 处理夏令时偏移（gap_time修正）
6. 输出最优模型的预测结果
```

### 4.2 候选模型列表

**日前预测：**
```python
model_name_list = ['rule', 'Timexer-pv-dayahead-mae-mape', 'GridXGBSimplePCA5Fold-dayahead']
```

**EC预测：**
```python
pv_ec_model_name_list = ['Timexer-pv-dayahead-ec-mae-mape', 'rule']
```

**日内预测：**
```python
intraday_model_name_list = ['GridXGBSimplePCA5Fold-intraday', 'rule', 'intraday_post_opt']
```

## 5. 后处理优化

### 5.1 深度模型后处理 `day_ahead_post_opt`

基于历史预测误差模式进行自适应修正：

```
1. 获取过去N天（配置history_days_for_deep_post_opt=7）的预测和实际值
2. 计算每天的误差模式
3. 学习系统性偏差
4. 应用修正到当前预测
```

### 5.2 日内修正（EC模型）

```python
ratio = sum_real / sum_pred  # 实际/预测 累计比
ratio = np.clip(ratio, 0.1, 10.0)  # 安全裁剪
corrected_pred = station_pred * ratio  # 等比例放缩
```

## 6. 模型存储与管理

### 6.1 存储位置

| 模型类型 | 存储位置 | 路径格式 |
|---------|---------|---------|
| XGBoost | S3 | `sigen-pv-prediction/{model_name}/{station_id}` |
| TimeXer | S3 + 本地缓存 | `sigen-pv-prediction/timexer/{checkpoint_name}.pth` |

### 6.2 Checkpoint管理

```python
# S3路径映射
'Timexer-pv-dayahead_mae'       → 'sigen-pv-prediction/Timexer/beta_cks_mae.pth'
'Timexer-pv-dayahead_mae_mape'  → 'sigen-pv-prediction/timexer/checkpoints_mae_mape_91.pth'
'Timexer-pv-dayahead-ec-mae-mape' → 'sigen-pv-prediction/timexer/checkpoints_ec_mae_mape.pth'

# 本地缓存路径
'tsxer_main/checkpoints/timexer/checkpoints_mae.pth'
'tsxer_main/checkpoints/timexer/checkpoints_mae_mape_91.pth'
'tsxer_main/checkpoints/timexer_ec/checkpoints_ec_mae_mape.pth'
```

### 6.3 模型加载策略

- 首次加载时从S3下载到本地
- 后续使用本地缓存（检查文件存在且非空）
- 自动处理 `module.` 前缀（DataParallel训练的模型）
- 支持GPU/CPU自动切换
