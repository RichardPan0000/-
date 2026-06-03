# 阶梯检测逻辑文档

## 概述

本文档记录 `LegacyPatternDetector` 中的阶梯检测算法逻辑，用于检测光伏 IV 曲线中的阶梯特征，从而识别遮挡/破损故障。

**相关文件**:
- `src/diagnosis/pattern_detectors/legacy_pattern_detector.py`
- `src/diagnosis/predictors/rule_predictor2.py`

---

## 整体调用链

```
LegacyPatternDetector.analyze()
    ├── RulePredictor2._detect_steps()        # 核心阶梯检测
    ├── mild_shading_features_simple()        # 低压段畸形检测
    └── analyze_high_segment_simple()        # 高压段畸形检测
```

---

## 核心方法 `_detect_steps()` 详解

### 步骤1：中值滤波平滑

```python
I_s = medfilt(I, kernel_size=11)  # 去除噪声
```

**目的**: 去除电流信号中的高频噪声，避免噪声被误判为阶梯。

### 步骤2：计算负梯度

```python
dI = np.gradient(I_s, Vgrid)  # dI/dV
neg_grad = -dI                  # 负梯度（下降时为正）
```

**物理意义**: 
- 正常 IV 曲线电流随电压单调递减
- 阶梯处会有明显的"陡降"，对应 `neg_grad` 出现峰值

### 步骤3：动态阈值计算（MAD方法）

```python
med = np.median(neg_grad)
mad = np.median(np.abs(neg_grad - med)) + 1e-9
threshold = med + slope_thresh_mul * mad   # 默认 3.5 倍MAD
```

**MAD (Median Absolute Deviation)** 是一种鲁棒的统计量，用于识别异常值。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `med` | 中位数 | 梯度分布的中心 |
| `mad` | 绝对中位差 | 梯度的离散程度 |
| `slope_thresh_mul` | 3.5 | 阈值倍数 |

### 步骤4：找出显著下降段

```python
candidates = neg_grad > threshold  # 超过阈值的点

# 合并连续点为段
segments = []
i = 0
while i < N:
    if candidates[i]:
        j = i
        while j + 1 < N and candidates[j + 1]:
            j += 1
        segments.append((i, j))
        i = j + 1
    else:
        i += 1
```

### 步骤5：过滤有效阶梯

```python
for (start, end) in segments:
    idx_l = max(0, start - 2)
    idx_r = min(len(I) - 1, end + 2)
    depth = I_s[idx_l] - I_s[idx_r]  # 计算下降深度
    
    if depth > Isc * 0.05:            # 深度超过 Isc 的 5%
        step_depths.append(depth)
        step_locs.append(loc_v)

num_steps = len(step_depths)
```

---

## 判断标准

| 条件 | 含义 | 判定结果 |
|------|------|----------|
| `num_steps >= 2` | 检测到2个或以上有效阶梯 | **break_shadow** |
| `depth > Isc * 0.05` | 阶梯深度超过短路电流的5% | 有效阶梯 |

---

## 可视化理解

```
电流 I
│
│    ┌──── 阶梯1 (深度 d1)
│    │
│    │   ┌──── 阶梯2 (深度 d2)  
│    │   │
│────┴───┴──── 电压 V

当 d1 > Isc*5% 且 d2 > Isc*5% 时，num_steps=2 → break_shadow
```

---

## 辅助畸形检测

### 低压段畸形检测 `mild_shading_features_simple()`

**检测区域**: 0 ~ 0.9 * Vmp

**算法**:
1. 对低压段电流进行 Savitzky-Golay 滤波平滑
2. 二阶多项式拟合
3. 计算残差能量 `res_energy = mean((I_fit - I_smooth)²)`

**判定**:
```python
if res_energy > 1e-5:
    return "break_shadow"
```

### 高压段畸形检测 `analyze_high_segment_simple()`

**检测区域**: Vmp * 1.05 ~ Voc

**算法**:
1. 线性拟合作为包络线
2. 计算三个指标:
   - `concave_area_norm`: 凹陷面积（归一化）
   - `mono_area_norm`: 单调性破坏面积（归一化）
   - `pv_right_ratio`: 高压段功率占比

**综合评分**:
```python
score = concave_area_norm * 100.0 + mono_area_norm * 100.0 + (1.0 - pv_right_ratio) * 80.0
```

**注意**: 当前代码中高压段检测**未启用**（被注释掉）。

---

## 关键阈值汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `medfilt_kernel` | 11 | 中值滤波窗口大小 |
| `slope_thresh_mul` | 3.5 | 梯度阈值倍数 |
| `step_depth_threshold` | Isc * 0.05 | 阶梯深度阈值 |
| `num_steps_limit` | 2 | 阶梯数量阈值 |
| `deform_low_res_energy` | 1e-5 | 低压段残差能量阈值 |
| `deform_high_score` | 200.0 | 高压段综合评分阈值（未启用） |

---

## 返回结果结构

```python
{
    "steps": {
        "num_steps": 2,              # 阶梯数量
        "max_step_depth": 0.15,      # 最大阶梯深度
        "step_depths": [0.12, 0.15], # 各阶梯深度列表
        "step_locs": [0.35, 0.68]     # 各阶梯位置（归一化电压）
    },
    "deformation": {
        "low_voltage": {"res_energy": 2.5e-6},
        "high_voltage": {"score": 152.0, ...}
    },
    "step_abnormal": True,           # 阶梯异常
    "deform_abnormal": False,        # 畸形异常
    "reason_candidates": ["steps=2"]
}
```

---

## 更新历史

| 日期 | 作者 | 说明 |
|------|------|------|
| 2026-03-31 | - | 初始文档创建 |