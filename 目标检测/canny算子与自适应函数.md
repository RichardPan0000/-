

# Canny边缘检测



Canny 边缘检测算子是经典的图像处理算法之一，由 John F. Canny 在 1986 年提出，目的是在尽可能减少误报和定位误差的前提下，检测图像中的“真实边缘”。它的原理由 **五个主要步骤**组成，每一步都有明确的数学依据。

------

## 🌟 Canny 边缘检测的五个步骤

------

### 1️⃣ 高斯滤波（Gaussian Filter）——去噪

**目的**：平滑图像，减少噪声对边缘检测的干扰。

- 使用高斯核（如 5×55 \times 5）对图像进行卷积。
- 原因是边缘检测对噪声很敏感，先平滑处理可以提升鲁棒性。

```python
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
```

------

### 2️⃣ 梯度计算（Sobel 算子）——获取边缘方向和强度

**目的**：计算每个像素的梯度强度（边缘强度）和梯度方向（边缘朝向）。

- 分别用 Sobel 卷积核计算 $G_x$ 和 $G_y$：
  $$
  G_x = \frac{\partial I}{\partial x}, \quad G_y = \frac{\partial I}{\partial y}
  $$
  

- 计算边缘强度：
  $$
  G = \sqrt{G_x^2 + G_y^2}
  $$
  

- 计算边缘方向（角度）：

  
  $$
  \theta = \arctan2(G_y, G_x)
  $$
  

  

------

### 3️⃣ 非极大值抑制（Non-Maximum Suppression）——细化边缘

**目的**：保留真正的边缘点，抑制梯度方向上的非最大值。

- 对于每个像素，在其梯度方向上检查邻居是否更大：
  - 若不是最大值 → 抑制（设为0）
  - 若是最大值 → 保留

效果是把粗边缘变成一像素宽的细线。

------

### 4️⃣ 双阈值处理（Double Threshold）——判断强弱边缘

**目的**：区分“强边缘”、“弱边缘”和“非边缘”。

- 设置两个阈值：高阈值 $T_{high}$、低阈值 $T_{low}$
  - $G \geq T_{high}$ → 强边缘（一定保留）
  - $T_{low} \leq G < T_{high}$ → 弱边缘（待定）
  - $G < T_{low}$ → 非边缘（剔除）

------

### 5️⃣ 边缘连接（Hysteresis）——确定最终边缘

**目的**：根据强边缘连接弱边缘，过滤掉孤立噪声点。

- 所有和“强边缘”连接的“弱边缘”也认为是真边缘。
- 其余弱边缘全部舍弃。

这一步通过递归或 BFS/DFS 实现边缘“扩张”。

------

## 总结图

```
图像 → 高斯滤波 → Sobel算子 → 非极大值抑制 → 双阈值 → 边缘连接 → 最终边缘
```

------

## 🎯 特点

- 边缘检测精度高，定位准确。
- 抗噪能力好，适合自然图像。
- 缺点是：计算量大，参数（两个阈值）敏感。

------

如果你需要，我可以提供一份完整的 Python + OpenCV 示例代码来演示每一步。是否需要？



```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像（灰度）
img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 1. 高斯滤波（去噪）
blur = cv2.GaussianBlur(img, (5, 5), 1.4)

# 2. 计算梯度（Sobel）
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_direction = np.arctan2(sobely, sobelx)

# 3. Canny 边缘检测（自动包含非极大值抑制 + 双阈值 + 边缘连接）
canny_edges = cv2.Canny(blur, threshold1=50, threshold2=150)

# 画图对比
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title('Original')
plt.imshow(img, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Gaussian Blur')
plt.imshow(blur, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Gradient Magnitude')
plt.imshow(gradient_magnitude, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Sobel X')
plt.imshow(sobelx, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('Sobel Y')
plt.imshow(sobely, cmap='gray')

plt.subplot(2, 3, 6)
plt.title('Canny Result')
plt.imshow(canny_edges, cmap='gray')

plt.tight_layout()
plt.show()

```



![](D:\pfh工作\学习笔记\-\目标检测\pic\canny算子边缘检测过程.png)







# 自适应函数

这段代码使用了 OpenCV 的 `cv2.adaptiveThreshold` 自适应阈值函数，其主要作用是**将灰度图转换成二值图像**，并且能**适应光照不均匀的场景**，非常适合处理如文档、条码、边缘、图形检测等任务。

------

### 👇 一行一行解释：

```python
processed = cv2.adaptiveThreshold(
    blurred,                  # 输入图像（必须是灰度图）
    255,                      # 最大值（即白色）
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 使用高斯加权的方式计算局部阈值
    cv2.THRESH_BINARY_INV,    # 小于局部阈值为白（255），否则为黑（0）
    11,                       # 邻域块大小（11x11），必须是奇数
    2                         # 常量C，局部阈值 = 加权平均 - C
)
```

------

### 🧠 原理说明：

- `adaptiveThreshold` 是一种**局部阈值处理方法**，不像 `cv2.threshold` 那样用一个全局值。
- 每个像素点的阈值由它**邻域内的像素值加权平均**决定。
- 可选的模式：
  - `cv2.ADAPTIVE_THRESH_MEAN_C`: 使用邻域均值
  - `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`: 使用邻域高斯加权均值（更平滑、精细）

------

### 🆚 `cv2.threshold` 对比：

| 方法                | 全局阈值 | 适应性强 | 光照不均也有效 | 应用场景                     |
| ------------------- | -------- | -------- | -------------- | ---------------------------- |
| `threshold`         | ❌        | ❌        | ❌              | 光照均匀的图像               |
| `adaptiveThreshold` | ✅        | ✅        | ✅              | 文档扫描、表格、条码、边缘图 |

------

### 📌 示例：

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
binary = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)
```

结果是清晰的黑白图（文字或结构为白，其它为黑），**非常适合后续轮廓查找、形态学处理、OCR等操作**。

如果你有图像示例，我也可以帮你调参优化这个效果。



下面是一个完整的 Python 示例代码，展示了 `cv2.adaptiveThreshold` 的作用。这个示例会从一张图片读取内容，先灰度化、再高斯模糊，然后应用自适应阈值，最后显示原图与处理结果对比：

```python
import cv2
import matplotlib.pyplot as plt

# 读取图像（以灰度形式）
img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 高斯模糊，去除噪声
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 自适应阈值处理
processed = cv2.adaptiveThreshold(
    blurred, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 
    11, 
    2
)

# 使用 matplotlib 显示原图和处理图像
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("原图（灰度）")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("高斯模糊")
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("自适应阈值")
plt.imshow(processed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

```

### 📌 效果说明：

- **原图**：原始的灰度图像。
- **高斯模糊**：平滑图像，减少噪声影响。
- **自适应阈值图像**：图像变成黑白二值，结构清晰，便于后续分析（如边缘检测、轮廓提取等）。



![](D:\pfh工作\学习笔记\-\目标检测\pic\cv2自适应 adaptive.png)





# 霍夫变换



```python
cv2.HoughLinesP(
    edges, 1, np.pi/180, 
    threshold=hough_threshold, 
    minLineLength=min_line_length, 
    maxLineGap=max_line_gap
)
```



maxLineGap :





### 工作原理

1. 线段连接机制：

- 当算法在图像中找到若干个潜在的线段点后，它会尝试将这些点连成线段

- 如果两个点之间的距离小于或等于max_line_gap，算法会认为它们属于同一条线段

- 算法会自动将这些点连接起来，形成一条连续的线段

1. 处理间断线段：

- 在实际图像中，由于噪声、光照变化或物体遮挡，线段常常不是完全连续的

- max_line_gap参数允许算法容忍这些小间隙，仍然将它们识别为同一条线段

### 参数值的影响

1. 较小的max_line_gap值（例如5-10像素）：

- 更严格的线段连接条件

- 倾向于将轻微间断的线段分开识别

- 结果中会有更多的短线段

- 适合需要精确识别独立线段的场景

1. 较大的max_line_gap值（例如30-50像素）：

- 更宽松的线段连接条件

- 会连接较远的线段点

- 能够跨越较大的间隙，检测到更长的线段

- 适合处理有噪声或间断的线条图像

- 但可能错误地连接不属于同一直线的点







## HoughLinesP 的整体流程

1. **边缘检测预处理**

   - 输入图需先经过边缘检测（通常用 Canny）来提取二值“边缘图” ([OpenCV 文档](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html?utm_source=chatgpt.com)).

2. **参数空间映射**

   - 对图中每个边缘点 (x,y)(x,y)，它对应到一组 (ρ,θ)(\rho,\theta) 参数：

     ρ=xcos⁡θ+ysin⁡θ,θ∈[0,π)  \rho = x\cos\theta + y\sin\theta,  \quad \theta\in[0,\pi)

   - 在极坐标累加器（二维数组）中，每个 (ρ,θ)(\rho,\theta) 单元记一票 ([OpenCV 文档](https://docs.opencv.org/4.x/d3/de6/tutorial_js_houghlines.html?utm_source=chatgpt.com)).

3. **概率采样**

   - 与标准 Hough 不同，`HoughLinesP` 对边缘点做随机或网格化采样，只对部分点进行投票，以加速计算 ([Paperspace by DigitalOcean Blog](https://blog.paperspace.com/understanding-hough-transform-lane-detection/?utm_source=chatgpt.com)).

4. **阈值过滤**

   - 只保留累加器中票数 ≥ `threshold` 的 (ρ,θ)(\rho,\theta) 参数对，这里 `threshold` 控制“足够多投票”才认为存在线段 ([Stack Overflow](https://stackoverflow.com/questions/24922897/houghlinesp-parameters-threshold-and-minlinelength?utm_source=chatgpt.com)).

5. **线段生成**

   - 将满足阈值的参数对转换回图像空间，并在这些投票对上寻找连续的像素段。
   - 只返回那些长度 ≥ `minLineLength` 的线段，且允许段与段间的最大间隙 ≤ `maxLineGap` 时将其合并为一条线 ([Medium](https://medium.com/@elvenkim1/how-to-use-houghlines-with-robot-b8ac9d1554b3?utm_source=chatgpt.com)).

------

## 参数详解

### 1. `threshold`：累加器阈值

- **意义**：在参数空间中，某一 (ρ,θ)(\rho,\theta) 单元所获投票数必须 ≥ `threshold` 才会被视为一条候选直线。
- **作用**：控制线检测的**置信度**；阈值越高，检测到的线越少且越可靠；阈值过低会产生大量虚假线 ([Stack Overflow](https://stackoverflow.com/questions/24922897/houghlinesp-parameters-threshold-and-minlinelength?utm_source=chatgpt.com)).

### 2. `minLineLength`：最小线段长度

- **意义**：仅返回像素端点距离 ≥ `minLineLength` 的线段。
- **作用**：过滤掉过短的、可能由噪声或零散边缘组成的线段；与 `threshold` 不同，它直接以线段的**几何长度**为过滤条件 ([Stack Overflow](https://stackoverflow.com/questions/24922897/houghlinesp-parameters-threshold-and-minlinelength?utm_source=chatgpt.com)).

### 3. `maxLineGap`：最大间隙

- **意义**：在线段生成时，允许两个线段间的间隔（以像素计）最长为 `maxLineGap`；若间隔小于该值，将它们视为同一条线。
- **作用**：弥补微小断裂，使同一条线的被检测线段连贯；`maxLineGap` 越大，越容易将分段合并，但可能误将相近平行段连接 ([Signal Processing Stack Exchange](https://dsp.stackexchange.com/questions/10467/influence-of-image-size-to-edge-detection-in-opencv?utm_source=chatgpt.com)).

------

## 实践建议

- **调参顺序**：
  1. 初始设定 `threshold`，大致排除大多数抖动噪声；
  2. 根据线段长度分布调整 `minLineLength`；
  3. 最后微调 `maxLineGap` 以连接小间隙 ([Medium](https://medium.com/@elvenkim1/how-to-use-houghlines-with-robot-b8ac9d1554b3?utm_source=chatgpt.com)).
- **分辨率注意**：
  - 若输入图缩放，多条短线可能分布密集，此时 `minLineLength` 要相应缩小。
  - `rho` 和 `theta` 可保持默认 (1, π/180)(1,\ \pi/180)，特殊场景可加粗采样步长以提速 ([Signal Processing Stack Exchange](https://dsp.stackexchange.com/questions/10467/influence-of-image-size-to-edge-detection-in-opencv?utm_source=chatgpt.com)).

------

通过理解“投票→阈值过滤→线段重构”这一核心流程，以及三个参数在不同阶段的作用，即可在各类**车道检测**、**建筑物边缘提取**等任务中灵活运用 `HoughLinesP`。