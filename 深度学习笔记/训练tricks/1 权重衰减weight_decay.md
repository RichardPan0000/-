



[权重衰减weight_decay参数从入门到精通_weight decay-CSDN博客](https://blog.csdn.net/zhaohongfei_358/article/details/129625803)

# 权重衰减weight_decay参数入门





本文内容
Weight Decay是一个正则化技术，作用是抑制模型的过拟合，以此来提高模型的泛化性。

目前网上对于Weight Decay的讲解都比较泛，都是短短的几句话，但对于其原理、实现方式大多就没有讲解清楚，本文将会逐步解释weight decay机制。

## 1.什么是权重衰减(Weight Decay)

Weight Decay是一个正则化技术，作用是抑制模型的过拟合，以此来提高模型的泛化性。

它是通过给损失函数增加模型权重L2范数的惩罚(penalty)来让模型权重不要太大，以此来减小模型的复杂度，从而抑制模型的过拟合。

看完上面那句话，可能很多人已经蒙圈了，这是在说啥。后面我会逐步进行解释，将会逐步回答以下问题：

1. 什么是正则化？
2. Weight Decay的减小模型参数的思想
3. L1范数惩罚项和L2范数惩罚项是什么？
4. 为什么Weight Decay参数是在优化器上，而不是在Loss上。
5. weight decay的调参技巧



## 2.什么是正则化

 正则化的目标是减小方差或是说减小数据扰动造成的影响。看下图理解此话

![img](https://i-blog.csdnimg.cn/blog_migrate/e333a15314e11e30d494f62af3919f96.png)

这幅图是随着训练次数，训练Loss和验证Loss的变化曲线。上面那条线是验证集的。很明显，这个模型出现了过拟合，因为随着训练次数的增加，训练Loss在下降，但是验证Loss却在上升。这里我们会引出三个概念：

- 方差(Variance)：刻画数据扰动所造成的影响。
- 偏差(Bias)：刻画学习算法本身的拟合能力。
- 噪声(Noise)：当前任务任何学习算法能达到的期望泛化误差的下界。也就是数据的噪声导致一定会出现的那部分误差。
  

> 通常不考虑噪声，所以偏差和噪声合并称为偏差。

### 2.1 什么是数据扰动

上面说方差是"刻画数据扰动造成的影响"，我们可以通过下面例子来理解这句话。

假设我们要预测一个y=x的模型：

![img](https://i-blog.csdnimg.cn/blog_migrate/51363a3edc1c4e780e6fa01e7a31eb6c.png)

绿色的线是真正的模型y=x，蓝色的点是训练数据，红色的线是预测出来的模型。这个**训练数据点距离真实模型的偏离程度就是数据扰动**。

如果我们使用数据扰动较小的数据，那么预测模型的结果就会和真正模型的差距较小，例如：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ce082f15f16a105b0a756fd890841e73.png)



当我们数据扰动越大，预测模型距离实际模型的差距就会越大。因此，我们减小过拟合就是让预测模型和真实模型尽可能的一致。通常有两种做法：

​	1.增加数据量和使用更好的数据。这也是最推荐的做法。

​	2.然而，通常我们很难收集到更多的数据，所以此时就需要一些正则化技术来减小"数据扰动"对模型预测带来的影响。



## 3.减小模型权重

**权重衰减（weights decay）** 就是减小模型的权重大小，而减小模型的权重大小就可以降低模型的复杂度，使模型变得平滑，进而减小过拟合。

假设我们的模型 
$$
y=w_0+w_1x+w_2x^2...+w_nx^n
$$
模型参数为
$$
W=(w_0,w_1,w_2,...,w_n)
$$
我们使用该模型根据一些训练数据点可能会学到如下的两种曲线：

![img](https://i-blog.csdnimg.cn/blog_migrate/476f14415ebf955fd1923cca5023b496.png)

很明显，蓝色的曲线显然过拟合了。如果我们观察W的话会发现，蓝色曲线的参数通常都比较大，而绿色曲线的参数通常都比较小。

上面只是直观的说一下。结论就是：**模型的权重数值越小，模型的复杂度越低。**

>该结论 可以通过实验观察出来，也可通过数学证明。（李沐说可以证明，感兴趣的同学可以搜一下）

## 4.为Loss增加惩罚项

上面说了Weight Decay目的是要让模型权重小一点（控制在某一个范围内），以此来减小模型的复杂性，从而抑制过拟合。而weight decay的具体做法就是在loss后面增加一个权重的L2范数惩罚。

### 4.1 通过公式理解Weight Decay

weight decay的具体公式就是：
$$
L=L_0+\frac{\lambda}{2}||W||^2
$$
其中$L_0$ 是原本的loss，$\lambda$是一个超参数，负责控制权重衰减的强弱。$||W||^2$为模型参数的2范数的平方。具体的，假设我们的模型有n个采纳数，即$W=[w_1,w_2,...,w_n]$则L为：

![image-20241019113154204](C:\Users\100488\AppData\Roaming\Typora\typora-user-images\image-20241019113154204.png)

### 4.2通过公式理解Weight Decay

接下来我们用图像来感受一下Weight Decay。假设我们的模型只有两个参数W1和W2，W1和W2与Loss=2有如下关系：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b0e2f4ec1364366ea0c4a1cb5e44a9a2.png)

这个绿色的椭圆表示，当W1和W2取绿色椭圆上的点时，Loss都是2.所以，当我们没有惩罚项时，对于Loss=2，取椭圆上的这些点都可以。若取到右上角的点，那么W1和W2的值就会比较大，所以我们希望W1和W2尽量往左下角靠。







## 6.weight_decay的一些trick



1. weight_decay并没有你想想中的那么好，它的效果可能只有一点点，不要太指望它。尤其是当你的模型很复杂时，权重衰退的效果可能会更小了。
2. 通常取1e-3，如果要尝试的话，一般也就是1e-2, 1e-3, 1e-4 这些选项。
3. 权重衰退通常不对bias做。但通常bias做不做权重衰退其实效果差不多，不过最好不要做。
4. weight_decay取值越大，对抑制模型的强度越大。但这并不说明越大越好，太大的话，可能会导致模型欠拟合。

> 针对第三点：对于一个二维曲线，bias只是让曲线整体上下移动，并不能减小模型的复杂度，所以通常不需要对bias做正则化。 
