https://blog.csdn.net/weixin_45144684/article/details/138455633

![](pic\yolov8结构图.png)





https://blog.csdn.net/java1314777/article/details/142665078









[【YOLOv8模型网络结构图理解】-CSDN博客](https://blog.csdn.net/dally2/article/details/136654811)



对比C3模块和C2f模块，可以看到C2f获得了更多的梯度流信息

![](pic\c2f.png)

```python
class C2f(nn.Module):
    
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) # 最左边的CBS模块
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2) 最右边的CBS模块
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)) # 接上了n个Bottleneck模块

    def forward(self, x):
    	# tensor.chunk(chunk数，维度)
        y = list(self.cv1(x).chunk(2, 1)) #先将输入特征图cv1卷积，然后chunk分2块
        y.extend(m(y[-1]) for m in self.m) #表示被切分的最后一块，即第二块，把第二块放进n个连续的Bottleneck里，加到y列表的尾部，y就变成了2+n块
        return self.cv2(torch.cat(y, 1)) #将y按第一维度拼接在一起，然后进行cv2卷积操作。

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


```



