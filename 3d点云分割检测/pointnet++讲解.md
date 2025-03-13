# PointNet++原理





##

## 2、最远点采样

参考：

[3D点云算法PointNet++中最远点采样源码实现及详解_最优距离点云选取-CSDN博客](https://blog.csdn.net/m0_54361461/article/details/128047232)



在PointNet++中，特征提取的采样点是通过最远点采样法得到的。其基本思想就是：

①首先随机取一个点a，然后遍历其余剩余的点，计算与a的距离，得到最远距离点b;

②遍历剩余点与a,b的距离，得到距离最远的点c；

③重复执行以上，直到取够设定点的个数。

具体可举例，下图展示的是一个batch中的计算过程，程序中8个batch同时进行的。

![](.\pic\7afee91d2a53db66d3a66e9c7c4cd0af.png)

![dc7f4c9aac937f37a4e9ed11c4a07fe7](.\pic\dc7f4c9aac937f37a4e9ed11c4a07fe7.png)



 ①随机选取中心点A，建立距离分布图distance并赋很大的值；

②分别计算A点到B、C、D、E点距离，得到A_Dis；将A_Dis与distance比较，如果Dis中存在小于distance中的值，就将其更新到distance中；然后根据新的distance图，获得最大距离值的点D；

③以D点为中心点，计算与其他点的位置，得到D_Dis；将D_Dis与distance比较，如果Dis中存在小于distance中的值，就将其更新到distance中；然后根据新的distance图，获得最大距离值的点E；

④重复以上循环，得到中心点B,C。

其程序及详解可参考源码。

``` python
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3],如batch=8,输入点N=1024，位置信息xyz=3
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]，返回值是采样后的中心点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    '''构建一个tensor，用来存放点的索引值（即第n个点）'''
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)#8*512
    '''构建一个距离矩阵表，用来存放点之间的最小距离值'''
    distance = torch.ones(B, N).to(device) * 1e10 #8*1024
    '''batch里每个样本随机初始化一个最远点的索引（每个batch里从1024个点中取一个）'''
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#type为tensor(8,)
    '''构建一个索引tensor'''
    batch_indices = torch.arange(B, dtype=torch.long).to(device)#type为tensor(8,)
    for i in range(npoint):
        centroids[:, i] = farthest #第一个采样点选随机初始化的索引
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)#得到当前采样点的坐标 B*3
        dist = torch.sum((xyz - centroid) ** 2, -1)#计算当前采样点与其他点的距离，type为tensor(8,1024)
        mask = dist < distance#选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]#将新的距离值更新到表中
        '''重新计算得到最远点索引（在更新的表中选择距离最大的那个点）'''
        farthest = torch.max(distance, -1)[1]#max函数返回值为value,index，因此取[1]值，即索引值，返回最远点索引
    return centroids
```





最远点采样的原因：

在处理点云数据时，最远点采样（Farthest Point Sampling，FPS）是一种常用的下采样方法，旨在从大量点云中选取具有代表性的子集。例如，尽管原始点云可能包含数百万个点，但通过FPS可以将其下采样至4096个点，以降低计算复杂度，同时尽可能保留点云的几何特征。

FPS的基本原理是迭代地选择距离当前已选点集最远的点，从而确保采样点在空间中均匀分布。具体步骤如下：

1. **初始化**：从点云中随机选择一个点作为初始采样点，加入采样点集。
2. **迭代选择**：对于每个未被选择的点，计算其到当前采样点集的最小距离。选择其中距离最大的点，加入采样点集。
3. **重复**：重复步骤2，直到达到所需的采样点数。

在PointNet++中，FPS用于从原始点云中选择中心点，然后在这些中心点的邻域内进行特征提取。这种方法有助于网络捕捉点云的局部和全局特征，提高对复杂几何形状的理解能力。

需要注意的是，尽管FPS能够有效地减少点云规模，但在处理非常大的点云时，计算效率可能成为瓶颈。为此，一些研究提出了改进的FPS算法，以提高采样效率。 

[arXiv](https://arxiv.org/abs/2208.08795?utm_source=chatgpt.com)



总的来说，最远点采样在点云处理，特别是像PointNet++这样的深度学习框架中，扮演着关键角色。它通过选择空间上分布均匀的点，确保下采样后的点云仍能充分代表原始数据的几何特征。