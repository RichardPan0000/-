# 3d gaussian-splatting



## 参考：

[ https://blog.csdn.net/L_IRISRIIN/article/details/135482945 ](https://blog.csdn.net/L_IRISRIIN/article/details/135482945)



## 一些实现

1、[ 3D gaussian splatting 高斯的使用方法及训练 - 简书 ](https://www.jianshu.com/p/0c249855eb38)



[ 3DGS（3D Guassian Splatting）部署验证+个人数据训练 - 知乎 ](https://zhuanlan.zhihu.com/p/685698909)

可以用高版本的cuda进行训练。







[ Windows下3D Gaussian Splatting从0开始安装配置环境及训练教程_3d gaussian splatting安装教程-CSDN博客 ](https://blog.csdn.net/weixin_64588173/article/details/138140240)





[ 3D-gaussian-splatting运行（免环境配置）_3d gaussian splatting3070-CSDN博客 ](https://blog.csdn.net/L_IRISRIIN/article/details/135482945)





2、linux下的实现

[ 【3DGS】Ubuntu20.04系统搭建3D Gaussian Splatting及可视化环境_3d gaussian splatting 环境搭建-CSDN博客 ](https://blog.csdn.net/weixin_48400654/article/details/137271501)

这个有数据集 [ 3D Gaussian Splatting for Real-Time Radiance Field Rendering ](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)



[ Ubuntu20.04 3DGS复现全流程-CSDN博客 ](https://blog.csdn.net/abbbbbk/article/details/140886831)



[ 3D Gaussian Splatting Linux端部署指南（含Linux可视化）_3d gaussian splatting 环境搭建-CSDN博客 ](https://blog.csdn.net/yjboxes/article/details/136213935)

这个应该比较全。





[ Gaussian Splatting 在ubuntu 环境下部署 - 知乎 ](https://zhuanlan.zhihu.com/p/688557162)

这个有colmap的安装



## 使用步骤

1、使用convert.py，通过colmap来建立稀疏点云。

对应图片的位置



```
python convert -s data_dir
```

﻿

注意：

图片文件夹下一定要是一种路径方式，否则报错。就是data_dir 下要有input文件夹。



```
-- data_dir:
    --input
        --1.jpg
        --2.jpg
        --3.jpg
```



2、进行训练重建稠密点云：

安装好所有环境的基础上

```
python train.py -s file/tandt/truck
```

这个truck 有colmap 转换好的点云文件和image文件

```
--truck
    --images
        --1.jpg
        --2.jpg
    --sparse
        --0
            --cameras.bin
            --images.bin
            --points3D.bin
            --points3D.ply
            --project.ini
    
```

其中spare是稀疏点云的文件夹。



## 一些issue

[ https://blog.csdn.net/woyaomaishu2/article/details/139353111 ](https://blog.csdn.net/woyaomaishu2/article/details/139353111)



colmap使用命令行出现的一些错误问题：

[ https://github.com/colmap/colmap/issues/1305 ](https://github.com/colmap/colmap/issues/1305)



## web渲染

# 使用web进行渲染



﻿

﻿

three.js进行渲染：

﻿[ https://github.com/mkkellogg/GaussianSplats3D?tab=readme-ov-file ](https://github.com/mkkellogg/GaussianSplats3D?tab=readme-ov-file)﻿

​     three.js 好像是封装了webgl

﻿

这个稍微更逼真一点点。



使用webgl进行渲染

﻿[ https://github.com/antimatter15/splat ](https://github.com/antimatter15/splat)﻿