

[爆肝5万字❤️Open3D 点云数据处理基础（Python版）_open3d中文文档-CSDN博客](https://blog.csdn.net/weixin_46098577/article/details/120167360)



https://www.open3d.org/docs/latest/tutorial/core/tensor.html#Tensor-creation

官网教程





#### Access estimated vertex normal (访问估计的顶点法线)

可以从downpcd的法线变量中检索估计的法线变量。

```
print('Print a normal vector of the 0th point')
print(downpcd.normals[0])
```

要查看其他变量，请使用help（downpcd）。发现向量可以被转换成一个numpy 数组，使用np.asarray。

```
print('Print the normal vector of the first 10 points')
print(np.asarray(downpcd.normals)[:10,:])
```

![](D:\潘方宏工作\学习笔记\open3d学习\pic\image-202410218646.png)

#### crop point cloud （裁剪 点云）

```
![image-202410218646](D:\潘方宏工作\学习笔记\open3d学习\pic\image-202410218646.png)print("Load a polygon volume and use it to crop the original point cloud")
demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
# 读取用于裁剪点云的多边形体积，路径由demo_crop_data.cropped_json_path提供
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
# 使用多边形裁剪点云，返回裁剪后的点云（在这里是‘椅子’）
chair = vol.crop_point_cloud(pcd)

# 可视化裁剪后端额点云
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])
```

这段代码的目的是加载一个点云数据，并使用一个多边形体积来裁剪（剪切）点云，以获取所需的部分。

![image-20241021101616509](D:\潘方宏工作\学习笔记\open3d学习\pic\image-20241021101616509.png)



read_selection_polygon_volume 读取指定多边形选择区域的 json 文件。vol.crop_point_cloud(pcd) 过滤掉点。只留下椅子。

#### Paint point cloud



```
print("Paint chair")
chair.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])
```

![image-20241021102429756](C:\Users\100488\AppData\Roaming\Typora\typora-user-images\image-20241021102429756.png)

paint_uniform_color 将所有点涂成统一的颜色。颜色在 RGB 空间中，范围为 [0, 1]。