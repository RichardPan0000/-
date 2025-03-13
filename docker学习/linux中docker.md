





## docker的安装

[Ubuntu下 Docker、Docker Compose 的安装教程_ubuntu安装docker compose-CSDN博客](https://blog.csdn.net/justlpf/article/details/132982953)



如果使用gpu的方式：





 参考：

[在Docker中使用GPU_docker 使用gpu-CSDN博客](https://blog.csdn.net/ytusdc/article/details/139301315)





**官方安装过程：**[Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit 1.15.0 documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1、安装过程汇总

安装过程

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
 
 
sudo apt-get update
 
sudo apt-get install -y nvidia-container-toolkit
```



2重启docker

```
sudo systemctl restart docker
```

```
sudo docker run --rm --gpus all nvidia/cuda:${根据网站查询得到} nvidia-smi
 
# 例如  --rm 退出容器以后，这个容器就被删除了，方便在临时测试使用。
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```



出现下面的显示，表示正常。

![img](https://i-blog.csdnimg.cn/blog_migrate/b0e3ee52ac0143e563abad7eb9bd9b5b.png)





## 2、docker常用加速源

https://blog.csdn.net/llc580231/article/details/139979603



vim  /etc/docker/daemon.json

```
{
  "registry-mirrors": [
    "https://register.liberx.info",
    "https://dockerpull.com",
    "https://docker.anyhub.us.kg",
    "https://dockerhub.jobcher.com",
    "https://dockerhub.icu",
    "https://docker.awsl9527.cn"
    ]
}
```



然后重启docker服务

```
# 重新加载配置文件
sudo systemctl daemon-reload
# 重新启动docker服务
sudo service docker restart
```

