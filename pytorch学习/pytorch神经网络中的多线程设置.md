https://www.cnblogs.com/xyz/p/15854307.html



## [pytorch中神经网络的多线程数设置：torch.set_num_threads(N)](https://www.cnblogs.com/xyz/p/15854307.html)

实验室的同学一直都是在服务器上既用CPU训练神经网络也有使用GPU的，最近才发现原来在pytorch中可以通过设置 **torch.set_num_threads(args.thread)** 来限制CPU上进行深度学习训练的线程数。

 

 

 

**torch.set_num_threads(args.thread)**  在使用时的一个注意事项就是如果不设置则默认使用物理CPU核心数的线程进行训练，而往往默认设置是可以保证运算效率最高的，因此该设置线程数是需要小于物理CPU核心数的，否则会造成效率下降。

 

既然默认设置既可以保证最高的运算效率那么这个设置的意义在哪呢，这个设置的意义就是在多人使用计算资源时限制你个人的改应用的计算资源占用情况，否则很可能你一个进程跑起来开了太多的线程直接把CPU占用率搞到50%或者直接奔100%去了。

 

总的说，该设置是为了在多人共享计算资源的时候防止一个进程抢占过高CPU使用率的。

 

 

 

给一个自己的设置代码：（实现了pytorch的最大可能性的确定性可复现性，并设置训练、推理时最大的线程数）



``` python
# pytorch的运行设备
device = None


def context_config(args):
    global device

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(args.thread)  # 设置pytorch并行线程数
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')
```



 

**结果：**

**发现如果线程设置过多，超过CPU的物理线程数运行效率不仅没有提升反而下降，正常默认设置即可。**