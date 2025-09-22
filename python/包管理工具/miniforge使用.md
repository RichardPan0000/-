# miniforge 使用注意事项





miniforge 是anaconda的平替工具，anaconda是商业软件，可个人用，团队用不行。



miniforge 在vscode 和cursor中能直接指定解释器，但是在pycharm中支持不好，指定解释器位置时候，python3.8的是识别不了的。而base环境的python.exe 是可以指定的，虚拟环境env中，python3.8的识别不了。而env中python3.13 直接用会出现以下问题，装了setuptools 之后可识别。



![](D:\pfh工作\学习笔记\-\python\包管理工具\pic\pycharm导入包.jpg)





![](D:\pfh工作\学习笔记\-\python\包管理工具\pic\运行管理.jpg)



如果 解释器错误，需要在运行管理，这里，调整调整解释。
