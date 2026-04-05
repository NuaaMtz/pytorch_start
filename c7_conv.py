# 卷积操作：卷积核滑动，对应元素相乘再相加

# 定义图片
from turtle import forward
from numpy import dtype, float32
import torch
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 定义卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])


# 添加通道和 batchsize
## 添加通道很好理解，图片都是单通道或者三通道数据
## 为什么要添加batch维度？
## 模型当做是一个公式，想象中的梯度更新应该是一个输入更新一次
## 梯度，而是一批数据同时对同一个模型计算，然后计算平均/或其他梯度

input=torch.reshape(input,(1,1,5,5))# (batch,channel,h,w)
## 每个通道的卷积核是一样的
kernel=torch.reshape(kernel,(1,1,3,3))


# 卷积操作
import torch.nn.functional as F # 导入函数
conv_output=F.conv2d(input,kernel,stride=1,padding=1)# 步长，填充
print(conv_output)



# 在网络中搭建一个卷积层
## 用卷积层和搭建卷积层不是一个函数！
from torch import nn
from typing_extensions import override # override
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()  # 继承父类构造函数，这是强制的
        # 构造卷积层，用卷积层和搭建卷积层不是一个函数！
        # self.first_conv2d=F.conv2d()
        # 搭建卷积层只要求限制输入输出通道、步长、填充
        # 高宽在上述参数决定后，自动计算
        # 输出高度 = (输入高度 + 2×padding - kernel_size) / stride + 1
        # 输出宽度 = (输入宽度 + 2×padding - kernel_size) / stride + 1
        ## 输出通道从技术上说是任意的，但是要遵循一些原则，一般来说就是两种设计模式：2 的幂次和倍增
        self.first_conv2d=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=1,padding=0)

    # 在module的 call 函数中被调用,并把输入参数原样传递给 forward
    @override 
    def forward(self,x):
        x=self.first_conv2d(x)# 调用 conv2d 的call 函数
        return x


mymodule=MyModule()
# 模型输入的类型默认是torch.float32
# 因此输入类型要对
# input=torch.tensor(input,dtype=torch.float32)# 遵循警告的提示
input=input.to(torch.float32)
output=mymodule(input)
print(output)
