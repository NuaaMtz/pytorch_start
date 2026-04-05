# 简单的网络模型，一般就是卷积+池化循环，最后（展平+线性）+线性

import torch
from torch.nn import Sequential
from typing_extensions import override
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Flatten, Linear


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = Conv2d(
            kernel_size=3,
            stride=1,
            padding=1,
            in_channels=1,
            out_channels=3
        )
        self.maxpool1 = MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.maxpool2 = MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )
        # 展平+线性
        self.flatter1 = Flatten()
        # 线性层输入特征需要自己计算，匹配前面的卷积+池化循环
        # 线性层的输入特征，在经过展平后，计算公式为C*H*W，其中 C 应该是前面
        # 自己设置的池化层的最后的输出通道，H*W 需要从 输入尺寸开始计算。

        # 输出特征按照自己的需求设置，比如二分类就设置为 2，如果是中间层的线性层，则按照自己的需求

        # 卷积层输出尺寸公式：（floor 代表向下取整，即直接去掉小数点，而不是四舍五入）
        # 高：H_out = floor((H_in + 2×padding - kernel_size) / stride) + 1
        # 宽：W_out = floor((W_in + 2×padding - kernel_size) / stride) + 1

        # 池化层输出尺寸公式：
        # 高：H_out = floor((H_in + 2×padding - kernel_size) / stride) + 1
        # 宽：W_out = floor((W_in + 2×padding - kernel_size) / stride) + 1

        # 假设一开始输入H=32，W=30
        # 线性层输入特征数必须准确，否则会在前向传播时报错,因此，必须先确定输入图像的尺寸！！！
        self.linear1 = Linear(5760, 2)

    @override
    def forward(self, input):
        input = self.conv1(input)
        input = self.maxpool1(input)
        input = self.conv2(input)
        input = self.maxpool2(input)
        input = self.flatter1(input)
        input = self.linear1(input)
        return input


mymodule = MyModule()

input = torch.ones((64, 1, 32, 30))  # batc_sie=64

output = mymodule(input)

print(output.shape)  # 输出为([64,2])，说明同时运行了 64 个图片，得到 64 个二分类模型参数


# 代码，__init__定义的层好像必须再在 forward 中再写一遍
# 为了省这部分功夫，就有了 Sequential()


class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()
        self.model1 = Sequential(
            Conv2d(
                kernel_size=3,
                stride=1,
                padding=1,
                in_channels=1,
                out_channels=3
            ),
            MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            Flatten(),
            Linear(5760, 2)
        )

    def forward(self, input):
        output=self.model1(input)
        return output


mymodule = MyModule2()

input = torch.ones((64, 1, 32, 30))  # batc_sie=64

output = mymodule(input)
print(output.shape)  # 输出为([64,2])，说明同时运行了 64 个图片，得到 64 个二分类模型参数