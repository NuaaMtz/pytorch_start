# 前面的各种层只是定义了前向传播
# 但是没有做反向传播，也就没有做更新梯度的操作

## 在反向传播之前，要先计算损失，因为反向传播计算的是损失的梯度


import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./input/CIFAR10/train", train=True, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = DataLoader(dataset, batch_size=1, drop_last=True)

# 获取每个图片的尺寸示例
for img, label in dataloader:
    print("图片尺寸:", img.shape)  # 输出: torch.Size([1, 3, 32, 32])
    break  # 这里只查看第一个图片的尺寸, 如果想看所有图片去掉break即可


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model1 = Sequential(
            Conv2d(
                kernel_size=3,
                stride=1,
                padding=1,
                in_channels=3,# 对应图片尺寸
                out_channels=6
            ),
            MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            Conv2d(
                in_channels=6,
                out_channels=18,
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
            Linear(18432, 10)# 10 分类，因为dataset 是 10 分类
        )

    def forward(self, input):
        output=self.model1(input)
        return output

loss = nn.CrossEntropyLoss() # 交叉熵    
mymodule=MyModule()



# 一个 epoch 下
for data in dataloader:
    inputs,lables=data
    outputs=mymodule(inputs)# 前向传播
    # 计算损失
    ## 初期的前向传播过程中，模型并不知道输出的 10 分类分别代表什么
    ## 是在一次一次的计算损失，然后反向传播的过程中，让模型知道了
    ## 这 10 个分类到底是什么
    ## 这里模型输出的是数字，但是在计算 loss 的过程中，用 sofemax把这些数字转换为了
    ## 概率值，
    total_loss=loss(outputs,lables)# 这里output出来的是10分类
    total_loss.backward()# 反向传播，更新梯度

