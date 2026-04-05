# 1. 优化器
# 2. 梯度清零



import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./input/CIFAR10/train", train=True, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = DataLoader(dataset, batch_size=1, drop_last=True)



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
optim=torch.optim.SGD(mymodule.parameters(),lr=0.01)# 优化器绑定模型，设置学习率




# 一个 epoch 下
for data in dataloader:
    # 梯度清零
    optim.zero_grad()# 希望每个 batch 独立
    inputs,lables=data
    outputs=mymodule(inputs)# 前向传播
    # 计算损失
    ## 初期的前向传播过程中，模型并不知道输出的 10 分类分别代表什么
    ## 是在一次一次的计算损失，然后反向传播的过程中，让模型知道了
    ## 这 10 个分类到底是什么
    ## 这里模型输出的是数字，但是在计算 loss 的过程中，用 sofemax把这些数字转换为了
    ## 概率值，
    total_loss=loss(outputs,lables)# 这里output出来的是10分类
    # 反向传播之前，先清理上一个 batch 训练的时候的梯度
    ## 原因：我们训练的时候，是先将 batch 个数据独立走
    ## 前向传播，并且计算损失，然后一次性计算一个 batch 下的综合损失，
    ## 再反向传播
    total_loss.backward()# 反向传播
    optim.step()# 梯度更新




# 几个 epoch 下
epoch_size=10


loss = nn.CrossEntropyLoss() # 交叉熵    
mymodule=MyModule()
optim=torch.optim.SGD(mymodule.parameters(),lr=0.01)# 优化器绑定模型，设置学习率

for epoch in range(0,epoch_size):
    for data in dataloader:
        # 梯度清零
        optim.zero_grad()# 希望每个 batch 独立
        inputs,lables=data
        outputs=mymodule(inputs)# 前向传播
        # 计算损失
        ## 初期的前向传播过程中，模型并不知道输出的 10 分类分别代表什么
        ## 是在一次一次的计算损失，然后反向传播的过程中，让模型知道了
        ## 这 10 个分类到底是什么
        ## 这里模型输出的是数字，但是在计算 loss 的过程中，用 sofemax把这些数字转换为了
        ## 概率值，
        total_loss=loss(outputs,lables)# 这里output出来的是10分类
        # 反向传播之前，先清理上一个 batch 训练的时候的梯度
        ## 原因：我们训练的时候，是先将 batch 个数据独立走
        ## 前向传播，并且计算损失，然后一次性计算一个 batch 下的综合损失，
        ## 再反向传播
        total_loss.backward()# 反向传播
        optim.step()# 梯度更新


