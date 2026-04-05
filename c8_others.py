# 1. 最大池化，也叫下采样：跟卷积一样滑动，但是取最大值。
## 最大池化主要是要设置卷积核尺寸，主要卷积核尺寸小于输入大小即可
# 2. 非线性激活：只改变数据
## 非线性激活不需要设置参数，其输入和输出尺寸通道一致


# 3. 展平层+线程层： 就是把多维的数据展平成一维，再通过线性层




## 1. dataset 数据
from typing_extensions import override
import torch
import torchvision
### 数据类型转换
tv_trans_tensor=torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)

### dataset
dataset_train=torchvision.datasets.CIFAR10(
    root="./input/CIFAR10/train",
    train=True,
    transform=tv_trans_tensor,# transform不是 traget_transform
    download=False
)

### dataloader 
from torch.utils.data import DataLoader
dataloader_train=DataLoader(dataset_train,batch_size=64)


## 2. 模型

from torch import nn
from torch.nn import MaxPool2d
from torch.nn import ReLU,Flatten,Linear
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.maxpool=MaxPool2d(
            kernel_size=3,
            ceil_mode=True
        )
        self.relu1=ReLU()
        self.flatten1=Flatten()
        self.linear1=nn.Linear()

    @override
    def forward(self,x):
        x=self.maxpool(x)
        x=self.relu1(x)
        return x


## 3. 调用模型
mymodule=MyModule()
from torch.utils.tensorboard import SummaryWriter 
writer=SummaryWriter("output/maxpool")

## 4. 
epoch_size=10
for epoch in range(0,epoch_size):
    # 
    step=0
    for data in dataloader_train:
        img,label=data
        writer.add_images(f"input_{epoch}",img,step)
        img=mymodule(img)
        writer.add_images(f"output_{epoch}",img,step)
        step+=1

writer.close()
