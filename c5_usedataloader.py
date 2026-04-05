

# 下载数据集
from sympy import false
import torchvision

# 转换为 tensor
tv_trans_tensor=torchvision.transforms.ToTensor()

# 获取 dataset
train_dataset=torchvision.datasets.CIFAR10(
    root="./input/CIFAR10/train",
    train=True,
    download=False,
    transform=tv_trans_tensor
)

import torch
# dataset 转换为dataloader
train_dataloader=torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)


# epoch
epoch_size=2

# 导入类
from torch.utils.tensorboard import SummaryWriter 

writer=SummaryWriter("output/usedataloard")



# 这是简单的，最普通的循环：按照每个 epoch 查看
for epoch in range(0,epoch_size):
    # 这是比较复杂的一种可迭代对象循环
    ## 1. train_dataloader每次交给data 一个 batch_size 个数的数据量
    step=0
    for data in train_dataloader:
        img,label=data # 这里解包出 4 个 img 和 4 个 lable(batch_size=4)
        writer.add_images(
            "tag_epoch_"+str(epoch),# 相同的epoch下放在一栏 
            img_tensor=img,
            global_step=step
        )
        step+=1
writer.close()
        
    