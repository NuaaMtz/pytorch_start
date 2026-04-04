# 用pytorch，输入数据格式都是dataloader格式。
# 而dataset可以很方便转换为网络的输入，而不需要自己再手动去做
# 复杂计算

import torchvision 
# 加载dataloader
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter




# 全部转换为tensor
tv_trans_compose=torchvision.transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
train_dataset=torchvision.datasets.CIFAR10(
    root="./input/CIFAR10/train/",
    train=True,
    download=False,
    transform=tv_trans_compose
)


# dataset包装为Dataloader
# help(DataLoader)
## 区分epoch和batch_size的意思
### epoch：假设训练数据为100，那么把这100个数据丢网络训练一次，就是一个epoch
### batch: 一个epoch有100个数据，但是每次都是丢batch个数据给网络



train_dataloader=DataLoader(
    dataset=train_dataset,# dataset
    batch_size=4,# 每次从数据集中取出batch_size个样本（如图像和标签）作为一个批次输入模型
    shuffle=True,# 是否在每个训练周期（epoch）开始时打乱数据顺序
    num_workers=0,# 用于数据加载的子进程数
    drop_last=False# 是否丢弃最后一个不完整的批次
)
## 注：返回的DataLoader是可迭代对象，每次迭代都是包含了batch_size个的数据的一个列表

for data in train_dataloader:
    print(type(data))# 列表,一般分两个，一个是img一个是lable
    for i,(img,lable) in enumerate(zip(data[0],data[1])):# 使用enumerate解包获取循环变量,zip合并同时迭代两个对象
        print("这是第",i,"个图像的尺寸：",img.shape,"其对应的标签为：",lable)
        
# tensorboard画图
 
writer=SummaryWriter("output/dataloader")

for i,data in  enumerate(train_dataloader):
    
    img,lable=data
    writer.add_images("tag",img,i)# images可以记录整个batch

writer.close()    
    




