# torchvision用来下载数据集
import torchvision

# help(torchvision.datasets.CIFAR10)# 手写数据集

## 下载这个数据集中的训练集
train_data=torchvision.datasets.CIFAR10(
    root="./input/CIFAR10/train",# 下载路径
    train=True,# 默认false下载测试集
    download=True
)

print(type(train_data))
print(len(train_data))

## 下载测试集
test_data=torchvision.datasets.CIFAR10(
    root="./input/CIFAR10/test",
    train=False,
    download=True
)

## 之后查看doc，了解数据集的结构
## 这个数据集是一个(img,lable)元组的列表：
first_img,first_label=train_data[0]
print(first_img,first_label)


## 使用tensorborad查看数据集
## 所有数据都转成tensor
from  torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
trans_tensor=transforms.ToTensor()
# for i in (0,len(test_data)):
#     test_data[i]=trans_tensor(test_data[i])# 原来的格式不是PIL/narry格式，不能直接转
## torchvision提供了统一转换的方法：
tv_trans_tensor=torchvision.transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
train_data=torchvision.datasets.CIFAR10(
    root="./input/CIFAR10/train",# 下载路径
    train=True,# 默认false下载测试集
    download=False,
    transform=tv_trans_tensor
)
first_img,first_label=train_data[0]
print(type(first_img))# 现在是tensor格式了



writer=SummaryWriter("output/torchvision")
for i in range(0,len(train_data)):
    img,label=train_data[i]
    writer.add_image("tag",img,i)
    
writer.close()
