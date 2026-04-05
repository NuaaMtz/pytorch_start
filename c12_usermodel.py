# 下载别人的模型，使用别人的模型
## 这种一般有别人的用法，比较少用 pytorch 内置的模型
import torchvision


# 下面演示如何加载和使用 torchvision 中的 VGG16 模型：

# 加载预训练的 VGG16 模型（使用ImageNet上训练好的参数）
vgg16_true = torchvision.models.vgg16(pretrained=True) 
# 这样下载得到的 vgg16_true，其卷积层、池化层等参数都是在ImageNet上训练好的。适合迁移学习或者做特征提取。

# 加载未预训练的 VGG16 模型（随机初始化参数）
vgg16_false = torchvision.models.vgg16(pretrained=False) 
# vgg16_false的参数全部是随机初始化的，需要从头开始训练。

print("ok") 
# 打印模型结构，方便查看网络的层次与参数信息
print(vgg16_true)


import torchvision
from torch import nn

# 加载CIFAR10数据集
dataset = torchvision.datasets.CIFAR10(
    "./dataset", 
    train=True, 
    transform=torchvision.transforms.ToTensor(), 
    download=True
)       

# 下载预训练好的VGG16模型，参数在ImageNet上已经训练好
vgg16_true = torchvision.models.vgg16(pretrained=True) 
# 此时，模型最后的全连接层输出1000类别（适合ImageNet的数据集）

# 由于CIFAR10一共有10个类别，因此需要修改模型最后的全连接层以适应CIFAR10
# 这里直接在原有的vgg16模型后面添加了一个线性层，将输出从1000维变为10维
# 注意：这种做法会让vgg16的原有输出(1000)再接上一层nn.Linear(1000, 10)
vgg16_true.add_module('add_linear', nn.Linear(1000, 10)) 

# 打印模型结构，观察在模型最后多了一层线性层以适配CIFAR10
print(vgg16_true)


import torchvision
from torch import nn

# 加载一个未经过预训练的VGG16模型，参数都是随机初始化
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 没有预训练的参数
print("修改前模型结构如下：")
print(vgg16_false)

# 默认的VGG16模型最后一层的全连接层输出是1000（用于ImageNet的1000分类），
# 这里我们将其改为输出10类，以适应CIFAR10数据集
vgg16_false.classifier[6] = nn.Linear(4096, 10)

print("修改后模型结构如下：")
print(vgg16_false)