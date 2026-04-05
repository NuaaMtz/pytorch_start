# 完整的训练套路
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch

## 0. 显卡
device=0
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


## 1. 数据集
# 准备数据集
train_data = torchvision.datasets.CIFAR10("./input/CIFAR10/train",train=True,transform=torchvision.transforms.ToTensor(),download=True)       
test_data = torchvision.datasets.CIFAR10("./input/CIFAR10/test",train=False,transform=torchvision.transforms.ToTensor(),download=True)       

# 训练集和测试集
train_dataloader = DataLoader(train_data, batch_size=64)        
test_dataloader = DataLoader(test_data, batch_size=64)


## 2.  模型

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()        
        self.model1 = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),  # 输入通道3，输出通道32，卷积核尺寸5×5，步长1，填充2    
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 展平后变成 64*4*4 了
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x


## 3. 训练前准备
### - 模型实例化
### - 优化器
### - 损失函数
### - epoch个数
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

#### 模型
mymodule=MyModule().to(device=device)# 放到显卡上

#### 优化器
learning = 0.01  # 1e-2 
optimizer = torch.optim.SGD(mymodule.parameters(),learning)   # 随机梯度下降优化器  

#### 损失函数
loss_fn = nn.CrossEntropyLoss() # 交叉熵，fn 是 fuction 的缩写

####  epoch
epoch_size=1000

#### tensorborad 
from torch.utils.tensorboard import  SummaryWriter
writer=SummaryWriter("output/trantProcedure")


## 4. 训练八股文

# 模型看最佳准确率
best_acc=0

### 1. 循环 epoch
for epoch in range(0,epoch_size):
    print(f"----- 这是第{epoch}轮训练的开始： -------" )
    # 当网络中有dropout层、batchnorm层时，这些层能起作用
    mymodule.train()# 开启训练模式,

    loss_of_each_epoch=0
    ### 2.  数据按照 batch_size 打开
    for trant_order,data in enumerate(train_dataloader):
        imgs,lables=data# 每个都是 batch_size 个图像/标签
        imgs=imgs.to(device)
        lables=lables.to(device)# 这里数据也要放显卡

        ### 3. 梯度清零
        optimizer.zero_grad() 

        ### 4. 前向传播
        outputs=mymodule(imgs)# batch_size 个输出

        ### 5. 计算损失
        loss=loss_fn(outputs,lables)

        ### 6. 反向传播
        loss.backward()

        ### 7. 梯度更新
        optimizer.step()

        ## 计算损失
        loss_of_each_epoch+=loss.item()# 计算累计的损失



        # print(f"第{epoch}轮训练，第{trant_order}个bacth,损失为{loss.item()}")# 这样直接输出太多了
        if trant_order%100==0:
            print(f"第{epoch}轮训练，第{trant_order}个bacth,损失为{loss.item()}")
    
    # 画图填充
    writer.add_scalar("loss vs. epoch",loss_of_each_epoch,epoch)


    ### 8. 验证
    # 当网络中有dropout层、batchnorm层时，这些层能起作用
    mymodule.eval()# 评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():# 确保下面的代码运行过程不会计算梯度
        for data in test_dataloader:
            imgs,labels=data
            imgs=imgs.to(device)
            labels=labels.to(device)
            #### 8.1 前向传播
            outputs=mymodule(imgs)

            #### 8.2 计算指标
            loss=loss_fn(outputs,labels)
            test_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        accuracy = 100 * correct / total
        print(f"第{epoch}轮测试集上的准确率为: {accuracy:.2f}%")
        writer.add_scalar("test accuracy vs. epoch", accuracy, epoch)
        
        # 每个 epoch 都保存一个模型
        torch.save(mymodule,f"./output/model/mymodule_{epoch}.pth")

        
            
    

writer.close()

# 测试



## 1. 加载测试图片
import requests
# 直接从网页加载图片为 img 对象，不保存到本地
import requests
from PIL import Image
from io import BytesIO

url = "https://png.pngtree.com/png-clipart/20230824/original/pngtree-beagle-wearing-glasses-reading-book-cartoon-style-png-image_9391075.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))



## 2. 加载最佳模型
best_model=torch.load("input/model/bestmodel.pth",map_location=torch.device('cpu'))

## 3. 对图片做训练一样的预处理 

## 3. 推理
best_model.eval()
with torch.no_grad():
    output=best_model(img)
    ## 4. 输出结果
    print(output.argmax(1))

