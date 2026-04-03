# 从常用工具箱中导入Dataset
from torch.utils.data import Dataset
from PIL import Image

import os

## 帮助文档告诉我们，要用Dataset必须要继承它，重写两个函数
# help(Dataset)


## 读取图片的方法
my_img=Image.open("./input/hymenoptera_data2/hymenoptera_data/train/ants/0013035.jpg")
# my_img.show()# 展示



## 要重写的两个函数
## __getitem__(self,idx): 要求输入一个样本的序号，返回对应的特征和标签
## __len__(self): 要求返回这个数据集的大小
class MyDataSet(Dataset):
    # 先构造特征和样本的列表
    def __init__(self,root_dir,lable_dir):
        self.root_dir=root_dir
        self.lable_dir=lable_dir
        self.path=os.path.join(self.root_dir,self.lable_dir)# 连接路径全称
        self.img_path=os.listdir(self.path)# 构造数据集
        
        
    # 要求override这个函数
    # 这个函数的功能是给出index，返回对应样本(包括特征和标签)
    def __getitem__(self, index):
        #要求返回: return (sample,label) 
        
        # 根据index获取特征 
        img_name=self.img_path[index]
        img_path=os.path.join(self.path,img_name)
        img=Image.open(img_path)# open才是真正的实例化
        
        # 获得标签(方便获取标签也是分root_dir和label_dir的原因)
        label=self.lable_dir
        
        return (img,label) 
    
    
    # 要求override这个函数
    # 要求返回数据集的大小
    def __len__(self):
        return len(self.img_path)
        

## 得到蚂蚁的数据
ants_data=MyDataSet("/home/mtzmo/Projects/pytorch_start/input/hymenoptera_data2/hymenoptera_data/train","ants")
## 得到蜜蜂的数据集
bees_data=MyDataSet("/home/mtzmo/Projects/pytorch_start/input/hymenoptera_data2/hymenoptera_data/train","bees")
## 基于Dataset的自定义Dataset遵守使用规则的情况下
## 可以进行相加减乘除
train_data=ants_data+bees_data# 两个数据集相加得到训练集 




## TensorBoard的使用
## 这个工具提供异步工具同步事件
## 也就是可以实现一边训练一边画图
from torch.utils.tensorboard import SummaryWriter
# help(SummaryWriter)

writer=SummaryWriter("output/test_writer")
for i in range(100):
    writer.add_scalar("title",3*i,i)# 标题，x，y
writer.close()

# 在项目根目录启动：
# tensorboard --logdir=output/test_writer --port 6006

# 浏览器打开：
# http://localhost:6006

# 进入 Scalars 页面，就能看到你写的 title 曲线。


## 也可以提供图片浏览
## 可以很方便查看输入的训练图片
import numpy as np


img_path="/home/mtzmo/Projects/pytorch_start/input/basic_data2/train/ants_image/0013035.jpg"
img_PIL=Image.open(img_path)
img_type_is_narry=np.array(img_PIL)
print(img_type_is_narry.shape)

writer=SummaryWriter("output/image_writer")
writer.add_image(
    "tag",
    img_type_is_narry,# 要求格式和通道符合要求
    1,# 表示该图片在第几步
    dataformats="HWC"   
    
)
writer.add_image(
    "tag",
    img_type_is_narry,# 要求格式和通道符合要求
    2,# 表示该图片在第几步
    dataformats="HWC"   
    
)



writer.close()

