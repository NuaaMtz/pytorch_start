from torchvision import transforms 
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
## tensor类型
img_path="/home/mtzmo/Projects/pytorch_start/input/basic_data/train/ants_image/5650366_e22b7e1065.jpg"
img_type_is_IMAGE=Image.open(img_path)
print(type(img_type_is_IMAGE))
print(img_type_is_IMAGE)

## ToTensor能够将narry或PIL格式转换为tensor格式
## help(transforms.ToTensor)

trans_tensor=transforms.ToTensor()# 调用构造
img_type_is_tensor=trans_tensor(img_type_is_IMAGE)# 调用__call__魔术方法 
print(type(img_type_is_tensor))
print("转换为tensor类型后：",img_type_is_tensor)


writer=SummaryWriter("output/trans_tensor")
writer.add_image(
    "tag",
    img_type_is_tensor,
    1
)# tensor格式肯定符合writer要求
writer.close()






## 可以对tensor类型的文件一键归一化
trans_normal=transforms.Normalize([0.1,0.2,0.3],[0.1,0.2,0.3])#三个通道分别归一化为mean=0.1/0.2/0.3，sigma=0.1/0.2/0.3
img_norm=trans_normal(img_type_is_tensor)
print("归一化后：",img_norm)
writer=SummaryWriter("output/trans_normal")
writer.add_image(
    "tag",
    img_norm,
    1
)
writer.close()


## 可以对tensor进行裁剪
img_type_is_tensor=trans_tensor(img_type_is_IMAGE)
trans_resized=transforms.Resize((51,51))# 裁切为51*51
img_resized=trans_resized(img_type_is_tensor)
print("裁切前的大小：",img_type_is_tensor.shape,"裁切后的大小：",img_resized.shape)

trans_resized=transforms.Resize(512)# 把比较短的边像素裁切为512个像素，保持原始宽高比
img_resized2=trans_resized(img_type_is_tensor)

writer=SummaryWriter("output/trans_resized")
writer.add_image(
    "tag",
    img_type_is_tensor,
    1
)

writer.add_image(
    "tag",
    img_resized,
    2
)
writer.close()



## 随机裁切+Composes
trans_randomclip=transforms.RandomCrop((200,500))
# trans_randomclip=transforms.RandomCrop(100)
# help(transforms.Compose)# 将一系列数据转换放在一起
trans_compose=transforms.Compose(
    [
        trans_randomclip,
        trans_resized
    ]
)

#  将图片多次应用转换
img_type_is_tensor=trans_tensor(img_type_is_IMAGE)
writer=SummaryWriter("output/trans_compose")
for i in range(0,10):
    # 图片应用转换
    img_type_is_tensor=trans_compose(img_type_is_tensor)
    writer.add_image("tag",img_type_is_tensor,i)
    
writer.close()
    




