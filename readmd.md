# 自学pytorch的教程
[课程链接](https://www.bilibili.com/video/BV1hE411t7RN?spm_id_from=333.788.player.switch&vd_source=77a088a2faecd65414dc2dd9e4a6c45e&p=8)

## c1_dataloader  
*作用*： 将数据包装成统一的形式，利于后续对图像的训练

Dataset，TensorBoard工具


## c2_transforms
*作用*：格式转换后，可以用很多额外的函数一次性做处理
- 将IMG/narry格式转换为tensor格式
- tensor格式带有一些其他属性，比如梯度属性，能够被后面的神经网络所调用
- 打包后可以一键归一化
- 裁切
- compose函数

## c3_titorchvision
*作用*: 下载内置数据集，并tensor化

## c4_dataloader
*作用*：按照 batch_size 读取数据

## c5_userdataloader
*作用*： 按照 epoch,batch 读取数据

## c6_nn_Module
*作用*： 一个简单的 自定义 nn.Module 写法
- 很重要的一点是知道forword 函数是被默认在 nn.Module 中继承的 call 函数所调用的。