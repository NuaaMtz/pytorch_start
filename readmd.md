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

## c7_conv
*作用*： 卷积层使用和定义


## c8_others.py
*作用*： 池化，激活，线性等其他层

## c9_sequential.py
*作用*：使用 sequential 将各种层包装起来

## c10_loss
*作用*：在一个 epoch 下，使用前向传播->计算损失->反向传播 流程

## c11_optim
*作用*：在多个epoch下，添加优化器，设置学习率，每个 epoch 下都有，使用梯度清零->前向传播->计算损失->反向传播->梯度更新 流程

## c12_usermodel
*作用*: 下载 pytorch 内置的模型，修改

## c13_tranPricedure
*作用*：使用 GPU 训练，走了训练的完整流程，包括训练、验证和测试
- 一个 epoch 下分训练和测试两个部分：
    - 训练：(开启训练模式)梯度清零->前向传播->计算损失->反向传播->梯度更新
    - 验证：（开启评估模式，不计算梯度）前向传播->计算损失(或者计算其他可保存的指标)
- 对训练集可计算的指标有主要是损失曲线，准确率曲线，（分类时）混淆矩阵，（二分类时）PR曲线和ROC曲线
- 对验证集可计算的指标包括：损失，准确率，真假阳性......

- 测试集：将没进过训练和验证的数据做一次单独的测试，看结果好坏