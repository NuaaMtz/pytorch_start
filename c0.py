# pytorch的两个法宝函数

## 导入package（库）
import torch 
 
##  cuda状态查询函数
print(torch.cuda.is_available())

## 一个库中包含很多个部分,dir和help函数就是帮助用户
## 去查看各个部分的函数

### dir
print(dir(torch))# 查看torch中的组成
print(dir(torch.acos))# 它的组成都是__*__，约定俗成认为这种函数不应该被更改

### help

help(torch.cuda.is_available()) # 输出这个组成的可用方法的帮助文档