# nn.Module 是神经网络的一个核心类
# 给训练模型一个模板，只需要修改重要的部分即可。

import torchvision
from typing_extensions import override
import torch
# 导入模块，看不出来和类的区别
from torch import nn


class MyModule(nn.Module):# 继承自 Module
    def __init__(self):# 构造
        #! 注：必须继承父类的构造函数，这是强制的！
        super(MyModule,self).__init__()# 继承父类构造函数<-------父类中的__call__函数调用了 forward，所以真正用的是不需要手动调用


    @override# 提醒自己这是重写的函数的修饰
    def forward(self,input):# input代表的是当前层输入
        print("__call__函数自动调用这个函数")        
        # call函数调用 forwad,并且原样传递参数给 forward
        output=input+1
        return output

import numpy 
mymodule=MyModule()
x_type_is_int=1.
x_type_is_narry=numpy.array(x_type_is_int)
trans_tensor=torchvision.transforms.ToTensor()# 这种转换必须是 3 通道的图片
# x_type_is_tensor=trans_tensor(x_type_is_narry)
x_type_is_tensor=torch.tensor(x_type_is_int)
x_type_is_tensor=torch.tensor(x_type_is_narry)
output=mymodule.forward(x_type_is_tensor)
print( "输入为：",x_type_is_tensor,"输出为：", output)


mymodule=MyModule()
mymodule(2)
