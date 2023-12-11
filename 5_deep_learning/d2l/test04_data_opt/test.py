#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/10 21:58
# Module    : test.py
# explain   : 04 数据操作 + 数据预处理【动手学深度学习v2】

import torch
import numpy


def create():
    x = torch.arange(12)
    print(x.shape)
    x2 = x.reshape(3, 4)
    print(x2.shape)

    # 全0
    x = torch.zeros(12, dtype=torch.float32)
    print(x)

    # 全1
    x = torch.ones(12, dtype=torch.int32)
    print(x)
    x = x.reshape(3, 4)
    print(x)

    # 三维
    x = x.reshape(2, 2, -1)
    print(x)
    print(x.shape)

    # 随机矩阵
    x = torch.randn(12, dtype=torch.float32)
    print(x)

    # 使用py列表来创建
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x)


def opt():
    x = torch.ones(3, 4, dtype=torch.int32)
    print(x)
    print(x.shape)
    y = torch.tensor([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    print(y)
    print(y.shape)

    # 形状必须相同
    # 对两个张量进行+、-、*、/、**都是对对应位置的元素进行对应操作
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    # print(y // x)
    # print(y ** x)

    print('\n\n')
    print(torch.exp(y))

    print('\n\n 张量链接 torch.cat')
    xy0 = torch.cat((x, y), dim=0)  # 行方向进行链接
    print(xy0)
    xy1 = torch.cat((x, y), dim=1)  # 列方向进行链接
    print(xy1)

    # 通过逻辑 == 来判断两个张量对应元素是否相同
    print(x == y)

    print(x.sum())
    print(y.sum())
    # 使用x.sum()返回的其实还是一个torch的张量对象，只不过改张量只有一个数字那就是x的所有元素累加的结果
    # 当张量中只有一个元素时，你可以使用int()将张量中的值提取为Python整数。
    print(int(y.sum()))

    pass


def broadcasting():
    # 在大多数情况下，我们将沿着数组中长度为1的轴进行广播，
    x = torch.arange(3).reshape(3, 1)
    y = torch.arange(2).reshape(1, 2)

    print(x)
    print(y)

    z = x + y
    print(z)

    a = torch.ones(3, 1, dtype=torch.int32)
    b = torch.tensor([1, 2])
    print(a + b)

    pass


def index_opt():
    # 张量中的元素可以通过索引访问

    x = torch.tensor([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    print(x)
    # 读取
    print(x[2,])
    print(x[:2,])
    print(x[:3,])
    print(x[:3,1:3])

    print('\n写入')
    x[2, ] = 66
    print(x)

    pass


def copy():
    ''' 张量的拷贝操作 '''

    x = torch.ones(3, 4, dtype=torch.int32)
    print(x)
    print(id(x))
    y = torch.tensor([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    print(y)
    print(id(y)) # 2033282285328

    # 现在把x+y的值存放到y里面
    y = x+y
    print(y)
    print(id(y)) # 2145061529424

    # 通过两个print(id(y))的输出可知 x+y产生了一个新的张量，py把这个新的张量地址赋值给了y，可以看出这时一个浅拷贝
    # 如何实现深拷贝？？？
    # 实现深拷贝，要使用：y[:] = x + y来实现
    y[:] = x + y
    print(y)
    print(id(y)) # 2145061529424，可以看出这个地址和上一次的地址就一样了
    # 因此实现深拷贝，即原地拷贝
    y += x
    print(y)
    print(id(y)) # 2145061529424

    # 单个张量的浅拷贝与深拷贝
    x = torch.ones(3, 4, dtype=torch.int32)
    print(x)
    print(id(x)) # 1203485163280
    z1 = x
    print(z1)
    print(id(z1)) # 1203485163280
    z2 = x.clone()
    print(z2)
    print(id(z2)) # 1204044588480

    pass


def transfor():
    '''py对象之间的转化'''

    x = torch.arange(12).reshape(3,4)
    y = numpy.arange(12,24).reshape(3,4)
    print(x, x.shape, type(x))
    print(y, y.shape, type(y))

    # 将 torch的Tensor张量对象转化为numpy的ndarray张量
    z = torch.tensor(y)
    print(z, z.shape, type(z))


    # 将大小为1的张量转化为py的一个标量
    # 使用x.item()或py的强制类型转化int()、float()
    a = torch.tensor([666])
    print(a, a.shape, type(a))
    print(a.dtype)
    b = a.item()
    c = int(a)
    print(b, type(b))
    print(c, type(c))


    pass


def mult_dot():
    x = torch.arange(3)
    y = torch.arange(3)
    # x和y两个张量做点积，对应位置元素做乘积然后相加成一个标量
    z = x.dot(y)
    print(x)
    print(y)
    print(z)
    print(z.item())

    print(x * y) # 两个张量对应位置做乘积，然后放到一个新的张量对应的位置上返回


    x = torch.arange(3).reshape(1, 3)
    y = torch.arange(3).reshape(3, 1)
    print(x)
    print(y)

    # 直接使用两个多维张量做点击
    # 如果你想对两个张量 x 和 y 进行点积（内积），可以使用 PyTorch 中的 torch.mm() 或者 torch.matmul() 函数
    print(torch.mm(x,y))
    print(torch.matmul(x,y))

    # 调整张量维度来做点积
    x = x.squeeze(0)  # 移除大小为 1 的维度
    y = y.squeeze(1)  # 移除大小为 1 的维度

    print(x)
    print(y)
    # 进行点积操作
    result = torch.dot(x, y)
    print(result)


    pass


def practice():
    '''
        运行本节中的代码。将本节中的条件语句X == Y更改为X < Y或X > Y，然后看看你可以得到什么样的张量。
    '''


    x = torch.ones(3, 4, dtype=torch.int32)
    print(x)
    print(x.shape)
    y = torch.tensor([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    print(y)
    print(y.shape)

    # 逻辑运算，对对应位置做逻辑运算，对应的结果放在改位置上，创建一个新的张量返回
    print(x == y)
    print(x > y)
    print(x < y)

    '''
        用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？
    '''
    a = torch.tensor([[[1,2,3]]])
    b = torch.tensor([[[1],[2],[3]]])
    c = torch.tensor([[[1]],[[2]],[[3]]])
    print(a, a.shape)
    print(b, b.shape)
    print(c, c.shape)
    print(a+b+c)




    pass


if __name__ == '__main__':
    # create()

    # opt()

    # broadcasting()

    # index_opt()

    # copy()

    # transfor()

    # mult_dot()

    practice()


    pass
