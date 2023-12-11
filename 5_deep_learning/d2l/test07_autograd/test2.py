#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/11 18:15
# Module    : test2.py
# explain   :  自动求解梯度

'''

PyTorch 是一个开源的机器学习库，提供了自动求导功能（Autograd），能够自动计算张量的梯度。这种功能的实现基于动态计算图（Dynamic Computational Graph）。

当你在 PyTorch 中定义张量并进行运算时，PyTorch 会自动构建一个计算图来表示这些操作。然后，你可以通过调用 `backward()` 方法来计算图中某个节点（通常是一个标量）相对于图中某些参数的梯度。

举个例子：

```python
import torch

# 创建一个需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 定义一个函数 f = x^2
f = x ** 2

# 对 f 进行求导
f.backward()

# 输出梯度值，即 f 对 x 的导数值
print(x.grad)
```

在这个例子中，`x` 是一个张量，我们对 `x` 进行操作得到了 `f = x^2`。通过调用 `f.backward()`，PyTorch 会自动计算出 `f` 对 `x` 的导数，并将结果存储在 `x.grad` 中。

PyTorch 的自动求导功能大大简化了梯度计算的过程，使得在神经网络训练和其他涉及梯度计算的场景中更加方便和高效。
'''
import math

import torch

import numpy as np
import matplotlib

matplotlib.use('tkagg')  # 指定使用 Agg 后端
import matplotlib.pyplot as plt


def test11():
    # 创建一个需要计算梯度的张量
    x = torch.tensor([2.0], requires_grad=True)

    # 定义一个函数 f = x^2
    f = x ** 2

    # 对 f 进行求导，并计算x在f上的梯度，且会将该梯度值存储在grad里面
    f.backward()

    # 输出梯度值，即 f 对 x 的导数值
    print(x.grad)

    pass


def test22():
    # 在某个区间内创建一些点来计算梯度
    x = torch.linspace(0, 100, 100, requires_grad=True, dtype=torch.float32)

    # 第一函数 f(x)=x^2
    f = x ** 2

    # 对f求导，并计算梯度
    f.sum().backward()  # 后向求导

    # 输出梯度
    print(x.grad)

    pass


def test_sin():
    # 创建一个需要计算梯度的张量，并指定在 (0, 2π) 之间的值
    x = torch.linspace(0, 2 * math.pi, 100, requires_grad=True)

    # 计算 sin(x)
    f_x = torch.sin(x)

    # 对 sin(x) 进行求和，转换为标量
    scalar_f = f_x.sum()

    # 对标量函数 scalar_f 进行求导
    scalar_f.backward()
    # f_x.sum().backward()

    # 输出梯度值，即 sin(x) 对 x 的导数值
    print(x.grad)

    # 获取梯度值，即 sin(x) 对 x 的导数值
    gradients = x.grad.numpy()

    # 绘制图像
    plt.figure(figsize=(8, 6))
    # plt.plot(x.detach().numpy(), gradients, label="Derivative of sin(x)")

    # 绘制 sin(x)
    plt.plot(x.detach().numpy(), f_x.detach().numpy(), label="sin(x)", color="blue")
    # 绘制导数
    plt.plot(x.detach().numpy(), gradients, label="Derivative of sin(x)", color="red")

    plt.xlabel("x")
    plt.ylabel("Gradient")
    plt.title("Derivative of sin(x) in (0, 2π)")
    plt.legend()
    plt.grid(True)
    plt.show()

    pass


def test_tan():
    # 定义域
    x = torch.linspace(0, 2 * math.pi, 100, requires_grad=True, dtype=torch.float32)

    # 定义函数
    f = torch.cos(x)

    # 求导，求导只能对标量进行，因此先将y的张量转化为标量，之后在进行求导
    f.sum().backward()

    # 输出
    print(x.grad)

    # 绘制图像
    plt.figure(figsize=(8, 6))
    # plt.plot(x.detach().numpy(), gradients, label="Derivative of sin(x)")

    # 绘制 sin(x)
    plt.plot(x.detach().numpy(), f.detach().numpy(), label="cos(x)", color="blue")
    # 绘制导数
    plt.plot(x.detach().numpy(), x.grad.numpy(), label="Derivative of cos(x)", color="red")

    plt.xlabel("x")
    plt.ylabel("Gradient")
    plt.title("Derivative of cos(x) in (0, 2π)")
    plt.legend()
    plt.grid(True)
    plt.show()

    pass


def test():
    x = torch.linspace(-5, 5, 100, requires_grad=True)

    f = x ** 2 + 5 * x + 10

    f.sum().backward()

    print(x.grad)

    # 绘制图像
    plt.figure(figsize=(8, 6))
    # plt.plot(x.detach().numpy(), gradients, label="Derivative of sin(x)")

    # 绘制 sin(x)
    plt.plot(x.detach().numpy(), f.detach().numpy(), label="cos(x)", color="blue")
    # 绘制导数
    plt.plot(x.detach().numpy(), x.grad.numpy(), label="Derivative of cos(x)", color="red")

    plt.xlabel("x")
    plt.ylabel("Gradient")
    plt.title("Derivative of cos(x) in (0, 2π)")
    plt.legend()
    plt.grid(True)
    plt.show()


    pass


if __name__ == '__main__':
    # test11()

    # test22()

    # test_sin()

    # test_tan()

    test()

    pass
