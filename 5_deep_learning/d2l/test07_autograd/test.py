#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/11 15:24
# Module    : test.py
# explain   :


import torch


# y = 2x^2  ==>>>  y'=4x
# 有x = [0,1,2,3]  ，故y'=[0,4,8,12]
x = torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)

print(torch.dot(x, x), type(torch.dot(x, x)))
print(torch.dot(x, x).shape)
y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(4*x)
print(x.grad == 4*x)


import numpy as np
import matplotlib
matplotlib.use('tkagg')  # 指定使用 Agg 后端
import matplotlib.pyplot as plt
# 生成 x 值，范围从 -2π 到 2π，间隔为 0.1
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
# 计算对应的 y 值
y = np.sin(x)

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = sin(x)')
plt.legend()
plt.grid(True)
plt.show()





# 使用py画出f(x)=sinx的图像和f'(x)的图像

# 定义输入变量 x，并设置 requires_grad=True 以便追踪梯度
# x = torch.linspace(-2 * np.pi, 2 * np.pi, 100, requires_grad=True)
x = torch.linspace(0, 2 * np.pi, 100, requires_grad=True)

# 计算 f(x) = sin(x)
f_x = torch.sin(x)

# 使用 PyTorch 的自动微分功能计算 f(x) 关于 x 的导数 f'(x)
f_x.backward(torch.ones_like(x), retain_graph=True)
f_derivative = x.grad

# 将 Tensor 转换为 NumPy 数组以进行绘图
x_np = x.detach().numpy()
f_x_np = f_x.detach().numpy()
f_derivative_np = f_derivative.detach().numpy()

# 绘制图像
plt.figure(figsize=(10, 6))

# 绘制 f(x) = sin(x) 的图像
plt.subplot(2, 1, 1)
plt.plot(x_np, f_x_np, label='f(x) = sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = sin(x)')
plt.legend()

# 绘制 f'(x) 的图像
plt.subplot(2, 1, 2)
plt.plot(x_np, f_derivative_np, label="f'(x)")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title("Plot of f'(x)")
plt.legend()

plt.tight_layout()
plt.show()
