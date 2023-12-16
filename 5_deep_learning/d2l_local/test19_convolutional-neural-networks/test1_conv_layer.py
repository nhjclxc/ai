#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/16 20:13
# Module    : test1_conv_layer.py
# explain   : 图像卷积
# http://localhost:8888/notebooks/d2l_local/d2l-pytorch/chapter_convolutional-neural-networks/conv-layer.ipynb


import torch
from torch import nn
from d2l import torch as d2l


# 1. 互相关运算


def corr2d(X, K):  # @save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def test1_corr2d():
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))
    pass


def test1_corr2d_2():
    def corr2d_2(X, K):
        # Y.h = (X.h-K.h+1, X.w-K.w+1)

        Y = torch.zeros((X.shape[0] - K.shape[0] + 1, X.shape[1] - K.shape[1] + 1))
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                sub_m = ((X[i: K.shape[0] + i, j: K.shape[1] + j]) * K)
                print(sub_m)
                y = sub_m.sum()
                print(y)
                Y[i, j] = y
        return Y

    X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = torch.tensor([[0, 1], [2, 3]])
    print(corr2d_2(X, K))


# 2. 卷积层
'''
卷积层中的两个被训练的参数是卷积核权重W和标量偏置b
'''


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # 随机初始化权重参数W和偏置b，偏置是一个标量
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 正向传播执行卷积核的计算
        return corr2d(x, self.weight) + self.bias


def test2_Conv2D():
    # 测试图像中目标的边缘检测
    X = torch.ones((6, 8))
    # 1->0或0->1的就是要检测出的边界
    X[:, 2:6] = 0
    print(X)

    # 构造一个高度为1宽度为2的卷积核来对列进行模拟边缘检测
    K = torch.tensor([[1.0, -1.0]])

    # [输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘]，
    Y = corr2d(X, K)
    print(Y)

    K = torch.tensor([[1.0], [-1.0]])
    print(corr2d(X.t(), K))

    pass


def test2_Conv2D_2():
    print('测试行列均可检测')
    X = torch.ones((8, 8))
    # 1->0或0->1的就是要检测出的边界
    X[2:6, 2:6] = 0
    print(X)

    K = torch.tensor([[1.0, -1.0], [-1.0, 0.0]])

    print(corr2d(X, K))

    pass


def test3_():
    # 通过数据来训练学习卷积核
    # 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    # 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
    # 其中批量大小和通道数都为1
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    true_K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, true_K)
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2  # 学习率

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        print(f'epoch {i + 1}, loss {l.sum():.3f}，w = {conv2d.weight.data.reshape((1, 2))}')

    print(conv2d.weight.data.reshape((1, 2)))

    pass


def test4_():
    # 对角线边缘检测

    # 对角线为1
    X = torch.ones((8, 8))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if i == j:
                X[i, j] = 0
    print(X)
    # X = torch.eye(8)

    K = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

    Y = corr2d(X, K)
    print(Y)

    return X, Y



def test4_1():
    # 通过数据来训练学习卷积核
    # 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(2, 2), bias=False)

    X, Y = test4_()
    X = X.reshape((1, 1, 8, 8))
    Y = Y.reshape((1, 1, 7, 7))
    lr = 0.01

    for i in range(9):
        Y_hat = conv2d(X)
        loss = (Y_hat - Y)**2
        conv2d.zero_grad()
        loss.sum().backward()

        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        print(f'epoch {i + 1}, loss {loss.sum():.3f}，w = {conv2d.weight.data.reshape((2, 2))}')

    print(conv2d.weight.data.reshape((2, 2)))


    pass


def test4_2():

    # 定义一个简单的2x2的边缘检测核
    edge_detection_kernel = torch.tensor([
        [1, -1],
        [-1, 1]
    ], dtype=torch.float32)

    # 使用conv2d对X和Y进行卷积操作，观察输出结果
    X = torch.tensor([
        [0., 1., 1., 1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 1., 1., 1.],
        [1., 1., 1., 1., 0., 1., 1., 1.],
        [1., 1., 1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 1., 1., 1., 0.]
    ])

    Y = torch.tensor([
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1.]
    ])

    # 对X进行卷积操作
    edge_detected_X = torch.nn.functional.conv2d(X.unsqueeze(0).unsqueeze(0),
                                                 edge_detection_kernel.unsqueeze(0).unsqueeze(0))
    print("X 的边缘检测结果：")
    print(edge_detected_X)

    # 对Y进行卷积操作
    edge_detected_Y = torch.nn.functional.conv2d(Y.unsqueeze(0).unsqueeze(0),
                                                 edge_detection_kernel.unsqueeze(0).unsqueeze(0))
    print("\nY 的边缘检测结果：")
    print(edge_detected_Y)

    pass


def test4_3():
    import torch.optim as optim

    # 输入（X）
    X = torch.tensor([
        [0., 1., 1., 1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 1., 1., 1.],
        [1., 1., 1., 1., 0., 1., 1., 1.],
        [1., 1., 1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 1., 1., 1., 0.]
    ])

    # 输出（Y）
    Y = torch.tensor([
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1.]
    ])


    # 将输入和输出调整为适当的形状
    X = X.unsqueeze(0).unsqueeze(0).float()  # 添加批次和通道维度
    Y = Y.unsqueeze(0).unsqueeze(0).float()

    # 定义一个简单的卷积神经网络模型
    model = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)

    # 定义损失函数和优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.MSELoss(output, Y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    # 使用训练好的模型对X进行预测

    X2 = torch.tensor([
        [0., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1.],
        [1., 1., 0., 1., 1.],
        [1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 0.]
    ])
    predicted_Y = model(X2.reshape(1,1,5,5))
    print("\n预测的输出：")
    print(predicted_Y)

    pass


if __name__ == '__main__':
    # test1_corr2d()

    # test1_corr2d_2()

    # test2_Conv2D()

    # test2_Conv2D_2()

    # test3_()

    # test4_()

    # test4_1()
    # test4_2()
    test4_3()

    pass
