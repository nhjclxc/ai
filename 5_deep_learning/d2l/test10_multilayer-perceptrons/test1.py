#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/12 20:31
# Module    : test1.py
# explain   :

# %matplotlib inline
# import torch
# from d2l import torch as d2l
# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# d2l.plt.show()
#
# y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
# d2l.plt.show()

import torch
from torch import nn
from d2l import torch as d2l
from IPython import display
# from d2l.d2l_local import torch as d2l
# from
# 5_deep_learning/d2l/d2l_local/torch.py
'''
手写数字的识别 
'''

#  1. 数据下载
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# 2.初始化模型参数  实现一个具有单隐藏层的多层感知机， 它包含256个隐藏单元
# num_inputs： 输入张量的维度，即每一张图片的像素点个数（特征个数）
# num_outputs：输出特征个数，即手写数字识别的0-9这10个数字
# num_hiddens：隐藏层的大小   超参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 第一层隐藏层的参数
# W1：行数num_inputs输入特征个数，列数num_hiddens隐藏层个数，构造一个[num_inputs,num_hiddens]的权重矩阵
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# b1：行数为1，列数num_hiddens就是每一个隐藏层单元对应的偏置
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

# 第二层隐藏层的参数
# W2：行数num_hiddens第一个隐藏层的输出特征个数作为第二个隐藏层的输入特征个数，
#   列数num_outputs隐藏层输出个数，构造一个[num_hiddens,num_outputs]的权重矩阵
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# b1：行数为1，列数num_outputs就是每一个隐藏层单元对应的偏置
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 整个多层感知机MLP的全部参数
params = [W1, b1, W2, b2]

# 3. 激活函数
def relu(X):
    # 创建一个与X相同的全0矩阵
    a = torch.zeros_like(X)
    # 实现x和0取最大的功能
    return torch.max(X, a)

# 4.模型
def net(X):
    # 将输入图片的矩阵转化为一维向量
    X = X.reshape((-1, num_inputs))
    # 隐藏层经过权重计算后在通过内部的激活函数，得到第一层隐藏层的输出
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    # H = relu(torch.mm(X,W1) + b1)
    # H为隐藏层的输出，将H作为第二层的输入特征，return返回的就是输出层得到的最终输出，只有隐藏层才需要激活函数激活
    return (H@W2 + b2)
    # return (torch.mm(X,W2) + b2)

# 5. 损失函数
# softmax函数将(-无穷, +无穷)的输出映射到(0,1)的输出
loss = nn.CrossEntropyLoss(reduction='none')

# 6. 训练




def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 训练

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        d2l.plt.show()
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# 多层感知机的训练过程与softmax回归的训练过程完全相同
# 迭代次数，学习率
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
predict_ch3(net, test_iter)
print('\nmlp')






