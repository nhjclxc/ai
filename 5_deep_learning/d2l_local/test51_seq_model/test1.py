#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/17 22:52
# Module    : test1.py
# explain   :


import torch
from torch import nn
from d2l import torch as d2l

# 生成一些数据：(使用正弦函数和一些可加性噪声来生成序列数据
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))


# 将输入特征与标签进行映射 ref：将输入特征与标签进行映射.jpg
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))


# 构造训练数据集
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# [使用一个相当简单的架构训练模型： 一个拥有两个全连接层的多层感知机]，ReLU激活函数和平方损失。
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    # 4表示tau的值，即输入特征是4，
    # 10表示的是这个隐藏层有10个隐藏单元
    # 使用relu作为激活函数
    # 10表示输出层接受的输入是上一层即隐藏层的输出的10个单元的数据
    # 1表示的就是通过T前面的tau=4个元素，预测出来的当前T的元素
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

# 训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 预测
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time','x',
         legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.show()


onestep_preds2 = net(features[600:])
d2l.plot([time[600:], time[tau+600:]],
         [x[600:].detach().numpy(), onestep_preds2.detach().numpy()], 'time2','x2',
         legend=['data2', '1-step preds2'], xlim=[1+600, 1000],
         figsize=(6, 3))
d2l.plt.show()