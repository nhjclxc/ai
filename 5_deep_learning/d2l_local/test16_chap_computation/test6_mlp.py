#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 18:20
# Module    : test6_mlp.py
# explain   : 使用nn.Module实现mlp，实现手写数字的识别



import random
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')  # 动态画图 指定使用 TkAgg 后端
from matplotlib import pyplot as plt


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return F.log_softmax(self.fc3(X), dim=1)


def train(net, train_iter, epochs, lr, test_iter):
    # 优化函数
    optimizer = torch.optim.Adam(net.parameters(), lr)
    print(f' init, assess {test(net, test_iter, test_size=10, assess=True)}')
    for epoch in range(epochs):
        for (input, label) in train_iter:
            # 梯度置0，防止梯度累加
            net.zero_grad()
            # 输入网络
            outputs = net.forward(input.view(-1, 28 * 28))
            # 计算损失
            loss = F.nll_loss(outputs, label)
            # 梯度反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        print(f'epoch {epoch}, assess {test(net, test_iter, test_size=10,assess=True)}')
    pass


def test(net, test_iter, test_size=5, assess = False):
    # 获取第一个批次的数据
    first_batch = next(iter(test_iter))
    x, y = first_batch[0], first_batch[1]
    # 随机生成len(test_iter)以内的test_size个随机数
    random_index_list = [random.randint(0, len(test_iter)) for _ in range(test_size)]
    random_inputs, random_labels = x[random_index_list], y[random_index_list]
    if assess :
        counter = 0
        for i in range(test_size):
            predict = torch.argmax(net.forward(random_inputs[i][0].view(-1, 28 * 28)))
            counter += int(random_labels[i]) == int(predict)
        return round(float(counter)/test_size, 3)
    else:
        for i in range(test_size):
            predict = torch.argmax(net.forward(random_inputs[i][0].view(-1, 28 * 28)))
            plt.figure(random_labels[i])
            plt.imshow(random_inputs[i][0].reshape(28, 28), cmap='gray')
            plt.title('prediction: ' + str(int(predict)))
            plt.axis('off')  # 不显示坐标轴
        plt.show()


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

def main():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    mlp = MLP()
    epochs, lr = 5, 0.01
    train(mlp, train_iter, epochs, lr, test_iter)

    test(mlp, test_iter, 5)

    pass


if __name__ == '__main__':
    main()