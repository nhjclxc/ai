#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/13 15:11
# Module    : impl_mnist_reg.py
# explain   : 利用nn.Module实现手写数字识别


import random
import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')  # 动态画图 指定使用 TkAgg 后端
from matplotlib import pyplot as plt

'''
https://www.bilibili.com/video/BV1GC4y15736/
'''

# 1.下载数据
def load_data_fashion_mnist(batch_size, resize=None):
    """
        下载手写数字识别数据集
    :param batch_size: 批量大小
    :param resize:
    :return: 返回（训练集，测试集）
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # FashionMNIST --> MNIST
    mnist_train = torchvision.datasets.MNIST( root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST( root="./data", train=False, transform=trans, download=True)
    '''
    torch.utils.data.DataLoader返回的数据结构
        返回一个迭代器，每个迭代返回的元素通常是一个包含输入特征和对应标签的元组 (inputs, labels) 
        inputs是一个<class 'torch.Tensor'>的对象，形状shape是torch.Size([256, 1, 28, 28])，包含了四个维度
            第一个维度就是你的批量大小batch_size，
            第二个维度表示通道数（channels），表示每个样本中包含了一个通道，因为是灰度图像0表示白1表示黑
            第三个维度是输入图片有多少行，第二个维度是输入图片有多少列
        
        labels是一个<class 'torch.Tensor'>对象，形状是torch.Size([256])
            表示的是[1,256]即1行256列，这个256也就是批量大小batch_size，和图片的个数一一对应，值为0-9的数字
    '''
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

# 2.模型定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 神经网络的权重w和偏置b在内部已经定义好了
        # 28*28表示输入图片像素点特征个数，64表示第一层隐藏层的隐藏单元个数
        self.fc1 = nn.Linear(28*28, 64)
        # 64表示第一层隐藏层的隐藏单元输出特征个数，32表示第二层隐藏层输出单元个数
        self.fc2 = nn.Linear(64, 32)
        # 32表示第二层隐藏层的隐藏单元输出特征个数，10表示输出层的分类个数
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        '''
        定义反向传播
        :param x: 图片x输入特征
        :return: 经过神经单元运算后的特征x的输出
        '''
        '''
        这段代码属于神经网络模型的前向传播部分，通常用于定义神经网络的结构及其计算过程。让我逐行解释：

        1. `x = nn.functional.relu(self.fc1(x))`: 这行代码对输入数据 `x` 应用了第一个全连接层 `self.fc1`，然后使用 ReLU 激活函数。`self.fc1` 是一个全连接层（即线性层），它将输入 `x` 进行线性变换并将结果传递给 ReLU 激活函数。ReLU 激活函数负责对每个神经元的输出进行非线性映射，将负数映射为零，保持正数不变。
        
        2. `x = nn.functional.relu(self.fc2(x))`: 同样地，这一行将上一层的输出 `x` 输入到第二个全连接层 `self.fc2` 中，然后再次应用 ReLU 激活函数。这个过程将网络的特征表示进一步转换，并且通过激活函数引入非线性。
        
        3. `x = nn.functional.log_softmax(self.fc3(x), dim=1)`: 这一行代码使用第三个全连接层 `self.fc3` 对输入 `x` 进行线性变换，并将结果应用于 LogSoftmax 激活函数。LogSoftmax 是 Softmax 函数与对数函数的组合，它将神经网络的原始输出转换为对数概率值，用于多分类问题的输出。`dim=1` 表示沿着第一个维度（通常是样本维度）进行计算，确保每个样本的输出是一个概率分布。
        
        整个函数 `forward` 描述了神经网络的正向传播过程，其中通过堆叠层和激活函数逐步转换输入数据，最终输出适合特定任务的预测结果。
        '''
        # 经过第一层
        # self.fc1(x)：表示全连接层的线性计算，即完成∑=X·W+b
        # nn.functional.relu()：表示隐藏单元的激活函数，将经过线性计算后的值经过激活函数后输出
        x = nn.functional.relu(self.fc1(x))
        # 经过第二层
        x = nn.functional.relu(self.fc2(x))
        # 经过最后一层输出层
        # 经过softmax之后将输出，转化为概率，归一化处理
        # dim=1 表示沿着第一个维度（通常是样本维度）进行计算，确保每个样本的输出是一个概率分布。
        x = nn.functional.log_softmax(self.fc3(x), dim=1)
        # 将经过变换的x特征输出
        return x

# 3. 评估函数
def evaluate(test_data, net):
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for (x, y) in test_data:
            # view(-1, 28*28)表示将输入图片矩阵的形状转化为列数是28*28的，行数有torch系统决定（一般就是1了，因为列数占据了所有）
            # 使用net.forward进行误差反向传播，得到当前这个图片经过神经网络之后的特征向量变化
            outputs = net.forward(x.view(-1, 28*28))
            # 获取经过神经网络之后概率最大的输出下标，即获取0-9最大的概率
            for i, output in enumerate(outputs):
                # 如果经过神经网络之后最大的概率是这个样本数据的真正数字
                if torch.argmax(output) == y[i]:
                    # 正确数加1
                    n_correct += 1
                n_total += 1
    # 返回正确率
    return n_correct/n_total


def train(epochs, lr, net, test_iter, train_iter):
    # 优化函数
    optimizer = torch.optim.Adam(net.parameters(), lr)
    # 进行训练
    for epoch in range(epochs):
        # 训练
        for (x, y) in train_iter:
            # 梯度置0，防止梯度累加
            net.zero_grad()
            # 输入特征正向传播，图片传入神经网络进行预测
            # 图片输出模型的时候将图片整成[1, 28*28]的格式
            outputs = net.forward(x.view(-1, 28 * 28))
            # 计算损失
            loss = nn.functional.nll_loss(outputs, y)  # nll_loss对数损失函数，为了匹配log_softmax
            # 梯度反向传播
            loss.backward()
            # 更新网络次数W，b
            optimizer.step()
        # 没训练完一次，预测一次准确率
        print(f'epoch {epoch}, accuracy: {evaluate(test_iter, net)}')


def test(net, test_iter, test_size):
    # 获取第一个批次的数据
    first_batch = next(iter(test_iter))
    x, y = first_batch[0], first_batch[1]
    # 随机生成len(test_iter)以内的test_size个随机数
    random_index_list = [random.randint(0, len(test_iter)) for _ in range(test_size)]
    random_inputs, random_labels = x[random_index_list], y[random_index_list]
    for i in range(test_size):
        predict = torch.argmax(net.forward(random_inputs[i][0].view(-1, 28 * 28)))
        plt.figure(random_labels[i])
        plt.imshow(random_inputs[i][0].reshape(28, 28), cmap='gray')
        plt.title('prediction: ' + str(int(predict)))
        plt.axis('off')  # 不显示坐标轴
    plt.show()


def main():
    # 获取数据
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 声明神经网络
    net = Net()

    # 超参数定义， 梯度下降学习率和训练批量
    epochs = 5
    lr = 0.001

    # 首先输出一次，为训练之前的准确率
    print(f' init accuracy: {evaluate(test_iter, net)}') # 0.1 因为...

    # 训练
    train(epochs, lr, net, test_iter, train_iter)

    # 测试
    test_size = 5
    test(net, test_iter, test_size)

    # for (n, (x, _)) in enumerate(test_iter):
    #     if n > 0:
    #         break
    #     print(type(x))
    #     predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
    #     plt.figure(n)
    #     plt.imshow(x[0].reshape(28, 28), cmap='gray')
    #     plt.title('prediction: ' + str(int(predict)))
    #     plt.axis('off')  # 不显示坐标轴
    # plt.show()
    pass


if __name__ == '__main__':
    main()
