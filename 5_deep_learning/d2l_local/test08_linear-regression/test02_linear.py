#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/11 21:21
# Module    : test02_linear.py
# explain   :线性回归的从零开始实现¶


import random
import torch
from d2l import torch as d2l






# 1. 生成特征和标签
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    # 均值为0，方差为1，里面的(num_examples, len(w))是参数size也就是torch的二维数组的形状即shape(num_examples, len(w))
    # 其中num_examples是行数即样本数，len(w)是列数即每一个样本的特征数
    X = torch.normal(0, 1, (num_examples, len(w)))

    # 实现：y=Xw+b+，torch.matmul(X, w)表示矩阵x和矩阵w乘积
    y = torch.matmul(X, w) + b
    # 加噪声，噪声的产生和x产生类似
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 初始化特征w1，w2和b
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成输入数据和对应的标签
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# 2. 读取数据集
# 回想一下，训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型。 由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。
# 在下面的代码中，我们[定义一个data_iter函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量]。 每个小批量包含一组特征和标签。
def data_iter(batch_size, features, labels):
    # 获取样本个数
    num_examples = len(features)
    # 获取所有索引
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    # 将所有索引再原地打乱
    random.shuffle(indices)
    # 在0到num_examples里面每次跳跃batch_size个索引
    for i in range(0, num_examples, batch_size):
        # 将i往后的batch_size个索引拿出来，最后一块可能是不到batch_size所以就取剩下的
        # batch_indices是一个torch的一位列表，就是这次没拿出来的索引构成的列表
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # 使用 PyTorch 中的张量索引机制
        yield features[batch_indices], labels[batch_indices]

'''
# 使用 PyTorch 中的张量索引机制
x = torch.tensor([11,22,33,44,55])
index = torch.tensor([0,2,3])
print(x[index])
'''

batch_size = 10

# 从迭代器里面拿出一个批量用于测试
iter= data_iter(batch_size, features, labels)
# 生成一个数据看看
test_x, test_y = None,None
for X, y in iter:
    print(X, '\n', y)
    test_x = X
    test_y = y
    break

# 3. 初始化模型参数¶
'''
[在我们开始用小批量随机梯度下降优化我们的模型参数之前]， (我们需要先有一些参数)。 在下面的代码中，我们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0。
'''
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 4. 定义模型
# 定义模型，将模型的输入和参数同模型的输出关联起来。]
def linreg(X, w, b):  #@save
    """线性回归模型"""
    # 实现 y=XW+b
    return torch.matmul(X, w) + b

# 5. [定义损失函数]
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    # 实现 (y^ - y)²/2
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 6. 定义优化算法)
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    # torch.no_grad():表示下面的代码不需要计算梯度
    with torch.no_grad():
        for param in params:
            # 更新参数
            # 实现参数更新的公式：即学习率的梯度下降更新path：超参数/学习率.png
            param -= lr * param.grad / batch_size
            # 超参数置0，防止参数累加
            param.grad.zero_()

# 7. 定义训练超参数
lr = 0.1  # 学习率
num_epochs = 10  # 模型训练迭代次数
net = linreg  # 方便后期模型的变更
loss = squared_loss  # 方便后期损失函数的变更


# 8. 模型训练
for epoch in range(num_epochs):
    # 使用data_iter这个生成式来获取每一个特征与其对应的标签，即每一条输入数据和对应的输出
    # for X, y in data_iter(batch_size, features, labels):
    for X, y in iter:
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


print('\n对模型进行测试')
for x, y in zip(test_x, test_y):
    print('y^', net(x,w,b))
    print('y', y)
    print()

# yyy =  tensor([2.1268])
# yyy^ tensor([2.1225], grad_fn=<AddBackward0>)

print('\n保存模型测试')
# 保存模型参数到文件 保存权重和偏置到文件
torch.save(w, 'weights.pth')  # PyTorch  -> pth
torch.save(b, 'bias.pth')

# 加载模型 加载权重和偏置
loaded_w = torch.load('weights.pth')
loaded_b = torch.load('bias.pth')
# 使用加载的权重和偏置创建模型
# 使用保存的参数进行模型测试
test_xx = torch.tensor([5, 10], dtype=torch.float32)
print(test_xx, net(test_xx, loaded_w, loaded_b))


'''
“true_w = torch.tensor([2, -3.4]) true_b = 4.2”
和“w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) b = torch.zeros(1, requires_grad=True)”
的区别是什么？分别有什么作用

# 通过下面的的true_w和true_b加上正太分布来模拟参数数据
1. `true_w` 和 `true_b`：
   - `true_w` 表示模型的真实权重，是在创建模型时预先定义好的真实参数。
   - `true_b` 则是模型的真实偏置项。

# 下面两个参数w和b就是我们要拟合的参数
2. `w` 和 `b`：
   - `w` 是模型的权重参数，使用正态分布随机初始化，形状为 (2, 1)。
   - `b` 是模型的偏置项参数，初始化为 0。

在模型训练中，优化算法会尝试调整 `w` 和 `b` 的值，使其逼近或接近 `true_w` 和 `true_b`，以使模型能够更好地拟合训练数据。这个过程就是训练过程中所谓的参数更新或优化，通常通过最小化损失函数来实现。
'''