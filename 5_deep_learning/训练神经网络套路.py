#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/23 0:30
# Module    : 神经网络套路.py
# explain   :


import torchvision.datasets
import torch.utils.data
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim

'''
原文链接：https://blog.csdn.net/qq_33952811/article/details/123301500
step1. 加载数据
step2. 定义网络
step3. 定义损失函数和优化器
step4. 训练网络，循环4.1到4.6直到达到预定epoch数量
– step4.1 加载数据
– step4.2 初始化梯度
– step4.3 计算前馈
– step4.4 计算损失
– step4.5 计算梯度
– step4.6 更新权值
step5. 保存权重
step6. 加载模型

'''

# 训练一个分类器

def train():
    '''训练'''
    '''1.加载数据'''
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = (
        'plane', 'car', 'bird', 'cat','deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    '''2.定义网络'''
    Net = LeNet()

    '''3.定义损失函数and优化器'''
    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.SGD(Net.parameters(),lr=1e-3,momentum=0.9)

    '''cuda加速'''
    device = ['gpu' if torch.cuda.is_available() else 'cpu']
    if device == 'gpu':
        criterion.cuda()
        Net.to(device)
        # Net.cuda()      #多GPU 请用 DataParallel方法

    '''4.训练网络'''
    print('开始训练')
    for epoch in range(3):
        runing_loss = 0.0

        for i,data in enumerate(trainloader,0):
            inputs,label = data             #1.数据加载
            if device == 'gpu':
                inputs = inputs.cuda()
                label = label.cuda()
            optimizer.zero_grad()           #2.初始化梯度
            output = Net(inputs)            #3.计算前馈
            loss = criterion(output,label)  #4.计算损失
            loss.backward()                 #5.计算梯度
            optimizer.step()                #6.更新权值

            runing_loss += loss.item()
            if i % 20 == 19:
                print('epoch:',epoch,'loss',runing_loss/20)
                runing_loss = 0.0

    print('训练完成')
    '''4.保存模型参数'''
    torch.save(Net.state_dict(),'cifar_AlexNet.pth')

    '''5.加载模型'''
    model = torch.load('model.pth')


if __name__=='__main__':
    train()
