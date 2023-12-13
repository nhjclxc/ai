#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/11 13:32
# Module    : test.py
# explain   :
import torch


def test1():
    # x = torch.tensor([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    # print(x)
    # print(id(x))
    #
    # print(x*x)

    x = torch.arange(12).reshape(3, 4)
    print(x)
    print(torch.sum(x, dim=0))
    print(torch.sum(x, dim=1))
    print(x.sum(dim=1))

    print('\n\n三维')
    x = torch.arange(24).reshape(2, 3, 4)
    print(x, x.shape)

    print()
    x0 = torch.sum(x, dim=0)
    print(x0, x0.shape)

    print()
    x1 = torch.sum(x, dim=1)
    print(x1, x1.shape)

    print()
    x2 = torch.sum(x, dim=2)
    print(x2, x2.shape)

    pass


def test2():
    A = torch.arange(12).reshape(3, 4)
    print(A)
    print(A.T)  # 转置

    B = torch.arange(12).reshape(3, 4)
    print(A == B)
    print(A * B)
    print(2 * A)

    print(A.sum())
    print(A.sum(axis=0))
    print(A.sum(axis=0, keepdims=True))

    A = torch.arange(12).reshape(3, 4)
    B = torch.arange(12).reshape(4, 3)
    print(A)
    print(B)

    # 向量点积
    print(A[0, :], (B[:, 0]))
    print(A[0, :].dot(B[:, 0]))

    # 矩阵乘向量
    print(B[:, 0])
    print(torch.mv(A, B[:, 0]))

    # 矩阵乘法
    print(torch.mm(A, B))
    print(torch.matmul(A, B))

    pass


def test():
    # 1.证明A的转置的转置时A
    A = torch.arange(12).reshape(3, 4)
    B = A.T
    C = B.T
    print(A)
    print(B)
    print(C)
    print(A == C)

    # 2  A.T + B.T = (A+B).T
    A = torch.arange(12).reshape(3, 4)
    B = torch.arange(12).reshape(3, 4)
    AB_T = (A + B).T
    ABT = A.T + B.T
    print(AB_T == ABT)

    # 3. 对方阵A，那么A+A.T总是对称的吗？
    A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(A + A.T)

    # 4. len()用于返回张量的第一维度的大小（总是返回张量第一维度的大小）
    A = torch.arange(24).reshape(2, 3, 4)
    print(len(A))
    B = torch.arange(12).reshape(3, 4)
    print(len(B))

    #
    A = torch.arange(12).reshape(3, 4)
    # print(A / A.sum(axis = 1))  # A.sum(axis = 1))没有使用keepdim=True这个矩阵被降为了
    print(A / A.sum(axis = 1, keepdim=True))

    #
    A = torch.arange(24).reshape(2, 3, 4)
    print('\n0')
    print(A.sum(axis = 0))
    print(A.sum(axis = 0, keepdim=True))
    print('\n1')
    print(A.sum(axis = 1))
    print(A.sum(axis = 1, keepdim=True))
    print('\n2')
    print(A.sum(axis = 2))
    print(A.sum(axis = 2, keepdim=True))


    #
    print(torch.norm(torch.ones((4, 9))))
    A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float32)
    print(torch.norm(A))



    pass


if __name__ == '__main__':
    # test1()

    # test2()

    test()

    pass
