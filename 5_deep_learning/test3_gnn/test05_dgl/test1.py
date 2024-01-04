#!/usr/bin/pytorchon
# -*- coding: utf-8 -*-
# Autorchor    : LuoXianchao
# Datetime  : 2023/12/29 9:13
# Module    : test1.py
# explain   :


import dgl
import networkx as nx
import torch 
import dgl

def test1():
    # 创建图结构

    # 定义边列表
    src = [0, 1, 2, 3]
    dst = [1, 2, 3, 0]

    # 创建图
    g = dgl.graph((src, dst))

    print(g)

    u = [0, 0, 0, 1]
    v = [1, 2, 3, 3]
    u_v = []
    for u, v in zip(u, v):
        u_v.append((u, v))
    gg = dgl.graph(u_v)
    print(gg)

    pass


def test2():
    # 添加边和节点属性

    u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
    g = dgl.graph((u, v))

    # 每个节点赋值特征
    g.ndata['x'] = torch.randn(g.num_nodes(), 3)  # 长度为3的节点特征
    g.ndata['mask'] = torch.randn(g.num_nodes(), 3)  # 节点可以同时拥有不同方面的特征
    g.edata['x'] = torch.ones(g.num_edges(), dtype=torch.int32)  # 每个边赋值特征
    print(g)

    print(g.ndata["x"][0])
    print(g.edata["x"][0])

    pass


def test3():
    # 批量构造图数据

    u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
    u1, v1 = torch.tensor([0, 1, 2, 3]), torch.tensor([2, 3, 4, 4])
    u2, v2 = torch.tensor([0, 0, 2, 3]), torch.tensor([2, 3, 3, 1])
    g = dgl.graph((u, v))
    g1 = dgl.graph((u1, v1))
    g2 = dgl.graph((u2, v2))
    print(g)
    print(g1)
    print(g2)

    graphs = [g, g1, g2]  # 图的列表
    graph = dgl.batch(graphs)  # 打包成一个batch的大图
    print(graph)

    pass


def test4():
    # 为指定节点嵌入特征


    # 创建一个图，并添加节点特征
    g = dgl.graph([(0, 1), (1, 2)])  # 示例图
    feat_dim = 16  # 特征维度为 16
    g.ndata['feat'] = torch.randn(g.number_of_nodes(), feat_dim)  # 随机初始化节点特征

    # 要进行特征嵌入的节点 ID
    node_id = 0  # 假设要嵌入特征的节点 ID 为 0

    # 生成新的特征嵌入
    new_embedding = torch.randn(feat_dim)  # 假设新的嵌入特征向量

    # 更新指定节点的特征嵌入
    print(g.ndata['feat'][node_id])
    print(g.ndata['feat'][1])
    g.ndata['feat'][node_id] = new_embedding
    print(g.ndata['feat'][node_id])
    print(g.ndata['feat'][1])

    pass


def test5():
    # 为指定边嵌入特征

    # 创建一个图，并添加边特征
    g = dgl.graph([(0, 1), (1, 2)])  # 示例图
    feat_dim = 16  # 特征维度为 16
    g.edata['feat'] = torch.randn(g.number_of_edges(), feat_dim)  # 随机初始化边特征

    # 要进行特征嵌入的边（在示例中为 (0, 1)）
    edge = (0, 1)  # 假设要嵌入特征的边为 (0, 1)

    # 生成新的特征嵌入
    new_embedding = torch.randn(feat_dim)  # 假设新的嵌入特征向量

    # 更新指定边的特征嵌入
    eid = g.edge_ids(edge[0], edge[1])  # 获取边的 ID
    print(g.edata['feat'][eid])
    print(g.edata['feat'][g.edge_ids(1, 2)])  # 没有更新嵌入的边
    g.edata['feat'][eid] = new_embedding
    print(g.edata['feat'][eid])
    print(g.edata['feat'][g.edge_ids(1, 2)])  # 没有更新嵌入的边

    pass


if __name__ == '__main__':

    # test1()

    # test2()
    
    # test3()

    # test4()

    test5()


    pass