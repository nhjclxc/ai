#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/25 10:55
# Module    : test03_dot2pyg.py
# explain   :


import re
import torch
from torch_geometric.data import Data
from itertools import chain

from test_cpg2apis.Vocab import Vocab

# DOT 格式图数据
dot_data = '''
digraph foo_CPG {
v1 [label="methodDeclaration:void foo()";]
v5 [label="MethodBlock:null";]
v6 [label="localVariableDeclaration:int x = source();";]
v7 [label="variableDeclaratorId:int x ";]
v8 [label="variableInitializer:= source();";]
v11 [label="ifStatement:if( x < Max)";]
v12 [label="parExpression:(x < Max)";]
v13 [label="then:null";]
v14 [label="localVariableDeclaration:int y = 2 * x;";]
v15 [label="variableDeclaratorId:int y ";]
v16 [label="variableInitializer:= 2 * x;";]
v19 [label="statementExpression:sink(y)";]
v20 [label="endIf:end if";]
v21 [label="methodDeclaration:int source()";]
v25 [label="MethodBlock:null";]
v26 [label="returnStatement:return 0;";]
v27 [label="methodDeclaration:void sink(int y)";]
v1 -> v5 [label="AST"];
v1 -> v6 [label="FLOWS_TO"];
v1 -> v6 [label="CDG_EPSILON"];
v5 -> v6 [label="AST"];
v6 -> v7 [label="AST"];
v7 -> v8 [label="AST"];
v6 -> v11 [label="FLOWS_TO"];
v1 -> v11 [label="CDG_EPSILON"];
v5 -> v11 [label="AST"];
v11 -> v12 [label="AST"];
v11 -> v13 [label="AST"];
v11 -> v14 [label="FLOWS_TO_TRUE"];
v11 -> v14 [label="CDG_TRUE"];
v13 -> v14 [label="AST"];
v14 -> v15 [label="AST"];
v15 -> v16 [label="AST"];
v14 -> v19 [label="FLOWS_TO"];
v11 -> v19 [label="CDG_EPSILON"];
v13 -> v19 [label="AST"];
v19 -> v20 [label="FLOWS_TO"];
v11 -> v20 [label="FLOWS_TO_FALSE"];
v21 -> v25 [label="AST"];
v21 -> v26 [label="FLOWS_TO"];
v21 -> v26 [label="CDG_EPSILON"];
v25 -> v26 [label="AST"];
v6 -> v21 [label="Call"];
v26 -> v6 [label="Return"];
v19 -> v27 [label="Call"];
v14 -> v19 [label=" (y)"];
v6 -> v11 [label=" ($THIS.Max)"];
v6 -> v11 [label=" (x)"];
v6 -> v14 [label=" (x)"];
}

'''

dot_data2 = '''
digraph source_CPG {
// graph-vertices
v1 [label="methodDeclaration:int source()";]
v2 [label="modifier:private";]
v3 [label="return:int";]
v4 [label="identifier:source";]
v5 [label="MethodBlock:null";]
v6 [label="returnStatement:return 0;";]
// graph-edges
v1 -> v2 [label="AST"];
v1 -> v3 [label="AST"];
v1 -> v4 [label="AST"];
v1 -> v5 [label="AST"];
v1 -> v6 [label="FLOWS_TO"];
v1 -> v6 [label="CDG_EPSILON"];
v5 -> v6 [label="AST"];
// end-of-graph
}
'''


def test1():
    pattern = re.compile(r'(\w+) -> (\w+) \[label="([^"]+)"\];')

    matches = re.findall(pattern, dot_data)

    for match in matches:
        src = match[0]
        dst = match[1]
        label = match[2]

        print(f"666 Source: {src}, Destination: {dst}, Label: {label}")

    # 解析节点和边信息
    node_pattern = re.compile(r'(\w+) \[label="([^"]+)";]')
    edge_pattern = re.compile(r'(\w+) -> (\w+) \[label="([^"]+)";]')

    node_map = node_pattern.findall(dot_data)
    print('node_map = ', node_map)
    edges = edge_pattern.findall(dot_data)

    # 创建节点和边缘的映射及节点标签
    edge_index = [(node_map[src], node_map[dst]) for src, dst in edges]

    # 创建 PyG 图数据
    x = torch.tensor([])  # 将节点标签作为节点特征
    # data = Data(x=x, edge_index=torch.tensor(edge_index).t(), edge_attr=)
    # print(data)

    pass


def get_node_map_list(dot_str_data):
    """
        解析dot语言的节点
    :param dot_str_data: dot图数据
    :return :
        node_list: 所有的节点集合 <br/>
        return : dot_node_map_list dot_str_data里面所有的图节点编号与节点的对应字典
    """
    node_list = []
    node_map_list = []
    if isinstance(dot_str_data, (list, tuple)):
        for dot_str in dot_str_data:
            get_node(node_map_list, dot_str, node_list)
    else:
        get_node(node_map_list, dot_str_data, node_list)

    return node_list, node_map_list


def get_node(node_map_list, dot_str_data, node_list):
    node_pattern = re.compile(r'(\w+) \[label="([^"]+)";]')
    node_map = node_pattern.findall(dot_str_data)
    # for _, value in node_map:
        # node_list.append(value)
    node_list.extend(value for _, value in node_map)
    node_dict = {key: value for key, value in node_map}
    node_map_list.append(node_dict)


def test2_node():

    dot_data_list = [dot_data, dot_data2]

    # 解析dot语言的所有标签
    node_list, dot_node_map_list = get_node_map_list(dot_data_list)
    print(len(node_list), node_list)
    print(len(dot_node_map_list), dot_node_map_list)

    # 构建节点词汇表
    node_vocab = Vocab(node_list)
    # print(list(vocab._token2idx.items()))

    length = len(node_vocab._token2idx.items())
    for i in range(length):
        token = node_vocab.to_tokens([i])
        print(i, ' 文本:', token, '索引:', node_vocab[token])

    # 节点嵌入

    pass


def get_edge_map_list(dot_str_data):
    """
        解析dot语言的节点
    :param dot_str_data: dot图数据
    :return : [[{'src' : '...', 'dst' : '...', 'label' : '...'}]]
            第一维度是表示每一个dot图，第二维度是这个图的每一条边的字典
    """
    edge_node_label_map_list = []

    if isinstance(dot_str_data, (list, tuple)):
        for dot_str in dot_str_data:
            edge_node_label_map_list.append(get_edge(dot_str))
    else:
        edge_node_label_map_list.append(get_edge(dot_str_data))

    edge_label_list = [entry['label'] for entry in list(chain.from_iterable(edge_node_label_map_list))]
    # print('label', labels)

    return edge_node_label_map_list, edge_label_list


def get_edge(dot_str_data):
    # edge_pattern = re.compile(r'(\w+) -> (\w+) \[label="([^"]+)"\];')
    edge_pattern = re.compile(r'(\w+) -> (\w+) \[label="([^"]+)"];')
    edge_map = []
    matches = re.findall(edge_pattern, dot_str_data)
    for match in matches:
        # src = match[0]
        # dst = match[1]
        # label = match[2]
        # print(f"Source: {src}, Destination: {dst}, Label: {label}")
        source, target, label = match
        result_dict = {'source': source, 'target': target, 'label': label}
        edge_map.append(result_dict)
    return edge_map


def test3_edge():
    dot_data_edge_map_list, edge_label_list = get_edge_map_list(dot_data)
    print(len(dot_data_edge_map_list), dot_data_edge_map_list)
    print(edge_label_list)

    pass



def test_embedding():
    word_to_ix = {"hello": 0, "world": 1}
    # 2 words in vocab，词汇表大小为2, 16 dimensional embeddings，嵌入的维度为 16
    embeds = torch.nn.Embedding(2, 16)
    lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    print(f'lookup_tensor = {lookup_tensor}')
    hello_embed = embeds(lookup_tensor)
    print(f'hello_embed = {hello_embed}')

    lookup_tensor2 = torch.tensor([0, 1], dtype=torch.long)
    print(embeds(lookup_tensor2))

    lookup_tensor2 = torch.tensor([0, 1], dtype=torch.long)
    print(embeds(lookup_tensor2))


    pass


def cpg2pyg(dot_data_list):
    """
        将dot格式的cpg图结构转化为pyg格式的图结构
    :param dot_data_list: dot格式图数据
    :return: pyg格式的图结构
    """
    embedding_size = 16

    # 1. 解析节点
    # 解析dot语言的所有标签
    node_list, node_map_list = get_node_map_list(dot_data_list)
    # 构造节点词汇表
    node_vocab = Vocab(node_list)
    # 节点嵌入
    node_embeds = torch.nn.Embedding(len(node_vocab), embedding_size)

    # 2. 解析边
    edge_node_label_map_list, edge_label_list = get_edge_map_list(dot_data_list)
    # 构造边词汇表
    edge_label_vocab = Vocab(edge_label_list)
    # 边嵌入
    edge_embeds = torch.nn.Embedding(len(edge_label_vocab), embedding_size)


    # 3. 构造pyg数据
    '''
    torch_geometric.data.Data是PyTorch
    Geometric中的一个核心数据结构，用于表示图数据。它包含以下属性：
        x：节点特征矩阵，维度为(num_nodes, num_node_features)。
        edge_index：边索引列表，描述了节点之间的连接关系。对于有向图，形状为(2,num_edges)，其中第一行是源节点，第二行是目标节点。对于无向图，应该包含双向的边索引。
        edge_attr：边特征矩阵，维度为(num_edges, num_edge_features)，用于存储边的特征信息。
    '''
    all_dataset = []
    for i, dot_str in enumerate(dot_data_list):
        # 每一趟就是构造一个图

        # 获取这张图的所有节点
        node_map = node_map_list[i]

        # 3.1 图节点获取嵌入
        # 转化字符串为字典
        node_list = list(node_map.values())
        node_index_map = {}
        for key, value in node_map.items():
            node_index_map[value] = len(node_list)
            node_list.append(value)

        x = node_embeds(torch.tensor(node_vocab[node_list], dtype=torch.long))
        # print('获取嵌入', x)

        # 3.2 构造边  边属性构造
        # 获取这个图的每一条边
        edge_node_label_map = edge_node_label_map_list[i]

        # 遍历每一条边
        edge_index = torch.empty(2, 0, dtype=torch.long) #空的二维张量
        edge_attr = torch.empty(0, embedding_size, dtype=torch.float) #空的二维张量
        for map in edge_node_label_map:
            # map['source']获取这个图的map里面source的dot节点vx
            # node_map[map['source']]获取这个图里面的dot节点vx对应Java代码解析出来的抽象语法树节点
            edge_source_node = node_map[map['source']]
            # 获取这个节点的嵌入
            source_node_index = node_index_map[edge_source_node]
            target_node_index = node_index_map[node_map[map['target']]]

            edge = torch.tensor([[source_node_index, target_node_index]])# 调整 edge 的形状为二维张量
            edge_index = torch.cat((edge_index, edge.t()), dim=1)

            # 边属性构造
            # map['label'] 表示获取label的标签值
            attr = edge_embeds(torch.tensor(edge_label_vocab[map['label']], dtype=torch.int)).reshape(1, embedding_size)
            edge_attr = torch.cat((edge_attr, attr), dim=0)

        # [Data(x=[5, 16], edge_index=[2, 7], edge_attr=[7, 16])]
        # x=[5, 16]表示节点：5表示有5个节点，16表示每一个节点有16个特征向量
        # edge_index=[2, 7]表示边：2表示源src和tag目标，7表示有7条边
        # edge_attr=[7, 16]表示边属性，7表示有7条边，16表示每一条边是一个包含16维的特征向量
        data = Data(x=x, edge_index=edge_index, edge_attr = edge_attr)
        # print(data.x)
        # print(data.edge_index)
        all_dataset.append(data)

        # 测试  构造一个含有节点，边和边特征的pyg图结构
        # # 例子：3个节点，3条边，每条边有2个特征
        # x = torch.tensor([0, 1, 1], dtype=torch.long)  # 示例的节点索引
        # edge_index = torch.tensor([ [0, 1, 1],
        #                             [1, 0, 2]], dtype=torch.long)  # 示例的边索引
        # # 每一条边有两个特征，
        # # 第0维：即行索引表示的是第几条边，也就是edge_index的第几列
        # # 第1维：即每一行表示的是这一条边的边特征属性，张量
        # edge_attr = torch.tensor([  [0.1, 0.2],
        #                             [0.3, 0.4],
        #                             [0.5, 0.6]], dtype=torch.float)  # 示例的边属性
        # Data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return all_dataset


def test4_cpg2pyg():
    data_cpg2pyg = '''
    digraph source_CPG {
    v1 [label="node1";]
    v2 [label="node2";]
    v3 [label="node3";]
    v4 [label="node4";]
    v5 [label="node5";]

    v1 -> v2 [label="PDG"];
    v1 -> v3 [label="PDG"];
    v1 -> v4 [label="CFG"];
    v1 -> v5 [label="CFG"];
    v2 -> v3 [label="AST"];
    v4 -> v5 [label="AST"];
    v4 -> v5 [label="PDG"];
    }
    '''

    dot_data_list = [data_cpg2pyg]
    dot_data_list = [dot_data, dot_data2]
    dot_data_list = [dot_data, dot_data2, data_cpg2pyg]

    print(cpg2pyg(dot_data_list))

    pass


if __name__ == '__main__':

    # test1()

    # test2_node()

    # test3_edge()

    # test_embedding()

    test4_cpg2pyg()



    pass



