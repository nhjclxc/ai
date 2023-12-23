#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/23 19:17
# Module    : test01_embedding.py
# explain   :

#
# 安装 graphviz 库
# !pip install graphviz

import graphviz


def test1():
    # 创建图的表示
    dot = '''
    graph graphname {
        cat -- dog;
        dog -- mouse;
        mouse -- mouse2;
        mouse2 -- mouse3;
        mouse3 -- mouse4;
    }
    '''

    # 使用 Graphviz 库将 Dot 语言表示的图渲染为图形对象
    graph = graphviz.Source(dot, format='png')

    # 显示图形
    graph.view()

    # 将图转换为邻接矩阵
    adjacency_matrix = [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ]

    # 输出邻接矩阵
    print("邻接矩阵:")
    for row in adjacency_matrix:
        print(row)
    pass


def test2():
    # 创建一个简单的图的邻接列表示例
    dot = {
        'cat': ['dog'],
        'dog': ['cat', 'mouse'],
        'mouse': ['dog', 'mouse2'],
        'mouse2': ['mouse', 'mouse3'],
        'mouse3': ['mouse2', 'mouse4'],
        'mouse4': ['mouse3']
    }

    # 输出邻接表
    for node in dot:
        print(f"{node}: {dot[node]}")

    # 创建 Dot 语言表示的图
    dot_code = 'digraph G {\n'
    for node in dot:
        for neighbor in dot[node]:
            dot_code += f'    {node} -> {neighbor};\n'
    dot_code += '}'

    # 渲染图形对象
    graph = graphviz.Source(dot_code, format='png')

    # 显示图形
    graph.view()

    pass


def test3():

    dot = '''
    digraph foo_CPG {
    // graph-vertices
        v1 [label="5:methodDeclaration:void foo()";]
        v2 [label="5:modifier:";]
        v3 [label="5:return:void";]
        v4 [label="5:identifier:foo";]
        v5 [label="5:MethodBlock:null";]
        v6 [label="6:localVariableDeclaration:int x = source();";]
        v7 [label="6:variableDeclaratorId:int x ";]
        v8 [label="6:variableInitializer:= source();";]
        v9 [label="6:type:int";]
        v10 [label="6:identifier:x";]
        v11 [label="7:ifStatement:if( x < Max)";]
        v12 [label="7:parExpression:(x < Max)";]
        v13 [label="7:then:null";]
        v14 [label="8:localVariableDeclaration:int y = 2 * x;";]
        v15 [label="8:variableDeclaratorId:int y ";]
        v16 [label="8:variableInitializer:= 2 * x;";]
        v17 [label="8:type:int";]
        v18 [label="8:identifier:y";]
        v19 [label="9:statementExpression:sink(y)";]
        v20 [label="endIf:end if";]
    // graph-edges
        v1 -> v2 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v1 -> v3 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v1 -> v4 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v1 -> v5 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v1 -> v6 [arrowhead=empty, color=red, style=dashed, label="FLOWS_TO"];
        v1 -> v6 [arrowhead=empty, color=blue, style=dashed, label="CDG_EPSILON"];
        v5 -> v6 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v6 -> v7 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v6 -> v9 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v6 -> v10 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v7 -> v8 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v6 -> v11 [arrowhead=empty, color=red, style=dashed, label="FLOWS_TO"];
        v1 -> v11 [arrowhead=empty, color=blue, style=dashed, label="CDG_EPSILON"];
        v5 -> v11 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v11 -> v12 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v11 -> v13 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v11 -> v14 [arrowhead=empty, color=red, style=dashed, label="FLOWS_TO_TRUE"];
        v11 -> v14 [arrowhead=empty, color=blue, style=dashed, label="CDG_TRUE"];
        v13 -> v14 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v14 -> v15 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v14 -> v17 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v14 -> v18 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v15 -> v16 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v14 -> v19 [arrowhead=empty, color=red, style=dashed, label="FLOWS_TO"];
        v11 -> v19 [arrowhead=empty, color=blue, style=dashed, label="CDG_EPSILON"];
        v13 -> v19 [arrowhead=empty, color=green, style=dashed, label="AST"];
        v19 -> v20 [arrowhead=empty, color=red, style=dashed, label="FLOWS_TO"];
        v11 -> v20 [arrowhead=empty, color=red, style=dashed, label="FLOWS_TO_FALSE"];
        v14 -> v19 [style=bold, label=" (y)"];
        v6 -> v11 [style=bold, label=" (x)"];
        v6 -> v14 [style=bold, label=" (x)"];
     // end-of-graph
    }
    '''

    # 使用 Graphviz 库将 Dot 语言表示的图渲染为图形对象
    graph = graphviz.Source(dot, format='png')

    # 显示图形
    graph.view()

    pass


def test4():
    '''
    在这个例子中，我们从邻接表中提取边，然后将边转换为 PyTorch 的 LongTensor，最后创建了一个包含边信息的 Data 对象。
    请注意，如果你有其他节点特征（如 x 和 y），你可以添加到 Data 对象中以完善图的表示。
    '''

    # 假设有一个邻接表 adj_list 表示的图：
    # 假设有一个邻接表表示的图
    adj_list = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2, 4],
        4: [3]
    }

    # 可以将它转换为 PyTorch Geometric 的 Data 对象：
    import torch
    from torch_geometric.data import Data

    # 构建邻接表
    adj_list = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2, 4],
        4: [3]
    }

    # 从邻接表中提取边
    edge_index = []
    for src, dsts in adj_list.items():
        edge_index.extend([(src, dst) for dst in dsts])

    # 转换边为 PyTorch 的 LongTensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 创建一个 Data 对象
    data = Data(edge_index=edge_index)

    # 输出 Data 对象
    print(data)

    pass

import re
import numpy as np

def dot_to_adjacency_matrix(dot_string):
    # 正则表达式匹配，提取节点和边的信息
    node_pattern = re.compile(r'(\w+) \[label="[^"]+";]')
    edge_pattern = re.compile(r'(\w+) -> (\w+) \[label="[^"]+";]')


    # 提取所有节点和边
    nodes = set(node_pattern.findall(dot_string))
    edges = edge_pattern.findall(dot_string)

    # 创建节点到索引的映射
    node_to_index = {node: i for i, node in enumerate(sorted(nodes))}

    # 创建邻接矩阵
    size = len(nodes)
    adj_matrix = np.zeros((size, size), dtype=int)

    # 填充邻接矩阵
    for src, dst in edges:
        src_index = node_to_index[src]
        dst_index = node_to_index[dst]
        adj_matrix[src_index, dst_index] = 1

    return adj_matrix

# 测试函数
dot_str = '''
digraph graphname {
    v1 [label="methodDeclaration:void foo();"]
    v2 [label="modifier:"]
    v3 [label="return:void"]
    v4 [label="identifier:foo"]
    v5 [label="MethodBlock:null"]
    v1 -> v2 [label="AST"];
    v1 -> v3 [label="AST"];
    v1 -> v4 [label="AST"];
    v1 -> v5 [label="AST"];
}
'''

adjacency_matrix = dot_to_adjacency_matrix(dot_str)
print(adjacency_matrix)


if __name__ == '__main__':
    # test1()

    # test2()
    #
    # test3()


    # test4()

    pass