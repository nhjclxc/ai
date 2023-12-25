#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/25 19:11
# Module    : test04_cpg_rec.py
# explain   :



# 1. 数据构造

import torch
from torch_geometric.data import Data, Dataset
from test03_dot2pyg import *


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

dot_data_list = [dot_data, dot_data2, data_cpg2pyg]

class CustomDataset(Dataset):
    def __init__(self, num_graphs, num_node, dim_feature, num_edge = None):
        super().__init__()
        self.num_graphs = num_graphs
        self.num_node = num_node
        self.data_list = [self.generate_random_graph(num_node, dim_feature, num_edge) for _ in range(num_graphs)]
        # self.data_list = cpg2pyg(dot_data_list)

    @staticmethod
    def generate_random_graph(num_node, feature_dim = 16, num_edge = None):
        # 随机生成节点特征
        num_nodes = int(torch.randint(1, num_node + 1, ()))
        x = torch.randn(num_nodes, feature_dim)  # 假设节点特征维度为 16
        if num_edge == None:
            num_edge = num_nodes * 2
        edge_index = torch.randint(0, num_nodes, (2, num_edge))  # 随机生成边
        return Data(x=x, edge_index=edge_index)

    def __len__(self):
        return self.num_graphs

    def len(self):
        return self.num_graphs

    def get(self, idx):
        return self.data_list[idx]

# 创建一个包含多个图的数据集
dim_feature = 16
# custom_dataset = CustomDataset(5, 30, dim_feature= dim_feature)

# 打印生成的图数据
# for i in range(len(custom_dataset)):
#     data = custom_dataset.get(i)
#     print(f"Graph {i + 1} data:", data)
#     print(data.x)
#
# for i, data in enumerate(custom_dataset):
#     print(f"enumerate Graph {i + 1} data:", data)


# 生成两个图数据
# graph_data_1 = CustomDataset.generate_random_graph(20)  # 生成包含 20 个节点的图数据
# graph_data_2 = CustomDataset.generate_random_graph(30)  # 生成包含 30 个节点的图数据
# print("Graph 1 data:", graph_data_1)
# print("Graph 2 data:", graph_data_2)

# edge_index = torch.tensor([[0, 0, 0, 1],
#                             [1, 2, 3, 2]], dtype=torch.long)
# x = torch.randn(4, 16)  # 4个节点，每个节点的特征向量大小为16
# data = Data(x=x, edge_index=edge_index)

# data = custom_dataset
data, node_vocab = cpg2pyg(dot_data_list)
# print(data)
print(len(node_vocab))

# 2. 定义 GAT 模型：
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNodeRecommendation(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATNodeRecommendation, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.relu(x)


# 3. 模型训练：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATNodeRecommendation(in_channels=16, out_channels=len(node_vocab), heads=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fun = nn.CrossEntropyLoss(reduction='mean') # F.nll_loss

def train(model, train_dataset, optimizer, loss_fun, epochs = 100):
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()
            out = model(data)
            tag = torch.tensor(data.y)
            loss = loss_fun(out, tag)  # 通过loss function指定节点
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_sum += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}], Loss: {loss_sum/len(train_dataset)}')
    print('训练结束')

train(model, data, optimizer, loss_fun, 100)

# 4. 模型测试：
model.eval()

def beam_search(model, data, k, width, depth):
    # 初始化初始节点
    initial_nodes = [0]  # 这里以第一个节点作为初始节点
    paths = {tuple(initial_nodes): 1.0}  # 存储每个路径及其对应的概率，初始概率为 1.0

    for _ in range(depth):
        new_paths = {}
        for path, score in paths.items():
            node = path[-1]  # 当前路径的最后一个节点
            with torch.no_grad():
                logits = model(data)  # 获取预测值
                # probabilities = F.softmax(logits[node], dim=0)  # 计算节点的预测概率
                try:
                    probabilities = F.softmax(logits[node], dim=0)  # 计算节点的预测概率
                except IndexError as ie:
                    print(data, len(logits), node, path)
                    # sys.exit(-1)

            top_k_prob, top_k_indices = torch.topk(probabilities, width)  # 获取 top k 的概率和索引
            for i in range(width):
                new_path = tuple(list(path) + [top_k_indices[i].item()])  # 扩展路径
                new_paths[new_path] = score * top_k_prob[i].item()  # 更新路径和概率
        paths = dict(sorted(new_paths.items(), key=lambda item: item[1], reverse=True)[:width])  # 保留 top k 的路径
    top_k_paths = list(paths.keys())[:k]  # 选取前 k 个路径
    return top_k_paths


# 测试集数据# 定义你的测试集数据
test_data = data[1]

# 选择一个节点作为初始节点，假设是第一个节点
initial_node = 0

# 使用 beam_search 获取前 k 个预测节点索引
top_k_paths = beam_search(model, test_data, k=5, width=2, depth=3)

# 输出前 k 个预测节点索引
print("Top k paths:")
for path in top_k_paths:
    print(path)
    print(node_vocab.to_tokens([path]))

