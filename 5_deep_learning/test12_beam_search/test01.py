import torch
import heapq
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import random

# 创建一个简单的图数据
num_nodes = 20
edge_list = [(i, i+1) for i in range(num_nodes - 1)]

x = torch.randn(num_nodes, 16)
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index)

# 定义图神经网络模型
class GATModel(torch.nn.Module):
    def __init__(self):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels=16, out_channels=8, heads=8, dropout=0.6)

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        return x

model = GATModel()

# 训练图神经网络模型... （这里省略了训练过程）

# 使用集束搜索进行节点序列推荐
# 在beam_search函数中添加对vocab的使用
def beam_search(data, model, start_node, sequence_length, beam_width, vocab):
    heap = []
    initial_state = (0, [start_node])
    heapq.heappush(heap, initial_state)

    # GNN模型预测节点
    with torch.no_grad():
        out = model(data)
        node_probs = F.softmax(out, dim=1)

    while heap:
        cost, path = heapq.heappop(heap)
        current_node = path[-1]

        if len(path) == sequence_length:
            return path

        # 获取当前节点的概率分布并采样下一个节点
        probs = node_probs[current_node]

        # 过滤出位于词汇表中的节点
        filtered_probs = probs.clone()
        for idx in range(filtered_probs.size(0)):
            if idx not in vocab:
                filtered_probs[idx] = 0.0

        # 集束搜索中根据概率分布采样下一个节点
        _, indices = filtered_probs.topk(beam_width)
        next_nodes = indices.tolist()

        for node in next_nodes:
            new_path = path + [node]
            new_cost = cost + 1

            if len(heap) < beam_width:
                heapq.heappush(heap, (new_cost, new_path))
            else:
                _, existing_path = heapq.heappop(heap)
                if new_cost < _:
                    heapq.heappush(heap, (new_cost, new_path))
                else:
                    heapq.heappush(heap, (_, existing_path))

    return None

# 设置推荐序列长度、起始节点和词汇表
sequence_length = 5
start_node = random.randint(0, num_nodes - 1)  # 随机选择一个起始节点

# 假设 vocab 是包含节点索引的列表或集合
vocab = [1, 5, 9, 12, 15, 18]  # 这里是一个简单的示例，你需要替换成你的词汇表

# 进行节点序列推荐
recommendation_sequence = beam_search(data, model, start_node, sequence_length, beam_width=3, vocab=vocab)
print("Recommendation Sequence:", recommendation_sequence)
