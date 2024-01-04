import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from test13_gnn_rnn.Vocab import Vocab

import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool




# 测试  构造一个含有节点，边和边特征的pyg图结构
# 例子：3个节点，3条边，每条边有2个特征
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # 示例的节点索引
node_embeds = torch.nn.Embedding(5, 16)
x = node_embeds(y)
edge_index = torch.tensor([ [0, 0, 0, 0, 1, 3],
                            [1, 2, 3, 4, 2, 4]], dtype=torch.long)  # 示例的边索引
# 每一条边有两个特征，
# 第0维：即行索引表示的是第几条边，也就是edge_index的第几列
# 第1维：即每一行表示的是这一条边的边特征属性，张量
edge_attr = torch.tensor([  [0.1, 0.2],
                            [0.3, 0.4],
                            [0.3, 0.4],
                            [0.3, 0.4],
                            [0.3, 0.4],
                            [0.5, 0.6]], dtype=torch.float)  # 示例的边属性
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

embedding_size = 16
vocab = Vocab(nodes, embedding_size=embedding_size)

output_size = len(vocab)

# 定义图神经网络编码器 GAT
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        # 对每一列进行全局池化,将x*n的张量降为1*n的张量便于输入RNN做解码.其中x是当前图的节点个数,n是词汇表大小即列别数
        x = global_mean_pool(x, data.batch)
        return x

# 定义带注意力机制的RNN解码器
class AttentionRNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionRNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, _ = self.rnn(input)
        output = self.fc(output)
        return output

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention_scores, dim=1)

class RNNWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = Attention(hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        attn_weights = self.attention(output[-1], output)
        context = torch.bmm(attn_weights.unsqueeze(0), output.transpose(0, 1)).squeeze(0)
        output = self.fc(context)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# 定义端到端模型
class GraphToSequence(nn.Module):
    def __init__(self, encoder, decoder):
        super(GraphToSequence, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, graph, hidden=None):
        encoded_graph = self.encoder(graph)
        # 假设将编码后的图表示作为输入序列，这里使用相同的表示来生成序列
        output, hidden = self.decoder(encoded_graph.unsqueeze(0), hidden)  # 将图表示扩展为 (1, *encoded_graph.shape)
        return output, hidden  # 移除扩展的维度，得到输出序列


    def beam_search(self, graph, vocab, beam_width=3, max_length=10):
        encoded_graph = self.encoder(graph)
        current_node = 0  # 初始节点为未知节点的索引（或其他合适的起始节点索引）

        # 初始解码器输入为起始节点的嵌入表示
        input_sequence = vocab.get_embedding_by_indices([current_node])

        sequences = [([current_node], 1.0)]  # 初始化序列：起始节点和初始概率
        ended_sequences = []

        while len(ended_sequences) < beam_width and len(sequences) > 0:
            candidates = []

            for seq, score in sequences:
                if len(seq) >= max_length:
                    ended_sequences.append((seq, score))
                    continue

                # 使用解码器进行序列生成
                output_sequence = self.decoder(encoded_graph.unsqueeze(0))  # 解码器预测
                output_probs = nn.functional.softmax(output_sequence, dim=-1)

                # 获取top k个候选节点
                top_k_probs, top_k_indices = torch.topk(output_probs[0, -1], beam_width)
                top_k_probs = top_k_probs.tolist()
                top_k_indices = top_k_indices.tolist()

                for prob, index in zip(top_k_probs, top_k_indices):
                    candidate_seq = seq + [index]
                    candidate_score = score * prob
                    candidates.append((candidate_seq, candidate_score))

            # 选择具有最高分数的 top k 个候选序列
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            sequences = []

            for candidate_seq, candidate_score in candidates:
                if candidate_seq[-1] == vocab.unk:  # 如果候选序列以未知节点结束，将其添加到结束序列
                    ended_sequences.append((candidate_seq, candidate_score))
                else:
                    sequences.append((candidate_seq, candidate_score))

            # 更新下一个时间步的解码器输入为得分最高的序列的最后一个节点的嵌入表示
            # best_sequence, _ = max(ended_sequences, key=lambda x: x[1])
            if len(ended_sequences) > 0:
                best_sequence, _ = max(ended_sequences, key=lambda x: x[1])
                input_sequence = vocab.get_embedding_by_indices([best_sequence[-1]])
                # 其他后续处理
            else:
                # 处理空序列的情况，例如返回默认序列或进行其他操作
                print("No sequences found.")

        return best_sequence



# 创建编码器、解码器和端到端模型实例
encoder = GATEncoder(in_channels=16, hidden_channels=32, out_channels=output_size, num_heads=2)
decoder = RNNWithAttention(input_size=output_size, hidden_size=128, output_size=output_size)  # 假设输出序列长度为10


# 定义模型、损失函数和优化器
model = GraphToSequence(encoder, decoder)  # 假设已经初始化好了 encoder 和 decoder
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器


# 训练模型
def train_model(model, train_loader, target_sequences, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        hidden = model.decoder.init_hidden()
        for graph_data, target_sequence in zip(train_loader, target_sequences):
            optimizer.zero_grad()

            # 前向传播
            output, hidden = model(graph_data, hidden)
            # predicted_sequence = nn.functional.relu(output)
            predicted_sequence = output
            loss = criterion(predicted_sequence, target_sequence)

            # 反向传播和优化
            # loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()
            i += 1

        # 计算每个epoch的平均损失
        epoch_loss = running_loss / i  # len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")



# 示例的图数据数量
num_graphs = 5
graph_dataset = []
target_sequence = []

# 生成示例图数据
for i in range(num_graphs):
    y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # 示例的节点索引
    node_embeds = torch.nn.Embedding(5, 16)
    x = node_embeds(y)
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 3], [1, 2, 3, 4, 2, 4]], dtype=torch.long)  # 示例的边索引
    # 每一条边有两个特征，
    # 第0维：即行索引表示的是第几条边，也就是edge_index的第几列
    # 第1维：即每一行表示的是这一条边的边特征属性，张量
    edge_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.3, 0.4],  [0.3, 0.4],
                                [0.3, 0.4], [0.5, 0.6]], dtype=torch.float)  # 示例的边属性
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph_dataset.append(graph)
    # 生成 3 到 5 个随机数并放入列表
    # num_elements = random.randint(3, 5)
    # random_numbers = [random.randint(0, 10) for _ in range(num_elements)]
    target_sequence.append(torch.randn(1, 21))

# 启用异常检测
torch.autograd.set_detect_anomaly(True)
# 调用训练方法
train_model(model, graph_dataset, target_sequence, criterion, optimizer, num_epochs=10)  # 假设有一个 train_loader 用于加载训练数据

# 使用端到端模型进行前向传播
generated_sequence = model(graph)

print("Generated Sequence Shape:", generated_sequence.shape)

# 使用端到端模型进行集束搜索
start_sequence = model.beam_search(graph, vocab)
decoded_sequence = vocab.to_tokens(start_sequence)
print("Decoded Sequence:", decoded_sequence)
#
