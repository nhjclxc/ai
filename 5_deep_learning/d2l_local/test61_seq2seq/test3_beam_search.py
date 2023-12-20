#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/20 15:26
# Module    : test3_beam_search.py
# explain   :

'''
https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24

'''
'''
假设预测的输出是一个1*5的张量（分别对应A,B,C,D,E），现在我要通过束搜索实现概率最高的长度为3的2个序列。输出张量由你随机生成

好的，让我生成一个随机的输出张量，每个值代表 A、B、C、D、E 五个类别的概率分布，并且帮你使用束搜索（beam search）来找到概率最高的两个长度为 3 的序列。

这段代码会生成一个随机的 1x5 输出张量，并使用束搜索算法寻找其中概率最高的两个长度为 3 的序列。你可以将随机输出部分替换为你的实际输出张量
'''
import torch
import numpy as np

# 随机生成一个形状为 (1, 5) 的张量，表示对应五个类别的概率分布
random_output = torch.tensor(np.random.rand(1, 5))

# 假设输出张量为 random_output
print("随机生成的输出张量：")
print(random_output)

# 使用束搜索（beam search）寻找概率最高的两个长度为 3 的序列
def beam_search(output, beam_width=2, sequence_length=3):
    """
        实现束搜索（beam search）
    :param output: 模型的输出张量
    :param beam_width: 其实就是k，也就是每一次要找出几个概率最大的
    :param sequence_length: 搜索的序列长度
    :return:
    """
    # 获取张量形状
    _, num_classes = output.shape

    # 初始化初始序列和其对应的概率值
    initial_sequence = [([], 1.0)]

    # 逐步生成序列
    for step in range(sequence_length):
        candidates = []
        for seq, prob in initial_sequence:
            for i in range(num_classes):
                candidate_seq = seq + [i]
                candidate_prob = prob * output[0, i].item()
                candidates.append((candidate_seq, candidate_prob))

        # 选取概率最高的一部分序列作为下一步的初始序列
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        initial_sequence = ordered[:beam_width]

    # 最终的序列结果
    sequences = [seq for seq, _ in initial_sequence]

    return sequences

# 应用束搜索得到结果
result_sequences = beam_search(random_output, beam_width=2, sequence_length=3)
print("\n概率最高的两个长度为 3 的序列：")
print(result_sequences)




import torch
import numpy as np

# 随机生成一个形状为 (5, 5) 的张量，表示 5 个样本，每个样本有 5 个类别的概率分布
random_output = torch.tensor(np.random.rand(5, 5))

# 假设输出张量为 random_output
print("随机生成的输出张量：")
print(random_output)

# 使用束搜索（beam search）寻找每个样本概率最高的两个长度为 3 的序列
def beam_search(output, beam_width=2, sequence_length=3):
    _, num_classes = output.shape

    result_sequences = []
    for i in range(output.shape[0]):
        # 每个样本进行束搜索
        sample_output = output[i:i+1]  # 取出一个样本的输出张量
        sequences = []
        initial_sequence = [([], 1.0)]  # 初始化初始序列和对应的概率值

        # 逐步生成序列
        for step in range(sequence_length):
            candidates = []
            for seq, prob in initial_sequence:
                for j in range(num_classes):
                    candidate_seq = seq + [j]
                    candidate_prob = prob * sample_output[0, j].item()
                    candidates.append((candidate_seq, candidate_prob))

            # 选取概率最高的一部分序列作为下一步的初始序列
            ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
            initial_sequence = ordered[:beam_width]

        # 最终的序列结果
        sequences = [seq for seq, _ in initial_sequence]
        result_sequences.append(sequences)

    return result_sequences

# 应用束搜索得到结果
result_sequences = beam_search(random_output, beam_width=2, sequence_length=3)
print("\n每个样本概率最高的两个长度为 3 的序列：")
print(result_sequences)




print('\n\n')
import torch
import numpy as np

# 随机生成一个形状为 (5, 6) 的张量，表示 5 个样本，每个样本有 6 个类别的概率分布
# 分别为A,B,C,D,E,F
token_index = ['A', 'B', 'C', 'D', 'E', 'F']

random_output = torch.tensor(np.random.rand(5, 6))

print("(5, 6)随机生成的输出张量：")
print(random_output)
beam_width, sequence_length=2, 3
top_values = None
indices = torch.tensor((sequence_length, beam_width))
target1 = []
target2 = []
# 一行一行地取出数据
i = 0
for _ in range(random_output.shape[0]):
    if i == 0:
        value = random_output[i]
        top_values, top_indices = value.topk(2)
        target1 = torch.tensor([top_indices[0]])
        target2 = torch.tensor([top_indices[1]])
    else:
        value = random_output[i]
        top_val, top_idx1 = value.topk(1)
        top_values[0] = top_values[0] * top_val[0]

        i += 1
        value2 = random_output[i]
        top_val2, top_idx2 = value2.topk(1)
        top_values[1] = top_values[1] * top_val2[0]
        target1 = torch.cat((target1, top_idx1), dim=0)
        target2 = torch.cat((target2, top_idx2), dim=0)

    i += 1
    if i >= random_output.shape[0]-1:
        break

print([token_index[i] for i in target1.tolist()])
print([token_index[i] for i in target2.tolist()])
