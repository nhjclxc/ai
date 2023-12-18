#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/18 16:27
# Module    : test1.py
# explain   :

import random
import torch
# from d2l import torch as d2l
from d2l_local.d2l_local import torch as d2l

lines = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
tokens = [token for line in lines for token in line]
vocab = d2l.Vocab(tokens)
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
# d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',xscale='log', yscale='log')



bigram_tokens = [pair for pair in zip(tokens[:-1], tokens[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])


trigram_tokens = [triple for triple in zip(tokens[:-2], tokens[1:-1], tokens[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])



bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
# d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
#             ylabel='frequency: n(x)', xscale='log', yscale='log',
#             legend=['unigram', 'bigram', 'trigram'])
#


# 读取长序列数据

# 随机采样
'''
(在随机采样中，每个样本都是在原始的长序列上任意捕获的子序列。) 在迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻。 对于语言建模，目标是基于到目前为止我们看到的词元来预测下一个词元， 因此标签是移位了一个词元的原始序列。
下面的代码每次可以从数据中随机生成一个小批量。 在这里，参数batch_size指定了每个小批量中子序列样本的数目， 参数num_steps是每个子序列中预定义的时间步数。
'''
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # num_steps类似于序列模型里面的tau
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    # random.randint(0, num_steps - 1)表示从0到num_steps-1随机取一个整数，从这个位置开始对corpus进行切割以便后面进行随机采样
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps #也就是由多少对 特征序列X和对应的标签Y
    # 长度为num_steps的子序列的起始索引
    # initial_indices里面保存每一个特征和标签的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)



# 下面我们[生成一个从0到34的序列]。 假设批量大小为2，时间步数为5，这意味着可以生成 ⌊(35-1)/5⌋ = 6个“特征－标签”子序列对。
# 如果设置小批量大小为2，我们只能得到3个小批量。
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


print('seq_data_iter_sequential')
'''
在迭代过程中，除了对原始序列可以随机抽样外， 我们还可以[保证两个相邻的小批量中的子序列在原始序列上也是相邻的]。 这种策略在基于小批量的迭代过程中保留了拆分的子序列的顺序，因此称为顺序分区。
'''
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

# 基于相同的设置，通过顺序分区[读取每个小批量的子序列的特征X和标签Y]。 通过将它们打印出来可以发现： 迭代期间来自两个相邻的小批量中的子序列在原始序列中确实是相邻的。
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)