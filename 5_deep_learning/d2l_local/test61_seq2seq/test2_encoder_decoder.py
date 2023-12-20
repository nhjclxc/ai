#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/20 10:23
# Module    : test2_encoder_decoder.py
# explain   :


import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


# 1. [实现循环神经网络编码器]
# 我们使用了嵌入层（embedding layer） 来获得输入序列中每个词元的特征向量。
# 嵌入层的权重是一个矩阵， 其行数等于输入词表的大小（vocab_size）， 其列数等于特征向量的维度（embed_size）。
# 对于任意输入词元的索引i ， 嵌入层获取权重矩阵的第i行（从0开始）以返回其特征向量。
# 另外，本文选择了一个多层门控循环单元来实现编码器。
# @save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        '''
        nn.Embedding 是 PyTorch 中的一个模块，用于表示嵌入层。它常用于将整数索引映射为密集向量（词嵌入），并且在自然语言处理和推荐系统等任务中得到广泛应用。
        在自然语言处理中，词嵌入是将单词表示为实数域上的向量，这种表示方式能够捕捉到单词之间的语义和语法信息。nn.Embedding 可以将词汇表中的每个单词（或其它离散的实体）映射到一个高维度的向量空间中的一个点，每个单词都有一个唯一的索引作为其输入。
        
        embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
            num_embeddings：表示嵌入层的大小，即嵌入矩阵的行数，通常为词汇表的大小，或者实体的总数。
            embedding_dim：表示每个嵌入向量的维度，即嵌入矩阵的列数，决定了嵌入后的向量维度大小。
        '''
        self.embedding = nn.Embedding(vocab_size, embed_size)

        '''
        nn.GRU 是 PyTorch 中的一个模块，代表着一个门控循环单元（Gated Recurrent Unit）。GRU 是循环神经网络（RNN）的一种变体，它具有类似于长短期记忆网络（LSTM）的结构，但相对于 LSTM 更加简化。
        GRU 主要用于处理序列数据，例如文本、时间序列等，在自然语言处理和时间序列预测等任务中经常被使用。它可以帮助模型学习数据中的长期依赖关系，并且相对于传统的 RNN 结构，GRU 具有更少的参数，更容易训练，同时在某些任务上表现也更好。
        
        gru_layer = nn.GRU(input_size, hidden_size, num_layers, ...)
            input_size：输入特征的大小（例如词嵌入的维度）。
            hidden_size：隐藏状态的大小（GRU 单元的输出维度）。
            num_layers：堆叠的 GRU 层的数量。
            其他参数：还可以设置诸如是否双向、是否批量归一化等超参数。
        '''
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输入'X'的形状：(batch_size,num_steps)
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)  # 转化输入张量X的维度,对调原来的0,1维度
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)  # self.rnn.forward(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
# batch_size=4,num_steps = 7
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
print(output)
print(output.shape)
print(state)
print(state.shape)



# 2. [解码器]
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # embed_size + num_hiddens 的目的是因为要把解码器的输入X和编码器的隐藏状态拼接作为输入
        # embed_size就是解码器输入的大小. num_hiddens 就是编码器的隐状态大小
        # 参考[实现解码器输入和编码器输出拼接.png]
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        # 输出层
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 编码器的输出有两个:output, state
        # enc_outputs[1]即为了获取state
        return enc_outputs[1]

    def forward(self, X, state):
        # 输入'X'的形状：(batch_size,num_steps)
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播上下文向量context，使其具有与X相同的num_steps
        # 由于编码器解码器都是深层的rnn,且具有相同的层数,这里的state是编码器的最右上角的输出,也就是编码器的最终输出
        # state[-1]表示state的最后一个维度
        # repeat表示对state[-1]这个张量进行复制,在0维度复制X.shape[0]次,在1,2维度不变
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 将输入x和上下文向量context拼接起来作为解码器的最终输入，在第2维度拼接
        X_and_context = torch.cat((X, context), 2)
        # 输入解码器rnn，编码器的最终输入状态state
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
print(output)
print(output.shape)
print(state)
print(state.shape)


# 3. 损失函数
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项  [通过零值化屏蔽不相关的项]"""
    # X是一个二维张量，
    # valid_len是一个一维张量，元素个数等于X的0维度长度，每一个元素表示对X的第0维的最大有效位数的标记

    # X.size(1)获取x的第一维度的长度，索引从0开始
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor(  [[1, 2, 3],
                    [4, 5, 6]])
print(sequence_mask(X, torch.tensor([1, 2])))
print(sequence_mask(X, torch.tensor([1, 2]), -1))


# [通过扩展softmax交叉熵损失函数来遮蔽不相关的预测]。 最初，所有预测词元的掩码都设置为1。 一旦给定了有效长度，与填充词元对应的掩码将被设置为0。 最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测。
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

loss = MaskedSoftmaxCELoss()
print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])))


# [训练]
#  特定的序列开始词元（“<bos>”）和 原始的输出序列（不包括序列结束词元“<eos>”） 拼接在一起作为解码器的输入。
#  这被称为强制教学（teacher forcing）， 因为原始的输出序列（词元的标签）被送入解码器。 或者，将来自上一个时间步的预测得到的词元作为解码器的当前输入。
# 训练的时候解码器的输入是真实的目标语句，而预测的时候的解码器输入则是解码器上一时刻的输出（第一个输入维<bos>）
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()	# 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# [预测]
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device,
                    save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq




# 评估
def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')