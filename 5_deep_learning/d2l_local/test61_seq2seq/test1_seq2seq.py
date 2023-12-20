#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/19 21:03
# Module    : test1_seq2seq.py
# explain   :


from torch import nn

'''
编码器-解码器架构[编码器解码器数据流.png]
机器翻译是序列转换模型的一个核心问题， 其输入和输出都是长度可变的序列。 
为了处理这种类型的输入和输出， 我们可以设计一个包含两个主要组件的架构： 
    第一个组件是一个编码器（encoder）： 它接受一个长度可变的序列作为输入， 并将其转换为具有固定形状的编码状态。 
    第二个组件是解码器（decoder）： 它将固定形状的编码状态映射到长度可变的序列。 这被称为编码器-解码器（encoder-decoder）架构，
'''

# 定义编码器
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


# 定义解码器
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

# 合并编码器解码器
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """
            编码器解码器模型 的前向传播
        :param enc_X: 编码器输入
        :param dec_X: 解码器输入
        :param args:
        :return:
        """
        # 以下就是实现:编码器解码器数据流.png

        # 实现编码器输入经过编码器后得到编码器的输出
        enc_outputs = self.encoder(enc_X, *args)
        # enc_outputs其实就是编码器的隐状态输出
        # 解码器拿到编码器的输出状态之后放入解码器里面初始化解码器
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 解码器接收编码器初始化的状态输出和解码器的输入,之后通过forward进行传播
        return self.decoder(dec_X, dec_state)


