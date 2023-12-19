#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/18 12:32
# Module    : test1.py
# explain   :


import collections # 集合Lib用于排序处理
import re #正则表达式用于处理文本数据
from d2l import torch as d2l

'''
    怎么样把文本数据变成可以使用深度学习的东西？？？

我们将解析文本的常见预处理步骤。 这些步骤通常包括：
    1.将文本作为字符串加载到内存中。
    2.将字符串拆分为词元（如单词和字符）。
    3.建立一个词表，将拆分的词元映射到数字索引。
    4.将文本转换为数字索引序列，方便模型操作。
'''

# 1. 读取数据集  https://www.gutenberg.org/ebooks/35
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 将所有的词汇返回构成一个list，
    # 遍历所有读取到的数据的每一行，将所有不是A-Za-z]的字符串变化为空格，strip()去掉前后的空格，.lower()所有字母变化为小写
    # 经过下面的return处理之后整个文本只有a-z和空格' '，即总共只有27个字符了
    # re.sub(pattern, repl, string, count=0, flags=0)：使用新的字符串替换与模式匹配的字符串。
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])


# 2. 词元化
# 下面的tokenize函数将文本行列表（lines）作为输入， 列表中的每个元素是一个文本序列（如一条文本行）（就是时间机器这个小说的每一行）。
# [每个文本序列又被拆分成一个词元列表]（使用词word或者char作为拆分），词元（token）是文本的基本单位。
# 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）。
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        # 将每一个单词word作为拆分依据，最后组成一个列表返回
        # sep：可选参数，表示分隔符，默认为 None，此时会以空白字符（空格、制表符、换行符等）作为分隔符
        return [line.split() for line in lines]
        # return [line.split(seq = ' ') for line in lines]
    elif token == 'char':
        # 每一个字符作为拆分依据，最后返回一个由字符组成的词列表
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


# 3. 词表
'''
词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。 
现在，让我们[构建一个字典，通常也叫做词表（vocabulary）， 用来将字符串类型的词元映射到从0开始的数字索引中]。
'''

class Vocab:  #@save
    """
    我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称之为语料（corpus）。 然后根据每个唯一词元的出现频率，为其分配一个数字索引。
    很少出现的词元通常被移除，这可以降低复杂性。 另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。
    我们可以选择增加一个列表，用于保存那些被保留的词元， 例如：填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）。
    """
    # 文本词表
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
            词汇表初始化函数
        :param tokens: 所有词元
        :param min_freq: 最小出现的单词频率
        :param reserved_tokens: 保留的词元，也就是一写固定的词元，比如未知<unk> unknow，开始<bos> begin of sentence，结束<eos> start of sentence
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        # 按照counter的count即单词出现的次数进行排序，
        # lambda x: x[1]：输入是一对key:value的键值对，lambda的操作是拿出value，也就是词频
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0，这个是一个列表
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # enumerate() 是 Python 内置函数之一，它用于同时返回数据的索引和对应的值，在迭代过程中非常有用。
        # 先对保留的固定词元做初始化，注意这个是一个字典
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # 对词汇表的所有单词做token <-> index的映射
        for token, freq in self._token_freqs:
            # 当单词出现的频率小于最小出现频率时，推出
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                # idx_to_token 的索引是数字0,1,2
                self.idx_to_token.append(token)
                # token_to_idx 的key是词元 the, i, time，value是对应单词在idx_to_token里面的索引
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """ 通过token列表获取对应的索引列表 """
        # tokens传入list, tuple则直接获取，如果是*args这种则递归获取最后构造出一个list返回
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """ 通过索引的列表获取对应的token """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        # 这里的tokens其实是二维列表；第一维的元素是每一个单词，第二维度的元素是每一行
        # for line in tokens：表示对每一行进行遍历
        # for token in line：表示把每一行的所有单词拿出来，最后返回成token，并构成一个列表
        '''
        token for line in tokens for token in line：这是列表推导式的语法结构。它包含了两个嵌套的 for 循环。首先，外层的 for line in tokens 循环遍历 tokens 列表中的每一行（子列表）。然后，内层的 for token in line 循环遍历每一行中的元素（每个单词或标记），并将它们添加到新的列表中。
        '''
        tokens = [token for line in tokens for token in line]
    # collections.Counter(tokens)的作用是对tokens的所有元素出现的次数进行统计，
    # 并构造成一个token和count的字典。
    # 即：{token:count}，也就是返回了token和其对应出现的频率
    return collections.Counter(tokens)
    '''
        tokens = [['the', 'time', 'traveller', 'for', 'so'],
              ['the', 'time', 'traveller', 'for', 'so'],
              ['the', 'time', 'traveller', 'for', 'so']]
        print(tokens)
        tokens = [token for line in tokens for token in line]
        print(tokens)
        print(collections.Counter(tokens))
        Counter({'the': 3, 'time': 3, 'traveller': 3, 'for': 3, 'so': 3})
    '''

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

# ，我们可以(将每一条文本行转换成一个数字索引列表)。
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])


''' 4. 整合所有功能
在使用上述函数时，我们[将所有功能打包到load_corpus_time_machine函数中]， 该函数返回corpus（词元索引列表）和vocab（时光机器语料库的词表）。 我们在这里所做的改变是：
    为了简化后面章节中的训练，我们使用字符（而不是单词）实现文本词元化；
    时光机器数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表，而不是使用多词元列表构成的一个列表。
'''
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus))
print(len(vocab))
