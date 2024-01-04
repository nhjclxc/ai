#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/1/4 15:49
# Module    : beam_search3.py
# explain   :

from math import log
import numpy as np

# https://blog.csdn.net/ZHUQIUSHI123/article/details/88741990

def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]

    for row in data:
        all_candidates = list()

        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1])  # 按score排序
        sequences = ordered[:k]  # 选择前k个最好的
    return sequences


def greedy_decoder(data):

    return [np.argmax(s) for s in data]


if __name__ == '__main__':

    data = [[0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.5, 0.3, 0.2]]
    data = np.array(data)
    result = beam_search_decoder(data, 3)
    print("****use beam search decoder****")
    for seq in result:
        print(seq)

    print("****use greedy decoder****")
    print(greedy_decoder(data))


