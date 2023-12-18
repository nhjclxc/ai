#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/18 13:16
# Module    : test2.py
# explain   :
import collections

tokens = [['the', 'time', 'traveller', 'for', 'so'],
          ['the', 'time', 'traveller', 'for', 'so'],
          ['the', 'time', 'traveller', 'for', 'so']]
print(tokens)
tokens = [token for line in tokens for token in line]
print(tokens)
print(collections.Counter(tokens))

for idx, token in enumerate(tokens):
    print(idx, token)

counter = collections.Counter(['the', 'time', 'traveller', 'for', 'so'])
_token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
print(counter)
print(_token_freqs)
idx_to_token = []
token_to_idx = {}
for token, freq in _token_freqs:
    idx_to_token.append(token)
    token_to_idx[token] = len(idx_to_token) - 1

print(idx_to_token)
print(token_to_idx)

original_list = ['', 1, 2, 'ww']
print(original_list)
filtered_list = list(filter(lambda x: x != '', original_list))
print(original_list)
print(filtered_list)


lst = ['the', 'time', 'traveller', 'for', 'so']
print(lst.get([0,2,3]))



