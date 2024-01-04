#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/1/4 14:55
# Module    : beam_search2.py
# explain   :

# https://pypi.org/project/pytorch-beam-search/

from pytorch_beam_search import seq2seq

# Create vocabularies
# Tokenize the way you need
source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
target = [list("ABCDEFGHIJKL"), list("MNOPQRSTWXYZ")]
# An Index object represents a mapping from the vocabulary
# to integers (indices) to feed into the models
source_index = seq2seq.Index(source)
target_index = seq2seq.Index(target)

# Create tensors
X = source_index.text2tensor(source)
Y = target_index.text2tensor(target)
# X.shape == (n_source_examples, len_source_examples) == (2, 11)
# Y.shape == (n_target_examples, len_target_examples) == (2, 12)

# Create and train the model
model = seq2seq.Transformer(source_index, target_index)    # just a PyTorch model
model.fit(X, Y, epochs = 100)    # basic method included

# Generate new predictions
new_source = [list("new first in"), list("new second in")]
new_target = [list("new first out"), list("new second out")]
X_new = source_index.text2tensor(new_source)
Y_new = target_index.text2tensor(new_target)
loss, error_rate = model.evaluate(X_new, Y_new)    # basic method included
predictions, log_probabilities = seq2seq.beam_search(model, X_new)
output = [target_index.tensor2text(p) for p in predictions]
print(output)

