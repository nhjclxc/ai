#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/19 13:54
# Module    : test3.py
# explain   :

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from d2l_local.d2l_local import torch as d2l



print(F.one_hot(torch.tensor([0, 2]), 5))
X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape)

