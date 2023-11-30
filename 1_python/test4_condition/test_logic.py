# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/11/30 9:22
# File      : test_logic.py
# Project   : 1_python
# explain   : 逻辑运算


'''
Python 中的逻辑运算包括常见的逻辑运算符和逻辑运算表达式，主要用于布尔类型的数据（True 和 False）。

常见的逻辑运算符包括：

逻辑与 (and)： 如果两个条件都为 True，则返回 True；否则返回 False。
逻辑或 (or)： 如果两个条件中至少有一个为 True，则返回 True；否则返回 False。
逻辑非 (not)： 用于取反一个布尔值，not True 返回 False，not False 返回 True。

'''
# 逻辑运算
x = True
y = False
print(x and y)  # 输出: False
print(x or y)   # 输出: True
print(not x)    # 输出: False

# 位运算
a = 5  # 二进制 0101
b = 3  # 二进制 0011
print(a & b)   # 输出: 1 (二进制 0001)
print(a | b)   # 输出: 7 (二进制 0111)
print(~a)      # 输出: -6 (二进制 11111010，按位取反并加一)

'''

在 Python 中，&、| 和 ! 这些符号通常用于位运算，而不是逻辑运算。在处理布尔值时，逻辑运算使用的是 and、or 和 not。

& 是按位与运算符，用于对两个整数的每个对应位执行位与操作。
| 是按位或运算符，用于对两个整数的每个对应位执行位或操作。
! 不是 Python 中的逻辑非运算符，而是用于位取反操作的按位取反运算符。
如果你想要进行逻辑运算，应使用 and、or 和 not。如果要进行位运算，才使用 &、| 和 ~。

虽然 & 和 | 可以用于逻辑运算，但是在处理布尔值时，应该使用 and 和 or，因为它们更易读且更贴近逻辑运算的本质。
'''

L = None
if L:
    print('1')
else:
    print(0)
L = []
if L:
    print('1')
else:
    print(0)
L = [1,2,3]
if L:
    print('1')
else:
    print(0)