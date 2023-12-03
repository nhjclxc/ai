#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/3 16:04
# Module    : test3.py
# explain   :


''''
py的所有异常类型都来自：BaseException（https://docs.python.org/3/library/exceptions.html#exception-hierarchy）

'''
import traceback

# num = int(input('输出一个数：'))
# try:
#     if num == 10:
#         # 手动抛出一个异常
#         raise '输出的数字为10'
# except RuntimeError as e:
#     traceback.extract_stack()
# else:
#     print('不是10')
# finally:
#     print('输入 结束')






from functools import reduce

def str2num(s):
    try:
        return int(s)
    except ValueError as ve:
        try:
            return float(s)
        except BaseException :
            raise '不是数字类型，不能操作'

def calc(exp):
    ss = exp.split('+')
    ns = map(str2num, ss)
    return reduce(lambda acc, x: acc + x, ns)

def main():
    r = calc('100 + 200 + 345')
    print('100 + 200 + 345 =', r)
    r = calc('99 + 88 + 7.6')
    print('99 + 88 + 7.6 =', r)

main()

