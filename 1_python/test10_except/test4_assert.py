#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/3 16:21
# Module    : test4_assert.py
# explain   : 使用断言assert做判断
import logging


def div(a, b):
    # 断言格式：assert 逻辑运算, 逻辑运算结果为False时抛出AssertionError异常的信息

    # 如果段誉结果为True，那么程序正常执行，否则抛出一个AssertionError类型的异常，也就是：raise AssertionError('除数不能为0')
    assert b != 0, '除数不能为0'
    return a / b

print(div(10, 2))
# print(div(10, 0))

# 在启动py脚本时，可以使用-O出书来关闭断言，如：python -O err.py，注意：-O时大写字幕O，不是数字0.关闭后，你可以把所有的assert语句当成pass来看。



'''
使用日志来输出信息
'''
# 配置日志输出器
import logging
# 定义日志信息输出级别
logging.basicConfig(level=logging.DEBUG)

# 使用
logging.debug(f'第一次输出日志 debug')
logging.info(f'第一次输出日志 info')
logging.warning(f'第一次输出日志 warning')
logging.error(f'第一次输出日志 error')
print(type(logging.INFO))
print(logging.INFO)



