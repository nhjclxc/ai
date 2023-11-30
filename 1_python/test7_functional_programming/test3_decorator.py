# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/11/30 18:37
# File      : test3_decorator.py
# Project   : 1_python
# explain   : 装饰器
import time


# 在原有的函数func上面增加日志输出（实现一个类似于切片编程的东西）
def func(*args):
    print(f'func被执行, args = {args}')
    return '返回值', 5


def log(func, args, result):
    '''
        定义输出日志的函数
    :param func: 要增加日志功能的函数
    :return:
    '''
    with open('log.txt', 'a', encoding='utf-8') as file:
        file.write(f'{time.asctime()}, func = {func.__name__}, args = {args},  result = {result}, \n')


# 定义闭包
def wrapper_func(func):
    # 被装饰器装饰的函数要是有参数的话，把对应的参数写道装饰器的内部函数上面，传入
    def inner_func(*args):
        print(f'{func.__name__}函数执行前，time = {time.time()}')
        result = func(args)
        # time.sleep(1)
        print(f'{func.__name__}函数执行后，time = {time.time()}')
        log(func, args, result)

    return inner_func


wfunc = wrapper_func(func)
wfunc('12', 'asac', 345, True)


# 使用装饰器的注解
# 下面的@wrapper_func的作用，就是在执行函数func22是，会把这个函数使用wrapper_func来进行装饰一下
@wrapper_func
def func22(*args):
    print(f'func22被执行, args = {args}')
    return '我是func22返回值哦', 666


# 只要加上注解 @wrapper_func就可以直接在func22使用装饰器了
func22(123, 'asc')
