# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/11/30 16:11
# File      : test2_closure.py
# Project   : 1_python
# explain   : 闭包

# 什么时闭包
#     闭包就是一个函数

# 创建闭包的条件（可以认为是回调函数）
#   1.要有函数的嵌套（外部函数/内部函数）
#   2.内部函数中使用外部函数的变量
#   3.外部函数必须要有一个返回值，该返回值是内部函数的函数名
# 闭包能做什么

# 怎么使用闭包

def add(x, y):
    return x + y

# 定义闭包
def fun_out(x):   #
    def fun_in(y): # 满足第一点，要有函数的嵌套
        return x + y # 满足第二点，内部函数使用到外部函数
    return fun_in # 满足第三点，外部函数的返回值是内部函数的函数名
    # 上面的把内部函数名作为返回值返回，把这个内部函数作为变量返回，在外部函数返回的时候再去调用返回值的这个函数。注意观察”fun_out(5)(6)“就可以理解了

# 使用闭包
print(fun_out(5)(6))
print(fun_out(5))
print(type(fun_out(5)))
ff = fun_out(5)
print(ff(55))

