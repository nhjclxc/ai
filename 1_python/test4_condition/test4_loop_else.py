#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/3 17:56
# Module    : test4_loop_else.py
# explain   : 带else的循环


'''
while...else： 在 while 循环正常结束（条件为假）时执行 else 块。
    如果 while 循环由于条件为假而退出，而不是由于 break 语句，则会执行 else 块。

    while循环正常结束会执行else里面的代码，如果是使用break跳出循环则不执行
'''
count = 0
while count < 5:
    print(count)
    count += 1
    # if count == 3:
    if count == 8:
        break
else:
    print("Loop finished without any break.")


'''
for...else： 在 for 循环正常完成（没有因为遍历完可迭代对象而提前退出）时执行 else 块。
    如果 for 循环没有被 break 中止，则会执行 else 块。
    
    for循环正常结束会执行else里面的代码，如果是使用break跳出循环则不执行
'''
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    # if num == 0:
    if num == 2:
        print("Zero detected!")
        break
else:
    print("No zeros found in the list.")