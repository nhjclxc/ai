#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 16:45
# File      : __init__.py.py
# Project   : 1_python
# explain   :



class Student:
    def __init__(self, name='None', age=18):
        self.name = name
        self.age = age

    def to_string(self):
        return f'Student = {{name = {self.name}, age = {self.age}}}'

if __name__ == '__main__':
    stu = Student()
    print(stu.to_string())
