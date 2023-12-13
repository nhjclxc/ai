#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/9 21:07
# Module    : test.py
# explain   :
print(111)


'''
安装d2l出现
"ERROR: Could not build wheels for pandas, which is required to install pyproject.toml-based projects"，
只需要把"pip install d2l"变成“pip install d2l pandas==1.5.3”即可解决（原因可能是你原来安装的pd和d2l的冲突了(个人见解)）

在conda里面输入：
pip install -U d2l==1.0.3

'''