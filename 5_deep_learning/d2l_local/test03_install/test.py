#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/9 21:07
# Module    : test.py
# explain   :
print(111)

"""

安装d2l-0.17.6以下方法都有一些问题，建议直接把[https://github.com/d2l-ai/d2l-zh.git]里面的d2l文件夹里面所有文件下载到本地自己创建一个d2l模块就可以不用安装沐神提供的这个包了



安装d2l-0.17.6【https://blog.csdn.net/sriting/article/details/129600084】
下载d2l-0.17.6的包【https://pypi.org/project/d2l/0.17.6/#files】：https://files.pythonhosted.org/packages/1d/e6/afdec5250085814182e4b4e2629905c81e5cdf200f17751bdb06818624b0/d2l-0.17.6-py3-none-any.whl
打开conda对应的虚拟环境：D:\develop\Anaconda3\envs\py310\python.exe -m pip install "E:\nbu\ai\5_deep_learning\d2l_local\d2l_local\d2l-0.17.6-py3-none-any.whl"
【python -m pip install "包路径"】



管理员权限进入cmd
安装：pip install d2l==0.17.6

安装d2l出现："ERROR: Could not build wheels for pandas, which is required to install pyproject.toml-based projects"，

先把原来的pandas卸载：pip uninstall pandas
再重新安装d2l，d2l里面会带pandas：pip install d2l==0.17.6 pandas==1.2.4


在conda里面安装
先激活对应的环境：conda activate env
安装：pip install -U d2l==0.17.6 pandas==1.2.4


'''出现如下：
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pandas
Failed to build pandas
ERROR: Could not build wheels for pandas, which is required to install pyproject.toml-based projects
'''
则：https://blog.csdn.net/sriting/article/details/129600084，
下载包：https://files.pythonhosted.org/packages/6d/0f/3fef6d450d8476b7d944c0811afc761b28d71bd9753b130ded449b9df379/d2l-1.0.0b0-py3-none-any.whl
cmd管理员权限打开编译文件：python -m pip install "E:\nbu\ai\5_deep_learning\d2l_local\d2l_local\d2l-1.0.0b0-py3-none-any.whl"


则：https://blog.csdn.net/sriting/article/details/129600084，
下载d2l-0.17.6的包【https://pypi.org/project/d2l/0.17.6/#files】：https://files.pythonhosted.org/packages/1d/e6/afdec5250085814182e4b4e2629905c81e5cdf200f17751bdb06818624b0/d2l-0.17.6-py3-none-any.whl
打开conda对应的虚拟环境：D:\develop\Anaconda3\envs\py310\python.exe -m pip install "E:\nbu\ai\5_deep_learning\d2l_local\d2l_local\d2l-0.17.6-py3-none-any.whl"





其他
# pip install -U d2l
"""