#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/8 13:58
# Module    : test5_json.py
# explain   : pandas处理json数据
import json

import pandas as pd
import numpy as np

# 读取csv文件
pd_json = pd.read_json('sites.json')
print(pd_json)
print(pd_json.to_string())

# 从URL读取json数据
# pd_url_json = pd.read_json('https://static.runoob.com/download/sites.json')
# print(pd_url_json.to_string())

# 读取具有嵌套的json数据
pd_json2 = pd.read_json('json2.json')
print(pd_json2)

print('\n\n 使用py内置的文件读取json数据后转化为datafarme')
# 读取文件
with open('json2.json', 'r') as f:
    json_data = json.loads(f.read())

'''
pd.json_normalize
参数说明：
    data：要规范化的 JSON 数据或包含 JSON 数据的字典。
    record_path（可选）：用于指定要规范化的嵌套路径。它是一个字符串或列表，表示规范化的起始位置。如果不提供此参数，默认将规范化整个 JSON 结构。
    meta（可选）：用于指定要包含的元数据列。元数据列是不包含在规范化的嵌套结构中，但想要将其添加到输出 DataFrame 中的列。通常用于保留关联信息。
    sep（可选）：用于指定在嵌套结构的列名中分隔不同级别的层次结构的分隔符，默认为'.'。
'''
# 使用pandas规范化为df
print(pd.json_normalize(json_data))
print(pd.json_normalize(json_data, record_path='students'))
print(pd.json_normalize(json_data,
                        record_path='students',
                        meta=['school_name']))

print('\n\n读取更复杂的json数据')
with open('nested_mix.json', 'r') as f:
    nested_mix = json.loads(f.read())

nested_mix_df = pd.json_normalize(nested_mix,
                        record_path='students',
                        meta=[['t_info'],
                            ['t_info', 'contacts'],
                            ['t_info', 'contacts', 'email']])
json_str = nested_mix_df.to_string()
print(json_str)
print(type(json_str))
# 将字符串的str与py的dict互转
for (index, item) in nested_mix_df['t_info'].items():
    # item是dict格式
    json_str = json.dumps(item)
    print(json_str, type(json_str))
    py_dict = json.loads(json_str)
    print(py_dict, type(py_dict))



