#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/9 16:31
# Module    : test.py
# explain   :

# 第1步：导入数据分析库pandas，数据可视化库matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # 第2步：导入数据集，查看数据集
dataset = pd.read_csv('./studentscores.csv')
print(dataset)

# # 第3步：提取特征
feature_columns = ['Hours']   # 输入就是你的特征
label_column = ['Scores']    # 输出就是你的标签
features = dataset[feature_columns]
label = dataset[label_column]

X = features.values
Y = label.values

# # 第四步：建立模型
# # 拆分数据，四分之三的数据作为训练集，四分之一的数据作为测试集
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

# 用训练集的数据进行训练
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

# # 可视化
# # 散点图：红色点表示训练集的点
plt.scatter(X_train , Y_train, color = 'red')
# # 线图：蓝色线表示由训练集训练出的线性回归模型
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()






