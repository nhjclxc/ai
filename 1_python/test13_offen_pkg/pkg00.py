import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

arr = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [11, 22, 33],
]
np_arr = np.array(arr)

print(arr)
print(np_arr)
print(np_arr.shape)
np_arr = np_arr.reshape(3, 4)
print(np_arr)
print(np_arr.shape)
np_arr = np_arr.reshape(2, 6)
print(np_arr)
print(np_arr.shape)

print('-------1-------')
np2 = np.array([-3, -2, -1, 0, 1, 2, 3])
print(np2)
print(np2.shape)
np2 = np2.reshape(7, 1)
print(np2)
print(np2.shape)

print('-------2-------')
np2 = np.array([-3, -2, -1, 0, 1, 2, 3])
print(np2)
print(np2.shape)
np2 = np2.reshape(-1, 1)
print(np2)
print(np2.shape)

print('-------3---------')
a = np.arange(6)
print(a.shape, a.ndim, '\n', a)
a = a.reshape(-1,1)
print(a.shape, a.ndim, '\n', a)
a = a.reshape(3,-1)
print(a.shape, a.ndim, '\n', a)
a = a.reshape(-1,3)
print(a.shape, a.ndim, '\n', a)




#
#
# # 1. 数据（示例，可替换为任意离散点）
# X = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1)
# y = np.array([-20, -5, -1, 1, 3, 10, 25])
#
# # 2. 多项式特征
# degree = 3
# poly = PolynomialFeatures(degree=degree, include_bias=False)
# X_poly = poly.fit_transform(X)
#
# # 3. 拟合线性模型到多项式特征
# model = LinearRegression()
# model.fit(X_poly, y)
#
# # 4. 预测与可视化
# X_fit = np.linspace(-3, 3, 200).reshape(-1, 1)
# X_fit_poly = poly.transform(X_fit)
# y_fit = model.predict(X_fit_poly)
#
#
# # 5. 打印模型参数
# print(f"多项式阶数：{degree}")
# print("拟合系数：", model.coef_)
# print("截距：", model.intercept_)
