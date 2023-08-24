import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# # 描述性统计
# print("特征平均值: ", np.mean(data, axis=0))
# print("特征中位数: ", np.median(data, axis=0))
# print("特征标准差: ", np.std(data, axis=0))

# 取一个特征用于简单线性回归
X = data[:, 4].reshape(-1, 1)
previous = data[:, 4]

# print("raw_df shape: ", raw_df.shape)
# print("previous shape: ", previous.shape)
# print("Xshape: ",X.shape)
# print("Yshape: ",target.shape)


# 划分训练集和测试集
# 划分训练集和测试集的目的：检验超参数，提高泛化能力，减少过拟合。
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

print("Xtest shape: ",X_test.shape)

# # 创建线性回归模型并训练
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = model.predict(X_test)
#
# # 评价模型性能
# print("均方误差 (MSE): ", mean_squared_error(y_test, y_pred))
# print("决定系数 (R^2): ", r2_score(y_test, y_pred))
#
# # 可视化结果
# plt.scatter(X_test, y_test, color='blue', label='label')
# plt.plot(X_test, y_pred, color='orange', label='pred')
# plt.legend()
# plt.show()