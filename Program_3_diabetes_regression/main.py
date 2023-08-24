import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X_all = diabetes.data
Y = diabetes.target

print("X大小： ", X_all.shape)
print("Y大小： ", Y.shape)

X = X_all[:,0].reshape(-1,1)

print("特征平均数： ", np.mean(X,axis=0))
print("特征中位数： ", np.median(X,axis=0))
print("特征标准差： ", np.std(X,axis=0))

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

y_gap = y_test - y_pred


print("均方误差： ",mean_squared_error(y_test,y_pred))
print("决定系数： ", r2_score(y_test,y_pred))
print("误差: ", np.mean(y_gap,axis=0))

plt.scatter(X_test, y_test, color='blue', label='label')
plt.plot(X_test, y_pred, color='orange', label='pred')
plt.scatter(X_test, y_gap, color='red', label='gap')

plt.legend()
plt.show()

