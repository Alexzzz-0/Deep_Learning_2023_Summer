import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    # 加载数据
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv(url, header=None, delim_whitespace=True, names=column_names)

    # 选择RM作为特征
    X = data['NOX'].values
    y = data['MEDV'].values

    # 特征缩放
    X = (X - np.mean(X)) / np.std(X)
    X = np.c_[np.ones(X.shape[0]), X]  # 添加偏置项

    return X, y

# 定义预测函数
# def predict(X, theta):
#     return np.dot(X, theta)

def predict(X, theta):
    predictions = []
    for i in range(len(X)):
        prediction = sum([X[i, j] * theta[j] for j in range(len(theta))])
        predictions.append(prediction)
    return np.array(predictions)


# 定义损失函数(mse)
def compute_cost(X, y, theta):
    return np.sum((predict(X, theta) - y) ** 2) / (2 * len(y))

# 定义梯度下降函数
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        theta -= alpha / m * np.dot(X.T, (predict(X, theta) - y))
        costs.append(compute_cost(X, y, theta))
    return theta, costs

def plot_prediction(X, y, theta):
    # 绘制拟合线
    plt.scatter(X[:, 1], y, label='Training Data')
    plt.plot(X[:, 1], predict(X, theta), color='orange', label='Prediction')
    plt.title('Linear Regression with Gradient Descent')
    plt.xlabel('Number of Rooms (Normalized)')
    plt.ylabel('House Price')
    plt.legend()
    plt.show()

def plot_cost(costs):
    # 绘制损失值
    plt.plot(costs)
    plt.title('Cost Function J')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

if __name__ == '__main__':
    # 加载数据
    X, y = load_data()

    # 初始化参数
    theta = np.zeros(2)

    # 设置学习率和迭代次数
    alpha = 0.004
    iterations = 500

    # 进行梯度下降
    theta, costs = gradient_descent(X, y, theta, alpha, iterations)

    # 绘制预测结果
    plot_prediction(X, y, theta)

    # 绘制损失值
    plot_cost(costs)