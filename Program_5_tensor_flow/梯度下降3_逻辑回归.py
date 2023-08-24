import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 我们只使用前两个特征。
    y = (iris.target != 0) * 1  # 将数据集修改为二分类任务
    return X, y

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    test = np.concatenate((intercept, X), axis=1)
    return np.concatenate((intercept, X), axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def predict_probs(X, theta):
    return sigmoid(np.dot(X, theta))

def predict(X, theta, threshold=0.5):
    return predict_probs(X, theta) >= threshold

def train(X, y, num_iter=100000, learning_rate=0.01):
    theta = np.zeros(X.shape[1])

    for _ in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= learning_rate * gradient

        if _ % 10000 == 0:
            print(f'loss: {loss(h, y)}')

    return theta

def plot(X, y, theta):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color='green', label='0')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color='orange', label='1')
    plt.legend()
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = predict_probs(add_intercept(grid), theta).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');
    plt.show()

if __name__ == "__main__":
    X, y = load_data()
    X = add_intercept(X)
    theta = train(X, y)
    plot(X, y, theta)