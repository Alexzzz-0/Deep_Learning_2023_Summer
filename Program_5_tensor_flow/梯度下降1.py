import numpy as np
import matplotlib.pyplot as plt


# 定义我们的目标函数
def f(x):
    return x ** 2


# 定义目标函数的梯度函数
def df(x):
    return 2 * x


# 定义梯度下降函数
def gradient_descent(initial_x, iterations, learning_rate):
    x = initial_x
    x_history = []  # 用来保存每次迭代后的x值
    for _ in range(iterations):
        x_history.append(x)
        grad = df(x)
        x = x - learning_rate * grad
    return x_history


if __name__ == '__main__':
    # 设置初始值，迭代次数和学习率
    initial_x = 5
    iterations = 8
    learning_rate = 0.2

    # 进行梯度下降
    x_history = gradient_descent(initial_x, iterations, learning_rate)

    # 绘制结果
    x = np.linspace(-6, 6, 100)
    y = f(x)

    plt.plot(x, y, label='f(x) = x^2')
    plt.plot(x_history, [f(x) for x in x_history], color='orange', label='Gradient Descent', marker='o')
    plt.legend()
    plt.show()