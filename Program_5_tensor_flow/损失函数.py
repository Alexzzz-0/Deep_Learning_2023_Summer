import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, theta):
    return sigmoid(X * theta)

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

def calculate_losses(X, y_true, theta_values):
    mse_losses = []
    ce_losses = []
    for theta in theta_values:
        y_pred = predict(X, theta)
        mse_loss = mse(y_true, y_pred)
        ce_loss = cross_entropy(y_true, y_pred)
        mse_losses.append(mse_loss)
        ce_losses.append(ce_loss)
    return mse_losses, ce_losses

def plot_losses(theta_values, mse_losses, ce_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, mse_losses, label='MSE')
    plt.plot(theta_values, ce_losses, label='Cross-Entropy')
    plt.title("Loss Functions")
    plt.xlabel("Model Parameter (theta)")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def main():
    # 定义数据和真实标签（0和1）
    X = np.linspace(0, 1, 100)
    y_true = np.random.randint(0, 2, size=100)

    # 定义参数空间
    theta_values = np.linspace(-10, 10, 400)

    mse_losses, ce_losses = calculate_losses(X, y_true, theta_values)
    plot_losses(theta_values, mse_losses, ce_losses)


if __name__ == "__main__":
    main()