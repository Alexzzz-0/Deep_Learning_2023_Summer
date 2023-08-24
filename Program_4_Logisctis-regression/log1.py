import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(9, 6))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [1, 1], "k--")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(x, y, "b-", linewidth=2, label=r"$\sigma(x) = \frac{1}{1 + e^{-x}}$")
plt.xlabel("x")
plt.legend(loc="upper left", fontsize=13)
plt.axis([-10, 10, -0.1, 1.1])
plt.show()