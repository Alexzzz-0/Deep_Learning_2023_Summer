import tensorflow as tf
import tensorflow.python.keras as kr
#from tensorflow.python.keras.datasets import mnist

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy


import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def load_and_preprocess_data():
    # 加载 MNIST 数据集
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    with np.load(r'E:\Other_Programs\DeepLearning\Datasets\mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']


    # 数据预处理，将图片的像素值范围从 [0, 255] 缩放到 [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0


    return (x_train, y_train), (x_test, y_test)

def build_and_train_model(x_train, y_train):
    # 建立模型
    model = Sequential([
        Flatten(input_shape=(28, 28)), # 将图片从 28x28 的矩阵压平成一个向量
        Dense(12, activation='relu'), # 第一个全连接层，有 128 个神经元，使用 ReLU 激活函数
        Dense(10) # 输出层，有 10 个神经元，对应 10 个数字（0-9）
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=5)

    return model

def evaluate_model(model, x_test, y_test):
    # 在测试集上评估模型
    _, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', accuracy)



def visualize_classification(x_test, y_true, y_pred, num_images=5):
    # 随机选择一些图像
    indices = np.random.choice(len(x_test), size=num_images, replace=False)

    fig, axs = plt.subplots(1, num_images, figsize=(2 * num_images, 2))
    for i, idx in enumerate(indices):
        image = x_test[idx].reshape(28, 28)  # Reshape the image back to its original shape
        true_label = y_true[idx]
        pred_label = y_pred[idx]

        axs[i].imshow(image, cmap='gray')
        axs[i].set_title(f"True: {true_label}, Pred: {pred_label}", fontsize=8)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_and_train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)

    # 得到模型预测的结果，取概率最高的类别作为预测的类别
    y_pred = np.argmax(model.predict(x_test), axis=1)
    visualize_classification(x_test, y_test, y_pred, num_images=10)

if __name__ == "__main__":
    main()