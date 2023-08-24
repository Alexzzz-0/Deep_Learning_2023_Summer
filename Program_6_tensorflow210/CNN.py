import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_and_preprocess_data():
    #(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    with np.load(r'E:\Other_Programs\DeepLearning\Datasets\mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


def build_and_train_model(x_train, y_train):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    return model


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


def visualize_classification(x_test, y_true, y_pred, num_images=5):
    # 随机选择一些图像
    indices = np.random.choice(len(x_test), size=num_images, replace=False)

    fig, axs = plt.subplots(1, num_images, figsize=(2 * num_images, 2))
    for i, idx in enumerate(indices):
        image = x_test[idx].reshape(28, 28)  # Reshape the image back to its original shape
        true_label = np.argmax(y_true[idx])  # Convert one-hot encoded label back to numerical
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
