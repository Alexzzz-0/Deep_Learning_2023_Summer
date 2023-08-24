import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.python.keras as kr
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras import layers
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy



def load_data():

    cut_imgs = []
    labels = []

    #对于每张照片
    for picname in os.listdir(r"E:\Other_Programs\DeepLearning\cut_pic"):
        picname_list = picname.split('_')

        picnames = 'E:\Other_Programs\DeepLearning\cut_pic/' + picname
        img = cv.imread(picnames,1)
        img_resize = cv.resize(img,(28,28),cv.INTER_AREA)
        cut_imgs.append(img_resize)
        labels.append(picname_list[0].strip())

        labels_num = np.asarray(labels,dtype=int)
        imgs_array = np.asarray(cut_imgs)
    #一共2806个标签和切片图

    return imgs_array,labels_num

def train_test_split_array(X,target):
    X_train, X_test, y_train, y_test = train_test_split(X, target,test_size=0.2,random_state=88)

    return X_train, X_test, y_train, y_test

def build_and_train_model(x_train, y_train):

    model = Sequential()

    # 第一层卷积层：6个卷积核，大小为5x5，激活函数为ReLU
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 3)))

    # 第一层池化层：最大池化，窗口大小为2x2，步长为2
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # 第二层卷积层：16个卷积核，大小为5x5，激活函数为ReLU
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))

    # 第二层池化层：最大池化，窗口大小为2x2，步长为2
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # 展平层：用于将上一层的二维输出转换为一维
    model.add(layers.Flatten())

    # 全连接层1：120个神经元，激活函数为ReLU
    model.add(layers.Dense(120, activation='relu'))

    # 全连接层2：84个神经元，激活函数为ReLU
    model.add(layers.Dense(84, activation='relu'))

    # 输出层：10个神经元，激活函数为Softmax，用于分类
    model.add(layers.Dense(4, activation='softmax'))

    # 编译并训练
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=2)

    return model

def evaluate_model(model, x_test, y_test):
    # 在测试集上评估模型
    _, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', accuracy)

def visualize_classification(x_test, y_true, y_pred, num_images=5):
    # 随机选择一些图像
    indices = np.random.choice(len(x_test), size=num_images, replace=False)

    # 测试读取方式是否正确 #结果可以显示
    for i in indices:
        cv.namedWindow('img',cv.WINDOW_AUTOSIZE)
        cv.imshow('img', x_test[i])
        cv.waitKey(0)
        cv.destroyAllWindows()
        print("test result:"+y_pred[i],"real result:"+y_true[i])


def main():
    imgs,labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split_array(imgs,labels)
    model = build_and_train_model(X_train,y_train)
    evaluate_model(model,X_test,y_test)

    # y_pred = np.argmax(model.predict(X_test), axis=1)
    # visualize_classification(X_test, y_test, y_pred, num_images=10)

if __name__ == "__main__":
    main()