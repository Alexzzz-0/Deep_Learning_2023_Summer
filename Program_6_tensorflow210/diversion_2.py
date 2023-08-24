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
        cut_imgs.append(img)
        labels.append(picname_list[0].strip())


    # img_test = cv.imread("E:\Other_Programs\DeepLearning\pic2000/01.jpg")
    # print(img_test.shape)
    # #the size of the picture is (4032,3024)

        #cv.imwrite(picname_split + ".jpg", img)

        labels_num = np.asarray(labels,dtype=int)
        imgs_array = np.asarray(cut_imgs)
    #一共2806个标签和切片图

    return imgs_array,labels_num

def train_test_split_array(X,target):
    X_train, X_test, y_train, y_test = train_test_split(X, target,test_size=0.2,random_state=88)

    return X_train, X_test, y_train, y_test

def build_and_train_model(x_train, y_train):

    #建立模型
    # model = Sequential([
    #     Flatten(input_shape=(64, 64,3)),
    #     Dense(4, activation='relu'),
    #
    #     Dense(4, activation='relu'),
    #     Dense(4)
    # ])
    model = Sequential()
    model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu', input_shape=(64, 64,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (4, 4), activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=4,batch_size=2)

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