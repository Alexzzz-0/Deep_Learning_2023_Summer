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
    for picname in os.listdir(r"E:/Other_Programs/DeepLearning/pic2000"):
    #for picname in os.listdir(r"E:\Other_Programs\DeepLearning\saveImg"):
        picname_split = picname.split('.')[0]

        #找到对应文档
        for txtname in os.listdir(r"E:/Other_Programs/DeepLearning/txt2000"):
            txtname_spilt = txtname.split('.')[0]

            if txtname_spilt == picname_split:
                with open(r'E:/Other_Programs/DeepLearning/txt2000/'+txtname,'r') as txt_context:
                    txt_lines = txt_context.readlines()
                    break

        #跳过空标签的图片
        if(len(txt_lines)==0):
            #print("This one is empty.")
            continue

        #获取图片信息
        picnames = 'E:/Other_Programs/DeepLearning/pic2000/' + picname
        #picnames = 'E:\Other_Programs\DeepLearning\saveImg/' + picname
        img = cv.imread(picnames, 1)
        h,w,_ = img.shape
        #print(picname)

        # #测试读取方式是否正确 #结果可以显示
        # cv.namedWindow('img',cv.WINDOW_NORMAL)
        # cv.imshow('img', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #print(txt_lines)

        #根据文档内容截取图片信息
        num = 0
        for line in txt_lines:

            contline = line.split(' ')

            # centerX = float((contline[1]).strip())
            # centerY = float((contline[2]).strip())
            width = float((contline[3]).strip())
            height = float((contline[4]).strip())

            xmin = float((contline[1]).strip()) -width / 2
            xmax = float(contline[1].strip()) + width / 2

            ymin = float(contline[2].strip()) - height / 2
            ymax = float(contline[2].strip()) + height / 2

            # 将坐标（0-1之间的值）还原回在图片中实际的坐标位置
            xmin, xmax = int(w * xmin), int(w * xmax)-5
            ymin, ymax = int(h * ymin), int(h * ymax)-5

            #注意YOLO对应坐标是横，纵，横，纵，但从数组shape得到的确是图像的高和宽
            cut_pic = img[ymin:ymax,xmin:xmax]


            # #测试读取方式是否正确 #结果可以显示
            # cv.namedWindow('img',cv.WINDOW_AUTOSIZE)
            # cv.imshow('img', cut_pic)
            # cv.waitKey(0)
            # cv.destroyAllWindows()


            #压缩图片大小

            pic_resize = cv.resize(cut_pic, (32,32),cv.INTER_AREA)
            # cv.imwrite(contline[0]+"_"+picname_split+"_"+str(num)+".jpg",pic_resize)
            # num = num + 1

            # #测试读取方式是否正确 #结果可以显示
            # cv.namedWindow('img',cv.WINDOW_AUTOSIZE)
            # cv.imshow('img', pic_resize)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cut_imgs.append(pic_resize)
            labels.append(contline[0].strip())

            # if (int(contline[0])==4):
            #     print(picname)


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
    model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(32, 32,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=4,batch_size=4)

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