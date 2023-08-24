import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as layers
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# # python 语言方式创建标量
# a = 1.2
# # TF 方式创建标量
# aa = tf.constant(1.2)
#
# print(type(a), type(aa), tf.is_tensor(aa), tf.is_tensor(a))

#？？？所以数组是一定要有两个以上的[]吗？否则会被认为是向量？向量与数组又怎么区分呢
x = tf.constant([[1,2.,3.3], [2, 3, 4]])
# 打印 TF 张量的相关信息
x.numpy()
print(x)