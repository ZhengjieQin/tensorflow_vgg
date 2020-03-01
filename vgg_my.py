import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
#x是数据，y是标签
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

#对训练集和验证集进行差分
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

#对数据进行归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#x_train 是一个三维的矩阵{None，28，28}->{none,28},归一化用的是x-均值/标准差，而验证集和测试集都用的是训练集的标准差和均值
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)

x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)

x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28,1)
'''
train_data = tf.data.Dataset.from_tensor_slices((x_train_scaled,y_train))
valid_data = tf.data.Dataset.from_tensor_slices((x_valid_scaled,y_valid))
test_data = tf.data.Dataset.from_tensor_slices((x_test_scaled,x_test))
'''
#构建模型,用keras.sequential
model = keras.models.Sequential()
#model.add(keras.layers.InputLayer(input_shape=[28,28,1]))
model.add(keras.layers.Conv2D(32,kernel_size=[3,3],padding='same',activation='relu',input_shape=[28,28,1]))
model.add(keras.layers.Conv2D(32,kernel_size=[3,3],padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))

model.add(keras.layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu'))
model.add(keras.layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))

model.add(keras.layers.Conv2D(128,kernel_size=[3,3],padding='same',activation='relu'))
model.add(keras.layers.Conv2D(128,kernel_size=[3,3],padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))

model.add(keras.layers.Conv2D(256,kernel_size=[3,3],padding='same',activation='relu'))
model.add(keras.layers.Conv2D(256,kernel_size=[3,3],padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation='relu'))
model.add(keras.layers.Dense(256,activation='relu'))
model.add(keras.layers.AlphaDropout(rate=0.5))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(100,activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics =["accuracy"])  #这是分类指标
#sparse 是因为标签类似与索引，所以要用one_hot,每一为代表一个特征，
#如果是两个值的化就是两个位



#model.summary()

#开启训练
history =model.fit(x_train_scaled,y_train,epochs=10,
          validation_data=(x_valid_scaled,y_valid))
print(history)

#将结果做成图
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10, 5))#图的大小
    plt.grid(True)  #设置网格
    plt.gca().set_ylim(0, 1) #设置y轴的范围
    plt.show()

plot_learning_curves(history)
model.evaluate(x_test_scaled,y_test)