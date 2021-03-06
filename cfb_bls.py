'''
@author:LYQ
@time:2020/12/28 下午5:20
@to do:use cfb-bls model to classify CIFAR-10
'''
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,concatenate,GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import layers,regularizers
import keras.backend.tensorflow_backend as ktf
from keras import optimizers,initializers
import numpy as np
import time
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import matplotlib
matplotlib.use('Agg')

# keras.__version__
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# GPU 显存自动调用
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)

#
np.random.seed(7)

def getTrainValTestDatas():
    # 从keras中载入数据集cifar10
    (X_train, Y_train), (x_test, y_test) = cifar10.load_data()  # X_train(50000,32,32,3),Y_train(50000,1)

    # 得到x_train,y_train,x_val,y_val
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, train_size=45000, shuffle=True, random_state=42, stratify=Y_train)

    # 将y_train,y_val,y_test进行one-hot编码
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_cfbbls_2blocks_2convs():
    model_input = Input(shape=(32, 32, 3))

    # conv-based feature block1
    c1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        model_input)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    c1 = Dropout(0.2)(c1)  # conv1
    c1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        c1)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c1)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=64 // 16, activation='relu')(squeeze)
    excitation = Dense(units=64, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 64))(excitation)
    c1_hat = Multiply()([c1, excitation])
    p1 = MaxPooling2D(pool_size=(2, 2))(c1_hat)
    p1 = Dropout(0.2)(p1)

    # conv-based feature block2
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p1)
    c2 = BatchNormalization(axis=3)(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c2)
    c2 = BatchNormalization(axis=3)(c2)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c2)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=128 // 16, activation='relu')(squeeze)
    excitation = Dense(units=128, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 128))(excitation)
    c2_hat = Multiply()([c2, excitation])
    p2 = MaxPooling2D(pool_size=(2, 2))(c2_hat)
    p2 = Dropout(0.2)(p2)

    z1 = Flatten()(p1)
    z2 = Flatten()(p2)
    merged = concatenate([z1, z2])

    d1 = Dropout(0.5)(merged)

    model_y = Dense(10, activation='softmax', kernel_initializer=initializers.lecun_normal(seed=None),
                    kernel_regularizer=regularizers.l1(0.001))(d1)

    bls_model = Model(input=model_input, output=model_y)

    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    bls_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    bls_model.summary()
    return bls_model

def create_cfbbls_3blocks_2convs():
    model_input = Input(shape=(32, 32, 3))

    # conv-based feature block1
    c1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        model_input)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    c1 = Dropout(0.2)(c1)  # conv1
    c1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        c1)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c1)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=64 // 16, activation='relu')(squeeze)
    excitation = Dense(units=64, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 64))(excitation)
    c1_hat = Multiply()([c1, excitation])
    p1 = MaxPooling2D(pool_size=(2, 2))(c1_hat)
    p1 = Dropout(0.2)(p1)

    # conv-based feature block2
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p1)
    c2 = BatchNormalization(axis=3)(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c2)
    c2 = BatchNormalization(axis=3)(c2)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c2)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=128 // 16, activation='relu')(squeeze)
    excitation = Dense(units=128, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 128))(excitation)
    c2_hat = Multiply()([c2, excitation])
    p2 = MaxPooling2D(pool_size=(2, 2))(c2_hat)
    p2 = Dropout(0.2)(p2)

    # conv-based feature block3
    c3 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p2)
    c3 = BatchNormalization(axis=3)(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c3)
    c3 = BatchNormalization(axis=3)(c3)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c3)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=256 // 4, activation='relu')(squeeze)
    excitation = Dense(units=256, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 256))(excitation)
    c3_hat = Multiply()([c3, excitation])
    p3 = MaxPooling2D(pool_size=(2, 2))(c3_hat)
    p3 = Dropout(0.2)(p3)

    z1 = Flatten()(p1)
    z2 = Flatten()(p2)
    z3 = Flatten()(p3)
    merged = concatenate([z1, z2, z3])

    d1 = Dropout(0.5)(merged)

    model_y = Dense(10, activation='softmax', kernel_initializer=initializers.lecun_normal(seed=None),
                    kernel_regularizer=regularizers.l1(0.001))(d1)

    bls_model = Model(input=model_input, output=model_y)

    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    bls_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    bls_model.summary()
    return bls_model

def create_cfbbls_3blocks_3convs():
    model_input = Input(shape=(32, 32, 3))

    # conv-based feature block1
    c1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        model_input)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    c1 = Dropout(0.2)(c1)  # conv1
    c1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        c1)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    c1 = Dropout(0.2)(c1)  # conv2
    c1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        c1)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c1)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=64 // 16, activation='relu')(squeeze)
    excitation = Dense(units=64, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 64))(excitation)
    c1_hat = Multiply()([c1, excitation])
    p1 = MaxPooling2D(pool_size=(2, 2))(c1_hat)
    p1 = Dropout(0.2)(p1)

    # conv-based feature block2
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p1)
    c2 = BatchNormalization(axis=3)(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c2)
    c2 = BatchNormalization(axis=3)(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c2)
    c2 = BatchNormalization(axis=3)(c2)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c2)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=128 // 16, activation='relu')(squeeze)
    excitation = Dense(units=128, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 128))(excitation)
    c2_hat = Multiply()([c2, excitation])
    p2 = MaxPooling2D(pool_size=(2, 2))(c2_hat)
    p2 = Dropout(0.2)(p2)

    # conv-based feature block3
    c3 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p2)
    c3 = BatchNormalization(axis=3)(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c3)
    c3 = BatchNormalization(axis=3)(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c3)
    c3 = BatchNormalization(axis=3)(c3)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c3)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=256 // 4, activation='relu')(squeeze)
    excitation = Dense(units=256, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 256))(excitation)
    c3_hat = Multiply()([c3, excitation])
    p3 = MaxPooling2D(pool_size=(2, 2))(c3_hat)
    p3 = Dropout(0.2)(p3)

    z1 = Flatten()(p1)
    z2 = Flatten()(p2)
    z3 = Flatten()(p3)
    merged = concatenate([z1, z2, z3])

    d1 = Dropout(0.5)(merged)

    model_y = Dense(10, activation='softmax', kernel_initializer=initializers.lecun_normal(seed=None),
                    kernel_regularizer=regularizers.l1(0.001))(d1)

    bls_model = Model(input=model_input, output=model_y)

    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    bls_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    bls_model.summary()
    return bls_model

def create_cfbbls_4blocks_2convs():
    model_input = Input(shape=(32, 32, 3))

    # conv-based feature block1
    c1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        model_input)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    c1 = Dropout(0.2)(c1)  # conv1
    c1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(
        c1)  # default:strides = (1,1),padding = 'valid'
    c1 = BatchNormalization(axis=3)(c1)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c1)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=64 // 16, activation='relu')(squeeze)
    excitation = Dense(units=64, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 64))(excitation)
    c1_hat = Multiply()([c1, excitation])
    p1 = MaxPooling2D(pool_size=(2, 2))(c1_hat)
    p1 = Dropout(0.2)(p1)

    # conv-based feature block2
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p1)
    c2 = BatchNormalization(axis=3)(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c2)
    c2 = BatchNormalization(axis=3)(c2)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c2)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=128 // 16, activation='relu')(squeeze)
    excitation = Dense(units=128, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 128))(excitation)
    c2_hat = Multiply()([c2, excitation])
    p2 = MaxPooling2D(pool_size=(2, 2))(c2_hat)
    p2 = Dropout(0.2)(p2)

    # conv-based feature block3
    c3 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p2)
    c3 = BatchNormalization(axis=3)(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c3)
    c3 = BatchNormalization(axis=3)(c3)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c3)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=256 // 4, activation='relu')(squeeze)
    excitation = Dense(units=256, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 256))(excitation)
    c3_hat = Multiply()([c3, excitation])
    p3 = MaxPooling2D(pool_size=(2, 2))(c3_hat)
    p3 = Dropout(0.2)(p3)

    # conv-based feature block4
    c4 = Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(p3)
    c4 = BatchNormalization(axis=3)(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu',
                kernel_initializer=initializers.lecun_normal(seed=None))(c4)
    c4 = BatchNormalization(axis=3)(c4)
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(c4)  # input:(batch_size,rows,cols,channels)
    excitation = Dense(units=512 // 4, activation='relu')(squeeze)
    excitation = Dense(units=512, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, 512))(excitation)
    c4_hat = Multiply()([c4, excitation])
    p4 = MaxPooling2D(pool_size=(2, 2))(c4_hat)
    p4 = Dropout(0.2)(p4)

    z1 = Flatten()(p1)
    z2 = Flatten()(p2)
    z3 = Flatten()(p3)
    z4 = Flatten()(p4)
    merged = concatenate([z1, z2, z3, z4])

    d1 = Dropout(0.5)(merged)

    model_y = Dense(10, activation='softmax', kernel_initializer=initializers.lecun_normal(seed=None),
                    kernel_regularizer=regularizers.l1(0.001))(d1)

    bls_model = Model(input=model_input, output=model_y)

    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    bls_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    bls_model.summary()
    return bls_model

if __name__ == '__main__':
    print('start...')

    '''preprocessing'''
    x_train, y_train, x_val, y_val, x_test, y_test = getTrainValTestDatas()
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])
    x_train = (x_train.astype('float') - mean) / std
    x_val = (x_val.astype('float') - mean) / std
    x_test = (x_test.astype('float') - mean) / std


    '''train model with datas for 10 times'''
    for i in range(10):
        model = create_cfbbls_2blocks_2convs()
        timeS = time.time()
        history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val), verbose=2,)#
        timeE = time.time()
        print('The total training Time is : ', timeE - timeS, ' seconds')
        timeS = time.time()
        print('i:', i, model.evaluate(x_test, y_test, verbose=2))
        timeE = time.time()
        print('The total testing Time is : ', timeE - timeS, ' seconds')
    print('end...')




