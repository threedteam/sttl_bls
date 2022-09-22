"""
Descripttion:
version:
Author: Lv Di
Date: 2022-03-15 15:50:27
LastEditors: Lv Di
LastEditTime: 2022-03-15 16:40:23
"""
import json

import numpy as np
from keras import initializers, regularizers, metrics, Model
from keras.datasets import cifar10
from keras.layers import Input, Flatten, concatenate, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.regularizers import l2

from utils.FeatureBlock import FeatureBlock


def image_generator():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
    )
    return datagen


def get_data(val_size=0.33, subtract_pixel_mean=False):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=42
    )

    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])
    x_train = (x_train.astype("float") - mean) / std
    x_val = (x_val.astype("float") - mean) / std
    x_test = (x_test.astype("float") - mean) / std

    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"shape for training :{x_train.shape} and {y_train.shape}")
    print(f"shape for validation :{x_val.shape} and {y_val.shape}")

    return x_train, y_train, x_test, y_test, x_val, y_val


def gdbls(use_ppv=False, parameters_info=None, dataset_info=None):
    # Grand Descent Board Learning System with Dropout

    # init parameters:
    filters = parameters_info["filters"]
    se_div = parameters_info["SE_div"]
    data_format = dataset_info["data_format"]
    kernel_init_seed = eval(parameters_info["kernel_init_seed"])
    l2_reg = parameters_info["kernel_regular_l2"]
    """
    REFERENCE: Bag of Tricks for Image Classification with CNN
    flexible width - organised feature blocks' parameters.
    1. filters: 
    exp: 128-128-128 gets the best score of 0.905
    reason: the original setting of 64-128-256 may ignore the edge information of the input tensor.
    2. div_n:
    exp: 16-8-4 of div_n setting's performance is 90.56%, min val_loss:0.45
    reason: more channels demands more units in SE blocks to learn the relationship between them
    """

    model_input = Input(eval(json.dumps(dataset_info["size"])))

    p1 = FeatureBlock(
        input_tensor=model_input,
        nb_convs=3,
        kernel_replace=3,
        kernel_init_seed=kernel_init_seed,
        l2_reg=l2_reg,
        out_channels=filters[0],
        use_ppv=use_ppv,
        div_n=se_div[0],
        data_format=data_format,
        down_sample=True,
    )
    p2 = FeatureBlock(
        input_tensor=p1,
        nb_convs=3,
        kernel_replace=3,
        kernel_init_seed=kernel_init_seed,
        l2_reg=l2_reg,
        out_channels=filters[1],
        use_ppv=use_ppv,
        div_n=se_div[1],
        data_format=data_format,
        down_sample=True,
    )
    p3 = FeatureBlock(
        input_tensor=p2,
        nb_convs=3,
        kernel_replace=5,
        kernel_init_seed=kernel_init_seed,
        l2_reg=l2_reg,
        out_channels=filters[2],
        use_ppv=use_ppv,
        div_n=se_div[2],
        data_format=data_format,
        down_sample=True,
    )

    """
    利用OVERALL SE机制为提取出的各层feature map赋予权重: 尺度特征聚合模块SFAM


    """

    # Flatten Layer
    z1 = Flatten()(p1)
    z2 = Flatten()(p2)
    z3 = Flatten()(p3)
    merged = concatenate([z1, z2, z3])

    # Overall Dropout Layer
    d1 = Dropout(parameters_info["overall_dropout_rate"])(merged)

    # Output Layer
    model_y = Dense(
        10,
        activation="softmax",
        kernel_initializer=initializers.lecun_uniform(seed=kernel_init_seed),
        kernel_regularizer=l2(parameters_info["kernel_regular_l2"]),
    )(d1)

    # create the result model
    bls_model = Model(inputs=model_input, outputs=model_y)

    # adam optimizer / amsgrad variant
    # used decay parameter to control the lr reduction rate.
    adam = keras.optimizers.Adam(
        learning_rate=parameters_info["init_lr"],
        beta_1=(parameters_info["adam_beta1"]),
        beta_2=(parameters_info["adam_beta2"]),
        # decay=parameters_info["decay"],
        amsgrad=True,
    )

    bls_model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=[
            metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    return bls_model


def gdbls_fixed():
    # Grand Descent Board Learning System with Dropout
    # This is a fixed model used to optimize parameters.

    model_input = Input((32, 32, 3))
    p1 = FeatureBlock(
        input=model_input,
        nb_convs=3,
        kernel=3,
        out_channels=160,
        use_ppv=True,
        div_n=8,
        data_format="channels_last",
    )
    p2 = FeatureBlock(
        input=model_input,
        nb_convs=3,
        kernel=3,
        out_channels=160,
        use_ppv=True,
        div_n=4,
        data_format="channels_last",
    )
    p3 = FeatureBlock(
        input=model_input,
        nb_convs=3,
        kernel=3,
        out_channels=160,
        use_ppv=True,
        div_n=2,
        data_format="channels_last",
    )

    # Flatten Layer
    z1 = Flatten()(p1)
    z2 = Flatten()(p2)
    z3 = Flatten()(p3)
    merged = concatenate([z1, z2, z3])

    # Overall Dropout Layer
    d1 = Dropout(rate=0.5)(merged)

    # Output Layer
    model_y = Dense(
        units=10,
        activation="softmax",
        kernel_initializer=initializers.lecun_normal(seed=None),
        kernel_regularizer=regularizers.l1(0.001),
    )(d1)

    # create the result model
    bls_model = Model(inputs=model_input, outputs=model_y)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=200000, decay_rate=0.95, staircase=True
    )
    adam = keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=True
    )

    bls_model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=[
            metrics.CategoricalAccuracy(name="accuracy"),
            metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    return bls_model
