"""
Descripttion:
version:
Author: Lv Di
Date: 2022-04-19 11:10:03
LastEditors: Lv Di
LastEditTime: 2022-04-19 19:31:46
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import backend as K
from keras import initializers, regularizers, metrics, Model
from keras.datasets import cifar10
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Input,
    Flatten,
    concatenate,
    Dropout,
    Multiply,
    MaxPooling2D,
    Activation,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
    ReLU,
    PReLU,
    subtract,
    minimum,
    Lambda,
    Concatenate,
    UpSampling2D,
    add
)
from keras.preprocessing.image import ImageDataGenerator
from custom_layers.ppvPooling2D import ppvPooling2D

def cal_mean(inputs):
    outputs = K.mean(inputs, axis=1, keepdims=True)
    return outputs

class GDBLSBase:
    def __init__(self, settings):
        # dataset parameters
        self.input_shape = settings["input_shape"]
        self.target_class = settings["target_class"]

        # structure parameters
        self.fbs = settings["fbs"]
        self.nb_convs = settings["nb_convs"]

        self.filters = settings["filters"]
        # must provide detailed filters
        """example:
        [
            [64,64,128],
            [128,128,256],
            [256,256,512]
        ]
        """
        assert len(self.filters[0]) == self.nb_convs

        self.kernels = settings["kernels"]
        # must provide detailed kernels
        """example:
        [
            [3,3,3],
            [3,3,3],
            [3,3,5]
        ]
        """
        assert len(self.kernels[0]) == self.nb_convs

        self.se_div = settings["se_div"]
        # must provide detailed se_divs
        """example:
        [
            4,4,4
        ]
        """
        assert len(self.se_div) == self.fbs

        self.use_ppv = eval(settings["use_ppv"])
        self.use_amsgrad = eval(settings["use_amsgrad"])

        # training parameters
        self.overall_dropout_rate = settings["overall_dropout_rate"]
        self.inside_dropout_rate = settings["inside_dropout_rate"]
        self.stride = settings["stride"]
        self.init_seed = eval(settings["init_seed"])  # None or fixed values
        self.regl2 = settings["regl2"]
        self.init_lr = settings["init_lr"]
        self.beta_1 = settings["beta_1"]
        self.beta_2 = settings["beta_2"]
        self.act = settings["activation"]

    def __str__(self):
        description = "this is a gdbls model with these options: \n"
        for k, v in vars(self).items():
            description += f"option {k} = {v}\n"
        return description

    def _aprelu(self, inputs):
        # zero feature map
        zeros_input = subtract([inputs, inputs])
        # positive feature map
        pos_input = Activation('relu')(inputs)
        # negative feature map
        neg_input = minimum([inputs, zeros_input])

        # define a network to obtain the scaling coefficients
        scales_p = Lambda(cal_mean)(GlobalAveragePooling2D()(pos_input))
        scales_n = Lambda(cal_mean)(GlobalAveragePooling2D()(neg_input))
        scales = Concatenate()([scales_n, scales_p])
        scales = Dense(2, activation='linear', kernel_initializer='he_normal'
                       , kernel_regularizer=regularizers.l2(self.regl2))(scales)
        scales = BatchNormalization(momentum=0.9, gamma_regularizer=regularizers.l2(self.regl2))(scales)
        scales = Activation('relu')(scales)
        scales = Dense(1, activation='linear', kernel_initializer='he_normal'
                       , kernel_regularizer=regularizers.l2(self.regl2))(scales)
        scales = BatchNormalization(momentum=0.9, gamma_regularizer=regularizers.l2(self.regl2))(scales)
        scales = Activation('sigmoid')(scales)
        scales = Reshape((1, 1, 1))(scales)

        # apply a paramtetric relu
        neg_part = keras.layers.multiply([scales, neg_input])
        return keras.layers.add([pos_input, neg_part])

    def _resnet_layer(self, inputs, ind, i, batch_normalization, conv_first):
        conv = Conv2D(
            self.filters[ind][i],
            kernel_size=self.kernels[ind][i],
            strides=self.stride,
            padding="same",
            kernel_initializer=initializers.lecun_uniform(seed=self.init_seed),
            kernel_regularizer=regularizers.l2(self.regl2),
        )

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if self.act == "aprelu":
                x = self._aprelu(x)
            else:
                x = Activation(self.act)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if self.act == "aprelu":
                x = self._aprelu(x)
            else:
                x = Activation(self.act)(x)
            x = conv(x)
        return x

    def _feature_block(self, inputs, ind, inside_dropout=True):
        x = inputs
        for i in range(self.nb_convs):
            x = self._resnet_layer(
                x, ind, i, batch_normalization=True, conv_first=True
            )
            if i < self.nb_convs - 1 and inside_dropout:
                x = Dropout(self.inside_dropout_rate)(x)

        # Squeeze&Extration block
        out_channels = self.filters[ind][-1]
        residual = BatchNormalization(axis=3)(x)
        if self.use_ppv is True:
            squeeze = ppvPooling2D(data_format=K.image_data_format())(residual)
        else:
            squeeze = GlobalAveragePooling2D(data_format=K.image_data_format())(residual)
        se_res = Dense(units=out_channels // self.se_div[ind], activation="relu")(squeeze)
        se_res = Dense(units=out_channels, activation="sigmoid")(se_res)
        se_res = Reshape((1, 1, out_channels))(se_res)
        thres = Multiply()([residual, se_res])
        thres = MaxPooling2D(pool_size=(2, 2))(thres)
        output = Dropout(self.inside_dropout_rate)(thres)
        return output

    def get_data(self, val_size=0.33, subtract_pixel_mean=False):
        pass

    def data_aug(self):
        pass

    def gen_model(self):
        pass
