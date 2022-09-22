"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-04-14 17:13:58
LastEditors: Lv Di
LastEditTime: 2022-04-14 21:08:45
"""
"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-04-14 17:13:58
LastEditors: Lv Di
LastEditTime: 2022-04-14 17:22:54
"""

from keras import backend as K


def mean_backend(inputs, axis=[1, 2], keepdims=True):
    return K.mean(
        inputs,
        axis=axis,
        keepdims=keepdims,
    )


def cast_backend(input, target):
    return K.cast(input, target)


def greater_backend(input, bias):
    return K.greater(input, bias)


def abs_backend(inputs):
    return K.abs(inputs)


def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs, 1), 1)


def sign_backend(inputs):
    return K.sign(inputs)


def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels) // 2
    inputs = K.expand_dims(inputs, -1)
    inputs = K.spatial_3d_padding(
        inputs, ((0, 0), (0, 0), (pad_dim, pad_dim)), "channels_last"
    )
    return K.squeeze(inputs, -1)
