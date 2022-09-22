"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-04-11 14:50:04
LastEditors: Lv Di
LastEditTime: 2022-04-14 17:03:03
"""
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    Multiply,
    MaxPooling2D,
    Activation,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
)
from keras import initializers
from keras.regularizers import l2
from utils.ppvPooling2D import ppvPooling2D


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    l2_reg=1e-4,
    kernel_init_seed=None,
    batch_normalization=True,
    conv_first=True,
):
    """
    2D 卷积批量标准化 - 激活栈构建器
    parameters:
        inputs (tensor): 从输入图像或前一层来的输入张量\n
        num_filters (int): Conv2D 过滤器数量\n
        kernel_size (int): Conv2D 方形核维度\n
        strides (int): Conv2D 方形步幅维度\n
        activation (string): 激活函数名\n
        batch_normalization (bool): 是否包含批标准化\n
        conv_first (bool): conv-bn-activation (True) 或bn-activation-conv (False)\n

    :return x (tensor): 作为下一层输入的张量
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer=initializers.lecun_uniform(seed=kernel_init_seed),
        kernel_regularizer=l2(l2_reg),
    )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def FeatureBlock(
    input_tensor=None,
    nb_convs=3,
    kernel_replace=5,
    kernel_init_seed=None,
    l2_reg=1e-4,
    out_channels=64,
    use_ppv=False,
    div_n=16,
    data_format="channels_last",
    down_sample=True,
):
    tmp = input_tensor
    for i in range(nb_convs):
        if i < nb_convs - 1:
            tmp = resnet_layer(
                inputs=tmp,
                num_filters=out_channels // 2,
                kernel_size=3,
                strides=1,
                activation="relu",
                kernel_init_seed=kernel_init_seed,
                l2_reg=l2_reg,
                batch_normalization=True,
                conv_first=True,
            )
        else:
            tmp = resnet_layer(
                inputs=tmp,
                num_filters=out_channels,
                kernel_size=kernel_replace,
                strides=1,
                activation="relu",
                kernel_init_seed=kernel_init_seed,
                l2_reg=l2_reg,
                batch_normalization=True,
                conv_first=True,
            )
        if i < nb_convs - 1:
            tmp = Dropout(0.25)(tmp)

    # Squeeze&Extration block
    residual = BatchNormalization(axis=3)(tmp)
    if use_ppv is True:
        squeeze = ppvPooling2D(data_format=data_format)(residual)
    else:
        squeeze = GlobalAveragePooling2D(data_format=data_format)(residual)
    se_res = Dense(units=out_channels // div_n, activation="relu")(squeeze)
    se_res = Dense(units=out_channels, activation="sigmoid")(se_res)
    se_res = Reshape((1, 1, out_channels))(se_res)
    thres = Multiply()([residual, se_res])

    if down_sample:
        thres = MaxPooling2D(pool_size=(2, 2))(thres)

    output = Dropout(0.25)(thres)
    return output
