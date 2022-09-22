"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-04-11 14:50:04
LastEditors: Lv Di
LastEditTime: 2022-04-14 21:08:59
"""
from tensorflow.python.keras.engine.base_layer import Layer
from utils.Backends import mean_backend, cast_backend, greater_backend


class ppvPooling2D(Layer):
    def __init__(self, data_format="channels_last", bias=0, keepdims=False, **kwargs):
        super(ppvPooling2D, self).__init__(**kwargs)
        self.data_format = data_format
        self.bias = bias
        self.keepdims = keepdims

    def call(self, input):
        if self.data_format == "channels_last":
            return mean_backend(
                cast_backend(greater_backend(input, self.bias), float),
                axis=[1, 2],
                keepdims=self.keepdims,
            )
        else:
            return mean_backend(
                cast_backend(greater_backend(input, self.bias), float),
                axis=[2, 3],
                keepdims=self.keepdims,
            )
