"""
Descripttion:
version:
Author: Lv Di
Date: 2022-04-19 11:09:48
LastEditors: Lv Di
LastEditTime: 2022-04-19 19:18:24
"""
from model.gdbls_base import *


class GDBLS(GDBLSBase):
    def __init__(self, settings):
        super().__init__(settings)

    def get_data(self, val_size=0.33, subtract_pixel_mean=False):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_size, random_state=42
        )

        mean = np.array([125.3, 123.0, 113.9])
        std = np.array([63.0, 62.1, 66.7])
        x_train = (x_train.astype("float") - mean) / std
        x_val = (x_val.astype("float") - mean) / std
        x_test = (x_test.astype("float") - mean) / std

        # x_train = x_train.astype('float32') / 255.
        # x_test = x_test.astype('float32') / 255.
        # x_test = x_test - np.mean(x_train)
        # x_train = x_train - np.mean(x_train)

        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean

        y_train = keras.utils.to_categorical(y_train, 10)
        y_val = keras.utils.to_categorical(y_val, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        print(f"shape for training :{x_train.shape} and {y_train.shape}")
        print(f"shape for validation :{x_val.shape} and {y_val.shape}")
        print(f"shape for test :{x_test.shape} and {y_test.shape}")

        return x_train, y_train, x_test, y_test, x_val, y_val
        # return x_train, y_train, x_test, y_test

    def data_aug(self):
        # datagen = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False,
        # )
        datagen = ImageDataGenerator(
            # randomly rotate images in the range (deg 0 to 150)
            rotation_range=10,
            # Range for random zoom
            zoom_range=0.20,
            # # shear angle in counter-clockwise direction in degrees
            # shear_range=15,
            # randomly flip images
            horizontal_flip=True,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
        )
        return datagen

    def gen_model(self):
        model_input = Input(self.input_shape)

        p1 = self._feature_block(model_input, 0, inside_dropout=True)
        p2 = self._feature_block(p1, 1, inside_dropout=True)
        p3 = self._feature_block(p2, 2, inside_dropout=True)

        # Flatten Layers
        z1 = Flatten()(p1)
        p2 = UpSampling2D(size=(2, 2))(p2)
        p2 = Conv2D(filters=128, kernel_size=(1, 1))(p2)
        z2 = Flatten()(p2)
        p3 = UpSampling2D(size=(4, 4))(p3)
        p3 = Conv2D(filters=128, kernel_size=(1, 1))(p3)
        z3 = Flatten()(p3)
        merged = add([z1, z2, z3])

        # Overall Dropout Layer
        d1 = Dropout(self.overall_dropout_rate)(merged)

        # Output Layer
        model_y = Dense(
            10,
            activation="softmax",
            kernel_initializer=initializers.lecun_uniform(seed=self.init_seed),
            kernel_regularizer=regularizers.l2(self.regl2),
        )(d1)

        # create the result model
        bls_model = Model(inputs=model_input, outputs=model_y)

        # adam optimizer / amsgrad variant
        adam = keras.optimizers.Adam(
            learning_rate=self.init_lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            amsgrad=self.use_amsgrad,
        )

        bls_model.compile(
            optimizer=adam,
            loss="categorical_crossentropy",
            metrics=[
                metrics.CategoricalAccuracy(name="accuracy"),
            ],
        )

        return bls_model
