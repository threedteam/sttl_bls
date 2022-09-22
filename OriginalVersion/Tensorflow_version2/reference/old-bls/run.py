"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-03-15 15:56:12
LastEditors: Lv Di
LastEditTime: 2022-03-15 16:36:02
"""
import sys
import time
from logger import Logger
from configs.config_reader import read_config

from tensorflow import keras
from keras import callbacks
import tensorflow as tf
from keras import backend as K

import json

from keras.callbacks import TensorBoard

import model.gdbls_cifar10 as cifar10
import model.gdbls_cifar100 as cifar100
import model.gdbls_svhn as svhn

from numpy.random import seed
from keras.backend import set_session

from logs.clear_logs import clear_checkpoints


def get_session():
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)
    return session


def run_exp(dataset, cfg, save_details=True, save_checkpoint=False):
    # set session information to custom GPU usage
    set_session(get_session())
    # set random seed to make the experiment more reliable.
    seed(7)
    # return a dictionary with config info.
    # load dataset info
    dataset_info = cfg["dataset"]
    # load parameters
    parameters_info = cfg["parameters"]
    init_lr = parameters_info["init_lr"]
    lr_reduction_factor = parameters_info["lr_reduction_factor"]
    # read settings from the config file
    settings_info = cfg["settings"]
    exp_name = settings_info["exp_name"] + " at " + time.strftime("%m-%d %H:%M:%S")

    log_pth = f"{settings_info['log_pth']}/{exp_name}.log"
    err_pth = f"{settings_info['err_pth']}/{exp_name}.log"
    checkpoint_pth = f"{settings_info['checkpoint_pth']}/{exp_name}/"
    model_png_pth = f"{settings_info['model_png_pth']}/{exp_name}.png"

    repeat_train = settings_info["repeat_train"]
    epochs_max = settings_info["epochs_max"]
    early_stop_patience = settings_info["early_stop_patience"]
    batch_size = settings_info["batch_size"]
    use_ppv = eval(settings_info["use_ppv"])

    sys.stdout = Logger(log_pth, sys.stdout)
    sys.stderr = Logger(err_pth, sys.stderr)

    # save current settings
    config = open(f"logs/exp_configs/{exp_name}.json", "w")
    json.dump(cfg, config)
    config.close()

    # load dataset and init record variables
    x_train, y_train, x_test, y_test, x_val, y_val = eval(dataset).get_data(
        settings_info["val_size"]
    )
    datagen = eval(dataset).image_generator()
    datagen.fit(x_train)
    datagen.fit(x_val)

    best_accuracy = -1
    sum_accuracy = 0
    total_train_time = 0

    for i in range(repeat_train):
        model = eval(dataset).gdbls(
            use_ppv=use_ppv, parameters_info=parameters_info, dataset_info=dataset_info
        )
        if i == 0:
            model.summary()
            keras.utils.plot_model(model, to_file=model_png_pth, show_shapes=True)

        """configure training settings"""
        checkpoint_callback = callbacks.ModelCheckpoint(
            checkpoint_pth,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        reduce = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_reduction_factor,
            patience=3,
            verbose=1,
            mode="auto",
            epsilon=0.0001,
            cooldown=0,
            min_lr=init_lr / 1000,
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=early_stop_patience
        )
        tensorboard_callback = TensorBoard(
            "./tensorboards/" + exp_name + "/" + time.strftime("%m-%d %H:%M:%S"),
            histogram_freq=1,
        )  # tensorboard visulation

        if save_details:
            cbs = [checkpoint_callback, early_stopping, tensorboard_callback, reduce]
        else:
            cbs = [checkpoint_callback, early_stopping, reduce]

        """test average network performance"""
        timeS = time.time()
        model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs_max,
            validation_data=datagen.flow(x_val, y_val, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            verbose=2,  # 1:motion 2:static
            callbacks=cbs,
        )
        timeE = time.time()

        print(
            "Train-",
            i + 1,
            ": The total training Time is : ",
            timeE - timeS,
            " seconds",
        )
        total_train_time += timeE - timeS

        timeS = time.time()
        model.load_weights(checkpoint_pth)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
        timeE = time.time()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
        sum_accuracy += accuracy

        print(f"Train-{i + 1}: Test accuracy: {round(accuracy * 100, 2)}%")
        print(
            "Train-", i + 1, ": The total testing Time is : ", timeE - timeS, " seconds"
        )

    print("average training time: ", (total_train_time / repeat_train))
    print("average acc: ", (sum_accuracy / repeat_train))
    print("best acc: ", best_accuracy)

    # clear redundant checkpoint file
    if not save_checkpoint:
        clear_checkpoints()
    # clear current ROM space to allocate for the next training
    K.clear_session()

    return sum_accuracy / repeat_train


def main(argv):
    if argv[0] == "cifar10":
        print("running gdbls on cifar10 dataset...")
    elif argv[0] == "cifar100":
        print("running gdbls on cifar100 dataset...")
    elif argv[0] == "svhn":
        print("running gdbls on svhn dataset...")
    else:
        print("ERROR: Dataset not supported.")
        sys.exit(2)
    run_exp(
        argv[0],
        read_config("configs/" + argv[0] + ".json")["config"],
        save_details=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
