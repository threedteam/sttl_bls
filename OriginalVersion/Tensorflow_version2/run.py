"""
Descripttion:
version:
Author: Lv Di
Date: 2022-04-19 11:09:29
LastEditors: Lv Di
LastEditTime: 2022-04-19 15:10:58
"""
import os
import shutil
import sys
import time
import json
from numpy.random import seed
import pandas as pd
from utils.recorder import Logger
from utils.cleaner import clear_checkpoints

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import callbacks
from keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from keras.backend import set_session

from configs.config_io import load, store
import model.gdbls_cifar10 as cifar10


# import model.gdbls_cifar100 as cifar100
# import model.gdbls_svhn as svhn


class run:
    def __init__(self, dataset_name, save_details=True, exp_name="Unnamed_exp", save_files=True):
        self.save_files = save_files
        self.dataset_name = dataset_name
        self.save_details = save_details

        # initilize experiment condition.
        # in default we use GPU 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=tf_config)
        set_session(session)
        seed(7)

        # load config file
        self.config = self._load_config()
        self.exp_name = exp_name + " at " + time.strftime("%m-%d %H:%M:%S")

        # related resources for the training process.
        (
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            self.x_val,
            self.y_val,
        ) = [-1 for i in range(6)]
        self.datagen = None
        self.batch_size = None

    def _load_config(self, config_dir="configs/"):
        return load(config_dir + self.dataset_name + ".json")

    def _custom_config(self, customs=[]):
        pass

    def cbs_def(self):
        checkpoint_cb = ModelCheckpoint(
            self.config["checkpoint_pth"] + "/" + self.exp_name,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        reduce_cb = ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.config["lr_reduction_factor"],
            patience=3,
            verbose=1,
            mode="auto",
            epsilon=0.0001,
            cooldown=0,
            min_lr=self.config["min_lr"],
        )
        earlystop_cb = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=self.config["early_stop_patience"]
        )
        tensorboard_cb = TensorBoard(
            self.config["tensorboard_pth"]
            + "/"
            + self.exp_name
            + "/"
            + time.strftime("%m-%d %H:%M:%S"),
            histogram_freq=1,
        )
        if self.save_details:
            cbs = [checkpoint_cb, earlystop_cb, tensorboard_cb, reduce_cb]
        else:
            cbs = [checkpoint_cb, earlystop_cb, reduce_cb]

        return cbs

    def _invoke(self, model_obj, ind):
        gdbls_model = model_obj.gen_model()
        if ind == 0:
            gdbls_model.summary()
            keras.utils.plot_model(gdbls_model, to_file=self.config['model_png_pth'] + f"/{self.exp_name}.png",
                                   show_shapes=True)

        self.datagen.fit(self.x_train)
        self.datagen.fit(self.x_val)
        """test average network performance"""
        timeS = time.time()
        gdbls_model.fit(
            self.datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
            epochs=self.config["epochs_max"],
            validation_data=self.datagen.flow(
                self.x_val, self.y_val, batch_size=self.batch_size
            ),
            steps_per_epoch=self.x_train.shape[0] // self.batch_size,
            verbose=2,  # 1:motion 2:static
            callbacks=self.cbs_def(),
        )
        timeE = time.time()
        invoke_time_cost = timeE - timeS

        """test accuracy for the invoke"""
        timeS = time.time()
        gdbls_model.load_weights(self.config["checkpoint_pth"] + "/" + self.exp_name)
        test_loss, test_accuracy = gdbls_model.evaluate(
            self.x_test, self.y_test, verbose=2
        )
        timeE = time.time()
        test_time_cost = timeE - timeS

        print(
            "-" * 20 + f"\nINFO: {ind + 1}th invoke complete after {invoke_time_cost} seconds.\n "
                       f"acc={test_accuracy},loss={test_loss}\n" + "-" * 20)

        return invoke_time_cost, test_time_cost, test_loss, test_accuracy

    def train_test(self):
        # empty model object to get data and augmentation generator
        model = eval(self.dataset_name).GDBLS(self.config)
        (
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            self.x_val,
            self.y_val,
        ) = model.get_data(
            self.config["val_size"], eval(self.config["subtract_pixel_mean"])
        )
        self.cb = self.cbs_def()
        self.batch_size = self.config["batch_size"]
        self.datagen = model.data_aug()

        config = open(self.config["exp_configs"] + f"/{self.exp_name}.json", "w")
        json.dump(self.config, config)
        config.close()

        # record training process.
        best_acc = -1
        sum_acc = 0
        best_record = None

        sum_train_time = 0
        sum_test_time = 0

        sys.stdout = Logger(
            self.config["log_pth"] + f"/{self.exp_name}_normal.log", sys.stdout
        )
        sys.stderr = Logger(
            self.config["err_pth"] + f"/{self.exp_name}_errs.log", sys.stderr
        )

        for i in range(self.config["repeat_train"]):
            model_for_invoke = eval(self.dataset_name).GDBLS(self.config)
            invoke_time_cost, test_time_cost, test_loss, test_accuracy = self._invoke(
                model_for_invoke, i
            )
            sum_train_time += invoke_time_cost
            sum_test_time += test_time_cost
            sum_acc += test_accuracy
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best_record = f"loss={test_loss}, acc={test_accuracy}, invoke_time={invoke_time_cost}, test_time={test_time_cost}"

        return (
            sum_train_time / self.config["repeat_train"],
            sum_test_time / self.config["repeat_train"],
            sum_acc / self.config["repeat_train"],
            best_record,
        )

    def explain(self):
        avg_train_time, avg_test_time, avt_acc, best_rec = self.train_test()
        print(f"avg_train_time={avg_train_time}")
        print(f"avg_test_time={avg_test_time}")
        print(f"avg_acc={avt_acc}")
        print(f"best_rec: {best_rec}")
        if not self.save_files:
            self._clear_recs()

    def _clear_recs(self):
        rec_dirs = [
            self.config["log_pth"],
            self.config["err_pth"],
            self.config["checkpoint_pth"],
            self.config["tensorboard_pth"],
            self.config["exp_configs"],
            self.config["model_png_pth"]
        ]
        for i in rec_dirs:
            shutil.rmtree(i)
            os.mkdir(i)

    # reserved interfaces for grid search
    def _gen_grid(self, opts):
        pass

    def grid_train(self, grids):
        pass


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

    exps = run(argv[0], save_details=True,
               exp_name=("UNNAMED_EXP" if len(argv) == 1 else argv[1]), save_files=True)
    exps.explain()


if __name__ == "__main__":
    main(sys.argv[1:])
