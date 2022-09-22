"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-04-19 11:42:29
LastEditors: Lv Di
LastEditTime: 2022-04-19 15:08:04
"""
import os
import shutil


def clear_all():
    dirs = ["logs/errs", "logs/normals", "logs/pths", "logs/tensorboards"]

    for i in dirs:
        shutil.rmtree(i)
        os.mkdir(i)


def clear_checkpoints():
    print(f"INFO: clearing saved checkpoints...")
    shutil.rmtree("logs/pths")
    os.mkdir("logs/pths")
    print(f"INFO: clearing saved checkpoints complete.")
