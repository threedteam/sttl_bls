import pandas as pd
import json
from model.gdbls_cifar10 import GDBLS
import os


def store(data):
    with open("data.json", "w") as fw:
        # 将字典转化为字符串
        # json_str = json.dumps(data)
        # fw.write(json_str)
        # 上面两句等同于下面这句
        json.dump(data, fw)


# load json data from file
def load(file):
    with open(file, "r") as f:
        data = json.load(f)
        return data


def read_config(conifg_file):
    print(f"\n\n{'-' * 20}\nINFO: executing task:{conifg_file}")
    df = pd.read_json(conifg_file)
    return df.to_dict()


# print(os.getcwd())
configs = load("configs/cifar10.json")
# print(configs["kernels"][0])
model = GDBLS(configs)
print(model)