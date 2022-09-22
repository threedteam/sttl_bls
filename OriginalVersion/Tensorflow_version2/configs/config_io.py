import json


# load json data from file
def load(file):
    with open(file, "r") as f:
        data = json.load(f)
        return data


# 将字典转化为字符串
def store(data, file):
    with open(file, "w") as fw:
        json.dump(data, fw)
