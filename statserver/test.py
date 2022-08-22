args = {
    'exp_name': None,
    'train_time': None,
    'val_time_epoch': None,
    'acc': None,
    'loss': None
}
import requests
import time
import uuid
import random


def test1():
    characters = []
    for i in range(10):
        json_data = {
            'exp_name': uuid.uuid4().__str__(),
            'train_time': time.time(),
            'val_time_epoch': random.randint(1, 5),
            'acc': random.randrange(90, 95, 1) / 100,
            'loss': random.randrange(10, 30, 1) / 10
        }
        r = requests.post("http://127.0.0.1:8003/report", json=json_data)
        print(f'send data:{json_data}')
        print(r.headers)
        print(r.text)


if __name__ == '__main__':
    test1()
