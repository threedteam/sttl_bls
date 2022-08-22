from flask import Flask, request
import json
import os
import pandas as pd

# 创建flask的应用对象
# __name__表示当前的模块名称
# 模块名: flask以这个模块所在的目录为根目录，默认这个目录中的static为静态目录，templates为模板目录
app = Flask(__name__)


def save2csv(args, data_path):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df.append(args, ignore_index=True)
        print(df)
        df.to_csv(data_path, encoding='utf-8', index_label=False)
    else:
        df = pd.DataFrame(args, index=[0])
        print(df)
        df.to_csv(data_path, encoding='utf-8', index_label=False)


def update(args):
    print(args)
    exp_name = args['exp_name']
    data_paths = [f'saves/{exp_name}.csv', 'saves/data.csv']
    for path in data_paths:
        save2csv(args, path)


# 定义url请求路径
@app.route('/report', methods=["POST"])
def accept():
    get_data = dict(request.json)
    update(get_data)
    return 'Stat Successfully Accepted.'


if __name__ == '__main__':
    # if os.path.exists('saves/data.csv'):
    #     os.remove('saves/data.csv')
    # 启动flask
    app.run(port=8003, debug=True)
