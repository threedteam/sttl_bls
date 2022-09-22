"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-03-15 15:46:55
LastEditors: Lv Di
LastEditTime: 2022-03-15 16:34:53
"""
import pandas as pd


def read_config(conifg_file):
    print(f"\n\n{'-'*20}\nINFO: executing task:{conifg_file}")
    df = pd.read_json(conifg_file)
    return df.to_dict()
