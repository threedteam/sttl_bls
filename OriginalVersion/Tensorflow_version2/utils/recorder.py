"""
Descripttion: 
version: 
Author: Lv Di
Date: 2022-04-19 11:42:11
LastEditors: Lv Di
LastEditTime: 2022-04-19 15:07:11
"""
import sys


class Logger(object):
    def __init__(self, file_name="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
