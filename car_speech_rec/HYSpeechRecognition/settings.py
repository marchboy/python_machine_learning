#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging


cur_path = os.path.dirname(os.path.realpath(__file__))
# log_path = os.path.join(os.path.dirname(cur_path), "logs")
log_path = os.path.join(cur_path, 'logs')

if not os.path.exists(log_path):
    os.mkdir(log_path)

class Log(object):
    def __init__(self, module_name):
        self.logname = os.path.join(
            log_path, 'car_audio_{}.log'.format(time.strftime("%Y%m%d")))
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(
            # '[%(asctime)s] --- %(funcName)s --- %(name)s --- %(levelname)s: %(message)s',
            '[%(asctime)s.%(msecs)03d] --- %(name)s --- %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def __console(self, level, message):
        # file_handler 输出到本地文件
        file_handler = logging.FileHandler(self.logname, encoding="utf-8")
        file_handler.setFormatter(self.formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # stream_handler 输出到控制台
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(self.formatter)
        # stream_handler.setLevel(logging.DEBUG)
        # self.logger.addHandler(stream_handler)

        if level == "info":
            self.logger.info(message)
        if level == "debug": 
            self.logger.debug(message)
        if level == "warning":
            self.logger.warning(message)
        if level == "error":
            self.logger.error(message)
        if level == "exception":
            self.logger.exception(message)

        self.logger.removeHandler(file_handler)
        # self.logger.removeHandler(stream_handler)
        file_handler.close()
        # stream_handler.close()

    def debug(self, message):
        self.__console("debug", message)

    def info(self, message):
        self.__console("info", message)

    def warning(self, message):
        self.__console("warning", message)

    def error(self, message):
        self.__console("error", message)

    def exception(self, message):
        self.__console("exception", message)

# if __name__ == '__main__':
#     print(cur_path)
#     print(os.path.dirname(cur_path))
#     print(log_path)
