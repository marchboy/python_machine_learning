#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created on 27/7/2018
# Author: jinjun


import pymysql
from db_config import MYSQL_INFO
from word_analysis import WordAnalysis


def access_db(command):
    conn = pymysql.connect(**MYSQL_INFO)
    cursor = conn.cursor()

    sql = "SELECT CAN_commands FROM manipulation_commands where command_text = '{}' and car_brand = '{}' and " \
          "car_model = '{}'; ".format(command.get("command_text"), command.get("car_brand"), command.get("car_model"))
    cursor.execute(sql)

    for can_info in cursor:
        print(can_info)

    cursor.close()
    conn.close()


def words_filter(args):
    path = "E:\pythonProject\python_machine_learning\CarAudio\stopwords"
    filename = "中文停用词表.txt"

    wa = WordAnalysis(path, filename)
    command_text = args.pop("command_text")
    
    words = wa.words_seg(command_text)
    word_list = wa.removestopwords(words)
    args["command_text"] = "".join(word_list)

    return args
    



# if __name__ == "__main__":
#     command = "打开天窗"
#     access_db(command)