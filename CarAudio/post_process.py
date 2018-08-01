#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created on 27/7/2018
# Author: jinjun


import pymysql
import os
import sys
# import MySQLdb
from settings import MYSQL_INFO
from word_analysis import WordAnalysis

# reload(sys)
# sys.setdefaultencoding('utf-8')

def access_db(command):
    # conn = MySQLdb.connect(**MYSQL_INFO)
    conn = pymysql.connect(**MYSQL_INFO)
    cursor = conn.cursor()

    sql = "SELECT CAN_commands FROM manipulation_commands where command_text = '{}' and car_brand = '{}' and " \
          "car_model = '{}'; ".format(command.get("command_text"), command.get("car_brand"), command.get("car_model"))
    cursor.execute(sql)

    content = '' 
    for can_info in cursor:
        content = can_info[0]

    cursor.close()
    conn.close()
    return content

def words_filter(args):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(cur_path, 'stopwords')
    #path = "E:\pythonProject\python_machine_learning\CarAudio\stopwords"
    filename = "ChineseStopwords.txt"

    wa = WordAnalysis(log_path, filename)
    command_text = args.pop("command_text")
    
    words = wa.words_seg(command_text)
    # words = wa.fetch_keywords(command_text)
    # words = wa.fetch_keywords_tfidf(command_text)
    print("after word segment: ", words)

    word_list = wa.removestopwords(words)
    print("after filter stopwords: ", word_list)
    
    args["command_text"] = "".join(word_list)

    return args
    



# if __name__ == "__main__":
#     command = "打开天窗"
#     access_db(command)
