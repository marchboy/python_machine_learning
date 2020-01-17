# -*- coding: utf-8 -*-

import os
import re
import time
import logging
import smtplib
import threading
import subprocess

from urllib import request
from urllib.error import URLError
from multiprocessing import Process

from email.mime.text import MIMEText
from email.header import Header


class AppiumServer:
    def __init__(self, port):
        self.port = port

    def start_server(self):
        """
        start appium server
        :return:
        """
        t1 = RunServer('appium -p %s' % self.port)
        p = Process(target=t1.start())
        p.start()

    def stop_server(self):
        """
        stop appium server
        """
        result = subprocess.getoutput('netstat -ano | findstr %s' % self.port)
        if result:
            array = re.split(r'\s+', result.strip())
            os.system('taskkill /pid %s -t -f' % array[4])

    def restart_server(self):
        self.stop_server()
        self.start_server()

    def is_running(self):
        response = None
        url = " http://127.0.0.1:"+str(self.port)+"/wd/hub"+"/status"
        try:
            response = request.urlopen(url, timeout= 5)

            if str(response.getcode()).startswith("2"):
                return True
            else:
                return False

        except URLError:
            return False
        
        finally:
            if response:
                response.close


class RunServer(threading.Thread):
    def __init__(self, cmd):
        threading.Thread.__init__(self)
        self.cmd = cmd

    def run(self):
        with open("./dingtalk_logs/running_appium_server.log", "w") as file_handle:
            subprocess.Popen(self.cmd, shell=True, bufsize=0, stdout=file_handle)


class Log(object):
    def __init__(self, module_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cur_path = os.path.dirname(os.path.realpath(__file__))
        self.log_path = os.path.join(self.cur_path, 'dingtalk_logs')
        self.logname = os.path.join(self.log_path, 'dingtalk_clock_log.log')
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(
            '[%(asctime)s] --- %(name)s --- %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def __console(self, level, message):
        # output to local file
        file_handler = logging.FileHandler(self.logname)
        file_handler.setFormatter(self.formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # output to console  
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)

        if level == 'info':
            self.logger.info(message)
        if level == 'debug':
            self.logger.info(message)
        if level == 'warning':
            self.logger.warning(message)
        if level == 'error':
            self.logger.error(message)
        if level == "exception":
            self.logger.exception(message)
        self.logger.removeHandler(stream_handler)
        self.logger.removeHandler(file_handler)

        file_handler.close()

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


def send_mail(email_content):
    mail_host="smtp.163.com" 
    mail_user="jjgdut@163.com"
    mail_pass="jj@kaola"

    sender = mail_user
    receivers = ['jjgdut@163.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    message = MIMEText(email_content, 'plain', 'utf-8')
    message['From'] = Header('DingLa', 'utf-8')
    message['To'] =  Header("Notification", 'utf-8')

    subject = "Secret info notifications"
    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtpObj = smtplib.SMTP() 
        smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
        smtpObj.login(mail_user,mail_pass) 
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("Email: Success")
        
    except smtplib.SMTPException:
        print("Error: Failed Send")


# Constant Settings

DESIRED_CAPS = {
    "platformName": "Android",
    "deviceName": "T8DDU16406001061",
    "appPackage": "com.alibaba.android.rimet",
    "appActivity": "com.alibaba.android.rimet.biz.LaunchHomeActivity",
    "noReset": "True",
    'newCommandTime': 8000000
}

DESIRED_CAPS_HUAWEI = {
    "platformName": "Android",
    "deviceName": "P7C0217C28012302",
    "appPackage": "com.alibaba.android.rimet",
    "appActivity": "com.alibaba.android.rimet.biz.LaunchHomeActivity",
    "noReset": "True"   #启动app时不要清除app里的原有的数据
}

LOGIN_INFO = {
    "user":'15989012443',
    "passwd":"jj@dingding"
}





