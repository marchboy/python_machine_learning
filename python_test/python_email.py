# -*- coding: utf-8 -*-
# """
# # pi.py
# from random import random
# from math import sqrt
# from time import clock
#
# DARTS = 120000
# hits = 0
#
# clock()
# for i in range(1, DARTS):
#     x, y = random(), random()
#     dist = sqrt(x**2 + y**2)
#     if dist <= 1.0:
#         hits += 1
# pi = 4 * (hits/DARTS)
# print("Pi的值是 %s" % pi)
# print("程序运行的时间是 %-5.5ss" % clock())
# """


def make_incrementor(n):
    return lambda x: x+n

f = make_incrementor(42)
print(f)

print(f(0))
print(f(1))

pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair: pair[1])
print(pairs)

print(pairs[1])


import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header


# Add SMTP Server
mail_host = 'smtp.qq.com'
mail_user = '1042391064@qq.com'
mail_pwd  = 'jj@tencent2015'


receiver = ['jinjun.gui@flamingo-inc.com']

#content = os.system("sh run_gamedmp_hls_online_realtime.sh")
message = MIMEText('content....', 'plain', 'utf-8')
#message['From'] = mail_user
#message['To'] = receiver

subject = 'Email Test'
message['Subject'] = Header(subject, 'utf-8')


try:
    smtpobj = smtplib.SMTP(mail_host, 465)
    smtpobj.set_debuglevel(1)
    #smtpobj.connect(mail_host)
    smtpobj.login(mail_user, mail_pwd)
    smtpobj.sendmail(mail_user, receiver, message.as_string())
    smtpobj.close()
    print("发送成功！")
except smtplib.SMTPException:
    print('Error!')
