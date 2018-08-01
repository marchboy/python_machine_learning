# /usr/bin/env python
# -*- coding:utf-8 -*-

import hashlib
import random
from http import client
from urllib import parse

appid = '20151113000005349'
secretKey = 'osubCEzlGjzvw8qdQc41'

httpClient = None
myurl = '/api/trans/vip/translate'
question = input("你要翻译什么呢：")
fromLang = 'zh'
toLang = 'en'
salt = random.randint(32768, 65536)

sign = appid + question + str(salt) + secretKey
m1 = hashlib.md5()
m1.update(sign.encode("utf-8"))
sign = m1.hexdigest()
my_url = myurl+'?appid='+appid+'&q='+parse.quote(question)+ \
         '&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

try:
    httpClient = client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', my_url)

    response = httpClient.getresponse()
    result = response.read()
    transfer = eval(result)
    print(transfer)
    dst_list = transfer.get("trans_result", None)
    dst = dst_list[0].get("dst")
    print(dst)

except Exception as e:
    print(e)
finally:
    if httpClient:
        httpClient.close()
