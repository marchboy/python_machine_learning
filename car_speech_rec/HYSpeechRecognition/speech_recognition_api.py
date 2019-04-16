#!/usr/bin/python
# -*- coding: UTF-8 -*-
import urllib2
import time
import urllib
import json
import hashlib
import base64

#-----------------------科大讯飞语音API--------------------------

def speech_recongnition_iflytec(file_content):
    APPID = '5b67f6ea'
    API_KEY = 'a347be7edc6a640ec62077c0a0ae5623'
    # APPID = '5b6a94bd'
    # API_KEY = 'a35ff8efc9f6ee22692d26c2634c194e'

    # f = open(AUDIO_PATH, 'rb')
    # file_content = f.read()
    base64_audio = base64.b64encode(file_content)
    body = urllib.urlencode({'audio': base64_audio})

    url = 'http://api.xfyun.cn/v1/service/v1/iat'
    api_key = API_KEY
    param = {"engine_type": "sms16k", "aue": "raw"}

    x_appid = APPID
    x_param = base64.b64encode(json.dumps(param).replace(' ', ''))
    x_time = int(int(round(time.time() * 1000)) / 1000)
    x_checksum = hashlib.md5(api_key + str(x_time) + x_param).hexdigest()
    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    req = urllib2.Request(url, body, x_header)
    result = urllib2.urlopen(req)
    
    result = result.read().decode("UTF-8")
    res = json.loads(result).get("data")

    return  res
