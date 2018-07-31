#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import requests
# from post_process import __defaultType__

user_info = {
    "car_brand":"BMW",
    "car_model":"530",
    # "command_text":"打开天窗ssssziziz",
    # "command_text":"请帮我打开天窗",
    "command_text":"我要关闭天窗aaa啊啊啊"
}



headers = {'content-type':'application/json'}
data = json.dumps(user_info)
print(data)
req = requests.post(
    "http://127.0.0.1:5000/car_commands",
    data=data,
    headers=headers
)

print(req)
print(req.text)
# result = req.json()

# print(retext