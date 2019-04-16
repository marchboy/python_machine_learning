#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import requests
# from post_process import __defaultType__

user_info = {
    "car_brand":"BMW",
    "car_model":"530",
    # "command_text":"打开天窗ssssziziz",
    "command_text":"请帮我打开天窗",
    # "command_text":"帮我把天窗关闭"
    # "command_text":"能帮我关闭前大灯嘛"
}



headers = {'content-type':'application/json'}
data = json.dumps(user_info)
print("This is RequestData:", data)
req = requests.post(
    "http://127.0.0.1:5678/car_commands",
    data=data,
    headers=headers
)

#print(req)
#print(req.text)
result = req.json()
print(result)
