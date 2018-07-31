#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from flask import Flask, request, Response
from post_process import words_filter


app = Flask(__name__)

@app.route('/car_commands', methods=['POST', 'GET'])
def car_commands():
    print(request.headers)
    # print(request.get_data())
    # print("-----------------")
    request_data = json.loads(request.data)
    print(words_filter(request_data))

    return "Hello World~"


if __name__ == "__main__":
    # app.run(host='0.0.0.0', threaded=True)
    app.run(debug=True)


