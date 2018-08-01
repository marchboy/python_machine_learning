#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from flask import Flask, request, Response
from post_process import words_filter, access_db
from settings import Log


app = Flask(__name__)

@app.route('/car_commands', methods=['POST', 'GET'])
def car_commands():
    # print(request.headers)
    # print(request.get_data())
    # print("-----------------")

    request_data = json.loads(request.data)
    logs = Log("RequestParas")
    logs.info(request_data)

    car_command = words_filter(request_data)
    CAN_command = access_db(car_command)

    result = {"CAN_command":CAN_command}
    logs = Log("CarCommand")
    logs.info(result)

    response = Response(json.dumps(result), mimetype='application/json')

    response.headers.add('Server', 'CarAudioPlay')
    return response


if __name__ == "__main__":
    # app.run(host='0.0.0.0', threaded=True)
    app.run(debug=True, host='127.0.0.1', port=5678)
    # app.run()


