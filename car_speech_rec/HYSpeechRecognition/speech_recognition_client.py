#!/usr/bin/env python
#-*- coding: utf-8 -*-

import json
import sys
import time
sys.path.append('./gen-py')

from hySpeechRecognition import HYSpeechRecognition
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

def file_analysis_service(client, filename):
    print("-------------------正在上传语音-------------------")
    audio_text = ""
    with open(filename, 'rb') as f:
        content = f.read()
        audio_text = client.hySpeechRecognition(filename, content)
    result = json.loads(audio_text)
    print("result:", result)
    errcode = result.get("errcode")
    print(result.get("response_data"))
    if errcode == 0:
        print("--------------语音上传成功，解析完成--------------")
    elif errcode == 1:
        print("-------------语音格式不正确，上传失败-------------")
    else:
        print("--------------语音上传成功，解析失败--------------")


def run_asr(filename):
    try:
        tsocket = TSocket.TSocket(host="192.168.2.233" , port=8989)
        transport = TTransport.TBufferedTransport(tsocket)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        
        client = HYSpeechRecognition.Client(protocol)
        transport.open()

        file_analysis_service(client, filename)

        transport.close()
    except Thrift.TException as tx:
        print('Wrong Message: %s' % (tx.message))


if __name__ == "__main__":
    file_path = "./wav_bak/"
    for i in range(0, 38):
        file_name = "REC{}.WAV".format(i)
        filename = file_path + file_name
        print(filename)
        t1 = time.time()
        run_asr(filename)
        print(time.time()-t1)
