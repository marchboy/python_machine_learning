#!/usr/bin/env python
#-*-coding:utf-8 -*-

import json
import sys
import os
sys.path.append('./gen-py')
from datetime import datetime
from hySpeechRecognition import HYSpeechRecognition
from thrift import Thrift
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from threading import Thread, Semaphore


class ThriftRPC(object):

    def __init__(self, host, port):
        tsocket = TSocket.TSocket(host, port)
        transport = TTransport.TBufferedTransport(tsocket)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        client = HYSpeechRecognition.Client(protocol)
        transport.open()

        self.client_ = client
        self.transport_ = transport

    def close(self):
        self.transport_.close()
    
    def get_speech_recognition_result(self, filename, buff):
        result = None
        if self.transport_.isOpen():
            result_str = self.client_.hySpeechRecognition(filename, buff)
            ret = json.loads(result_str)
            
            err_code = ret.get("errcode")
            if err_code == 0:
                result = ret.get("response_data")
                r_type = ret.get("response_type")
            else:
                result = 'response error! code:{}'.format(err_code)
        else:
            result = 'rpc error!'
        return result, r_type

def send_file(FILENAME):
    rpc = ThriftRPC(RPCHOST, RPCPORT)
    try:
        file = open(FILENAME, 'rb')
        data = file.read()
        result, r_type = rpc.get_speech_recognition_result(FILENAME, data)
        rpc.close()
        print(result, r_type)
        # return result
    except:
        print("open file error!")

if __name__ =="__main__":
    RPCHOST = '192.168.2.233'
    RPCPORT = 8989
    FILENAME = 'REC2.WAV'
    try:

        for i in range(5):
            t = Thread(target=send_file, args=(FILENAME,))
            t.start()

        # res = send_file(FILENAME)
        # print(res)
    except StopIteration as e:
        result_str = e.args
        print("received:{}".format(result_str))

