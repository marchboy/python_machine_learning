#!/usr/bin/env python
# -*- conding: utf-8 -*-

import sys
import os
import json
import time
import commands
import Queue
from datetime import datetime
from multiprocessing.dummy import Process as thread
from multiprocessing.dummy import Semaphore

reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append('./gen-py')

from hySpeechRecognition import HYSpeechRecognition
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer, TProcessPoolServer

from speech_recognition_api import speech_recongnition_iflytec
from settings_1 import Log


class AudioDataHandler(object):
    def __init__(self, file_path=None, content=None):
        self.file_path = "./wav/wav_" + time.strftime("%Y%m%d") + "/"

    def __requestsave__(self, filename, logger_name=None):
        global unixtime
        unixtime = int(time.time() * 1000)
        try:
            filename = filename.split("/")[-1] if "/" in filename else filename
            request_data = {"request_file": filename, "request_unixtime":unixtime, "errcode":0}
            return request_data
            # logs = Log(logger_name)
            # logs.info(request_data)
        except Exception as e:
            sys.stdout.write(e)
            response_data = {"response_data": "Wrong File Format", "errcode": 1}
            response_data = json.dumps(response_data)
            return response_data

    def __resultsave__(self, filename, content):
        # unixtime = int(time.time() * 1000)
        filename = filename.split("/")[-1] if "/" in filename else filename
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)
        filepname = self.file_path + str(unixtime) + "_" + filename
        with open(filepname, "wb") as f:
            sys.stdout.write("File writing......\n")
            f.write(content)
        sys.stdout.write("File Saved\n")
        return filepname

    def request_ifly_sdk(self, filename, content, s, resultQueue):
        t1 = datetime.now()
        sys.stdout.write("request_ifly_sdk starting:" + str(t1) + "\n")
        resp_data = self.__requestsave__(filename=filename, logger_name="RequestDataSDK")
        errcode = resp_data.pop("errcode")
        if errcode:
            return resp_data

        res_filename = self.__resultsave__(filename, content)
        speech_recognition_sdk = './asr_xf/bin/iat_sample '
        try:
            status, result = commands.getstatusoutput(speech_recognition_sdk + res_filename)
            if not status == 0:
                response_data = {"response_data": "sdk run error", "errcode": 3, "response_type":"sdk"}
                response_data.update(resp_data)
                logs = Log("ResponseDataSDK")
                logs.info(response_data)
                resultQueue.put(response_data)
            elif 'error' in result:
                errcont, errcode = result.split(':')
                errcont, _ = errcont.split(',')
                if eval(errcode) >= 1:
                    time.sleep(2)
                    response_data = {'response_data':'iflytec_Internal error', 'errcode':5, "response_type":"sdk"}
                    logs = Log("ResponseDataSDK")
                    logs.info(response_data)
                    resultQueue.put(response_data)
                    # pass
            elif result == '' or result is None:
                time.sleep(2) # SDK is not Support multi-processing, it may return an empty string, sleep 2 seconds to waiting for API.

            else:
                response_data = {"response_data": result.encode("utf-8"), "errcode": 0, "response_type":"sdk"}
                response_data.update(resp_data)
                response_data = json.dumps(response_data, ensure_ascii=False, encoding="utf-8")
                logs = Log("ResponseDataSDK")
                logs.info(response_data)
                resultQueue.put(response_data)
        except Exception as e:
            print(e)
            response_data = {"response_data": e.args, "errcode": 2}
            response_data.update(resp_data)
            response_data = json.dumps(response_data)
            logs = Log("ResponseDataSDK")
            logs.exception(response_data)
            resultQueue.put(response_data)
        finally:
            s.release()
            t2 = datetime.now()
            sys.stdout.write("request_ifly_sdk end:" + str(t2) + "\n")
            sys.stdout.write("request_ifly_sdk time used:" + str((t2 - t1).total_seconds()) + "\n")

    def request_ifly_api(self, filename, content, s, resultQueue):
        t1 = datetime.now()
        sys.stdout.write("request_ifly_api starting:" + str(t1) + "\n")
        resp_data = self.__requestsave__(filename=filename, logger_name="RequestDataAPI")
        errcode = resp_data.pop("errcode")
        if errcode:
            return resp_data
        try:
            audio_content = speech_recongnition_iflytec(content)
            result = audio_content.encode("utf-8")
            response_data = {"response_data": result, "errcode": 0, "response_type":"API"}
            response_data.update(resp_data)
            response_data = json.dumps(response_data, ensure_ascii=False, encoding="utf-8")
            logs = Log("ResponseDataAPI")
            logs.info(response_data)
            resultQueue.put(response_data)
        except Exception as e:
            response_data = {"response_data": e.args, "errcode": 4}
            response_data.update(resp_data)
            response_data = json.dumps(response_data)
            logs = Log("ResponseDataAPI")
            logs.exception(response_data)
            resultQueue.put(response_data)
        finally:
            s.release()
            self.__resultsave__(filename, content)
            t2 = datetime.now()
            sys.stdout.write("request_ifly_api end:" + str(t2) + "\n")
            sys.stdout.write("request_ifly_api time used:" + str((t2 - t1).total_seconds()) + "\n")

    def hySpeechRecognition(self, filename, content):
        resultQueue = Queue.Queue()
        s = Semaphore(0)
        t_api = thread(target=self.request_ifly_api, args=(filename, content, s, resultQueue, ))
        t_sdk = thread(target=self.request_ifly_sdk, args=(filename, content, s, resultQueue, ))
        t_api.setDaemon(True)
        t_sdk.setDaemon(True)
        t_sdk.start()
        t_api.start()
        s.acquire()

        result = resultQueue.get()
        try:
            return result
        except Exception as e:
            print(e)
            s.acquire()
            result = resultQueue.get()
            return result

try:
    handler = AudioDataHandler()
    processor = HYSpeechRecognition.Processor(handler)

    transport = TSocket.TServerSocket(host="0.0.0.0", port=8989)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    # server = TServer.TForkingServer(processor, transport, tfactory, pfactory)
    sys.stdout.write('Starting server at 192.168.2.233:8989\n')
    server.serve()

except BaseException as e:
    sys.stdout.write("This is BaseException content: {}".format(e))

sys.stdout.write("Done!\n")

q = Queue.Queue()
q.put(2)

