# coding:utf-8

import sys
import json 
import time
import logging
from datetime import datetime
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# reload(sys)
# sys.setdefaultencoding("utf-8")
sys.path.append("./WordsTransfer/gen-py")
from textFilter import HYTextFilterService
from ac_filter import AC 

class HYTextFilterServiceHandler:
    def __init__(self):
        ac = AC()
        kwords = ac.parse("./keywords_bak.txt")
        ac.init(kwords)

        self.ac_filter = ac.filter

    def hyTextFilter(self, msg):
        msg = msg.decode("utf8")
        try:
            response = self.ac_filter(msg.decode("utf8"))
            ret = {"Request":msg, "Response":response, "ErrorCode":0}
            logger.info(ret)
        except BaseException as e:
            ret = {"Request":msg, "Response":e, "Errcode":1}
            logger.error(ret)

        return str(response)


if __name__ == "__main__":
    RPCHOST = "0.0.0.0"
    RPCPORT = 9239

    try:
        logfile_path = "./"
        file_handler = logging.FileHandler(logfile_path + "text_filter_log.log")
        formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S"
                )
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    except BaseException as e:
        sys.stdout.write("This is log error: {}\n".format(e))

    try:
        handler = HYTextFilterServiceHandler()
        processor = HYTextFilterService.Processor(handler)
        transport = TSocket.TServerSocket(RPCHOST, RPCPORT)
        tfactory = TTransport.TBufferedTransportFactory()
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()
        server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
        #logging.basicConfig(level=logging.INFO)
        
        print('Starting thrift server at {}:{}...\n'.format(RPCHOST, RPCPORT))
        server.serve()

    except Exception as e:
        print("This is BaseException content: {}".format(e))
    
    sys.stdout.write("done!")
