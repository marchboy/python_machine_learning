# coding:utf-8

import sys
sys.path.append("./gen-py")
from textFilter import HYTextFilterService

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

try:
    transport = TSocket.TSocket('192.168.2.233', 9239)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = HYTextFilterService.Client(protocol)
    transport.open()

    print "client - input"
    text = [u"上个月习近平主席来广视察!",u"毛统计局uuuhdf大家开发商",u"针孔摄像机是什么"]

    for i in text:
        print(i)

        msg = client.hyTextFilter(i.encode("utf8"))
    print "server - output\n" + msg

    transport.close()

except Thrift.TException, ex:
    print "%s" % (ex.message)
