#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import sys
import commands
import subprocess


def main(args):
    rc, out = commands.getstatusoutput("./iat_sample " + args)
    return out

    # 返回状态码，结果
    #rc, out = commands.getstatusoutput("./iat_sample " + args)
    #res = json.loads()     
    
    # 返回命令执行状态码，而将命令执行结果输出到屏幕
    #res = os.system("./iat_sample ./wav/REC3.WAV")
    #return res

    
    # 可以获取命令执行结果，但是无法获取命令执行状态码
    #s = os.popen("./iat_sample " + args)
    #data = s.readlines()
    #return data
    
    #proc = subprocess.Popen("./iat_sample " + args , stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    #return proc.stdout.read()

if __name__ == "__main__": 
    args = sys.argv[1]

    c = main(args)
    print(c)

