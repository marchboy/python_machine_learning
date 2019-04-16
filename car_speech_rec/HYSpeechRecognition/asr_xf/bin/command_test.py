#/usr/bin/env python
# -*- coding:utf-8 -*-


import commands

retcode, ret = commands.getstatusoutput('./iat_sample ./wav/REC3.WAV')
print(retcode)
print(ret)

