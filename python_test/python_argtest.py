# -*- coding: utf-8 -*-
# !python.exe


import sys
import getopt
import argparse


# paras = sys.argv[1:]
# paras_ = sys.argv #可以看成是一个列表，第一个元素是程序本身，随后依次是外部给予的参数。
# print(paras)
# print(paras_)

# oprts, args = getopt.getopt(sys.argv[0:], 'h:o,i:',['input=','output=','help',''])
# print(oprts, '--', args)
# for option, value in oprts:
#     print(option)


# parser = argparse.ArgumentParser(description = "your script description")  #第一步，description 可以为空
# parser.add_argument("--mode", "-m",action = "store_true",help = "mode true") #添加 --mode 标签，别名 -m ,action 表示一出现即代表 True，help 为参数描述
# args = parser.parse_args()    #将变量以标签-值的字典形式存入args字典

# print(args.mode)


def usage():
    print(' -h help \n -i ip address \n -p port number \n')

if __name__ == '__main__':
    try:
        options, args = getopt.getopt(sys.argv[1:], '-hi:p:a:', ['help', 'ip=', 'port=', 'pass='])
        print(options)
        print(args)

        for name, value in options:
            print(name, value)
            if name in ('-h', '--help'):
                usage()
            elif name in ('-i', '--ip'):
                print(value)

    except getopt.GetoptError:
        usage()
