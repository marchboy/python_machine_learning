# -*- coding: utf-8 -*-
# https://yq.aliyun.com/articles/625294?spm=a2c4e.11163080.searchblog.57.5e032ec1X9gre2
# 线程锁用于必须以固定顺序执行的多个线程的调度,
# 线程锁的思想是先锁定后序线程，然后让前序线程完成任务再解除对后序线程的锁定

from multiprocessing import Process,Event 
from time import sleep

def wait_event(file):
    print("准备操作临界资源")
    e.wait()  # 等待主进程执行结束后set
    print("开始操作临界资源",e.is_set())
    fw = open('E:\\0.jpg','wb')
    with open(file,'rb') as f:  # 复制图片
        fw.write(f.read())

def wait_event_timeout(file):
    print("也想操作临界资源")
    e.wait(2)  # 等待主进程执行set并进行2秒超时检测
    if e.is_set():
        print("也开始操作临界资源")
        fw = open('E:\\1.png','wb')
        with open(file,'rb') as f:  # 复制图片
            fw.write(f.read())
    else:
        print("等不了了，不等了")

# 创建事件
e = Event()

file = 'E:\\dev_sm.jpg'

# 创建两个进程分别复制两个图片
p1 = Process(target = wait_event, args = (file,))
p2 = Process(target = wait_event_timeout, args = (file,))

p1.start()
p2.start()

# 主进程先复制图片 让子进程进入wait状态
print("主进程在操作临界资源")
sleep(3)
fw = open(file,'wb')
with open(file,'rb') as f:
    fw.write(f.read())
fw.close()
e.set()  # 子进程set
print("主进程操作完毕")


p1.join()
p2.join()

