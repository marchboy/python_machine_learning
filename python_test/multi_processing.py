# -*- coding: utf-8 -*-

# threading

import threading
import time

def worker(num):
    time.sleep(2)
    print("The num is %d" % num)
    return 0

def multi_threading():
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,), name='t.%d' % i)
        t.start()

def countdown(n):
    while n>0:
        print("T-minus", n)
        n -= 1
        time.sleep(1)

# http://python3-cookbook-personal.readthedocs.io/zh_CN/latest/c12/p01_start_stop_thread.html?highlight=daemon
# http://funhacks.net/explore-python/Process-Thread-Coroutine/coroutine.html
# https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001407503089986d175822da68d4d6685fbe849a0e0ca35000
class CountdownTask():
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False

    def run(self, n):
        while self._running and n > 0:
            print("T-minus", n)
            n -= 1
            time.sleep(2)

if __name__ == '__main__':
    # multi_threading()
    # t = threading.Thread(target=countdown, args=(10,), daemon=True)
    # t.start()
    # t.join()

    # if t.is_alive():
    #     print('Still running')
    # else:
    #     print('Completed')

    c = CountdownTask()
    t = threading.Thread(target=c.run, args=(10,))
    t.start()
    # c.terminate()
    t.join()

