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
    t = threading.Thread(target=countdown, args=(10,), daemon=False)
    t.start()

    if t.is_alive():
        print('Still running')
    else:
        print('Completed')


from math import l