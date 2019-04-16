# -*- coding: utf-8 -*-

import os
from logging import getLogger, FileHandler, INFO
from cloghandler import ConcurrentRotatingFileHandler

def log_add():
    logfile = os.path.abspath("mylogfile.log")
    log = getLogger()
    rotate_handler = ConcurrentRotatingFileHandler(logfile, "a", 1024*1024, 5)
    log.addHandler(rotate_handler)
    log.setLevel(INFO)
    log.info("Here is a very exciting log message for you.")

if __name__ == "__main__":
    log_add()