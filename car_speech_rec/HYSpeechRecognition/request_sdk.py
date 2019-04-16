#!/usr/bin/env python

import os


current_path = os.path.dirname(__file__)
parent_path = os.path.abspath(os.path.dirname(current_path))
parent_path = os.path.abspath(os.path.join(current_path, ".."))




if __name__ == "__main__":
    print(__file__)
    print(os.path.dirname(__file__))
    print(current_path, parent_path)