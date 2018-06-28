#! /usr/bin/python
# -*- coding:utf-8 -*-

class Main(object):
    _dict = dict()

    def __new__(cls):
        if 'key' in Main._dict:
            print('Exists')
            return Main._dict['key']
        else:
            print('New')
            return object.__new__(cls)

    def __init__(self):
        print("Init")
        Main._dict['key'] = self


class Class_method(object):
    bar = 1
    #@classmehtod
    def class_foo(cls):
        print("Hello", self)
        print(cls.bar)
        print(class_method.bar)




if __name__ == "__main__":
    
    print Class_method.bar
    Class_method.classmehtod()
    #main = Main()
    #main_1 = Main()
    #main_2 = Main()
