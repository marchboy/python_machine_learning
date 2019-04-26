#-*- coding: utf-8 -*-

import tensorflow as tf


class Bitmap(object):

    def __init__(self, max):
        self.size = self.calcElemIndex(max, True)
        self.array = [0 for i in range(self.size)]

    def calcElemIndex(self, num, up=False):
        if up:
            return int((num + 31 - 1) / 31) #向上取整
        return int(num / 31)

    def calcBitIndex(self, num):
        return num % 31

    def set(self, num):
        elemIndex = self.calcElemIndex(num)
        byteIndex = self.calcBitIndex(num)
        elem = self.array[elemIndex]
        print('--')
        print(elemIndex, byteIndex, elem)
        self.array[elemIndex] = elem | (1 << byteIndex)
        print(self.array)


if __name__ == '__main__':
    bitmap = Bitmap(90)
    print(bitmap.array)
    bitmap.set(0)
    print(bitmap.array)
    print('需要 %d 个元素。' % bitmap.size)
    print('50应存储在第{}个数组元素上'.format(bitmap.calcElemIndex(47)))
    print('50应存储在第{}个数组元素的第{}个位置上'.format(bitmap.calcElemIndex(47), bitmap.calcBitIndex(47)))

