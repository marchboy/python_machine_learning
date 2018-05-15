# -*- coding: utf-8 -*-


"""
print(pow(2,10))

#month.py
month = "JanFebMarAprMayJunJulAugSepOctNovDec"
n = input("请输入月份（1-12）：")
pos = (int(n) - 1) * 3
monthAbbrev = month[pos:pos+3]
print("月份的简写是：" + monthAbbrev + ".")



##--maxn.py

#寻找一组数据当中最大者
def main():
    n = eval(input("How many numbers are there?"))
    max = eval(input("Enter a number >> "))
    for i in range(n-1):
        x = eval(input("Enter a number >> "))
        if x > max:
            max = x
        print("The largest value is ", max)
main()


import pandas as pd
import os
os.getcwd()
path = 'E:\\MyStudy\\Python\\pydata_book_master\\ch02\\names'
os.chdir(path)

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    filename = 'yob%d.txt' % year
    frame = pd.read_csv(filename, names=columns)
    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces, ignore_index= True)
print(len(names))

"""

# A, B, C = input().split(' ')
# print("A + B + C = ", int(A) + int(B) + int(C))



# http://bbs.fishc.com/thread-79821-1-1.html
# https://blog.csdn.net/ly_ysys629/article/details/55224284
import pandas as pd
import random

df = pd.DataFrame(
    {"total_bill":[16.99, 10.34, 23.68, 23.68, 24.59],
    "tip":[1.01, 1.66, 3.50, 3.31, 3.61],
    "sex":['Female', 'Male', 'Male', 'Male', 'Female']})

# print(df.dtypes, '\n')
# print(df.index, '\n')
# print(df.columns, '\n')
# print(df.values, '\n')
# print(df)

print('-' * 50)
rnd_1 = [random.randrange(1,20) for x in range(1000)]
rnd_2 = [random.randrange(1,20) for x in range(1000)]
rnd_3 = [random.randrange(1,20) for x in range(1000)]
fecha = pd.date_range('2012-4-10', '2015-1-4')

data = pd.DataFrame({'fecha':fecha, 'rnd_1':rnd_1, 'rnd_2':rnd_2, 'rnd_3':rnd_3})
print(data)
print(data.describe())

print(data[0:10][["fecha", "rnd_1"]])
print(data[0:10])
print('-' * 50)
print(data.loc[1:5])  # 推荐此方法进行行列选择，而不是[],注意与iloc的区别

