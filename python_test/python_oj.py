# -*- coding = utf-8 -*-

#list_num = raw_input().split(' ')
# def devi():
#     a = int(list_num[0])
#     b = int(list_num[1])
#     if a % b == 0:
#         print("YES")
#     else:
#         print("NO")
#devi()

# num = input()
# lst = []
# def devision():
#     for i in xrange(num):
#         if num % (i + 1) == 0:
#             lst.append(i+1)
#     if lst == [1, num]:
#         print("Y")
#     else:
#         print("N")
# devision()

# n = input()
# def fibonacci(n):
#     if n <= 1:
#         return n
#     fibo = fibonacci(n - 1) + fibonacci(n - 2)
#     return fibo
# fibo = fibonacci(n)
# print(fibo)

# import numpy as np
# list = input().split(' ')
# num_
# matrix = np.zeros((int(num_list[0], int(num_list[1]))))

# -*- coding: utf-8 -*- 
import urllib.request
import urllib.parse
import json

content=input("Please input words what you want translate:")
url="http://fanyi.baidu.com/v2transapi"

form_data={
    "query":content,
    "form":"zh",
    "to":"en"
    }

head={}
head["User-Agent"]="Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Mobile Safari/537.36"

#下面将form_data转译成机器码
form_data=urllib.parse.urlencode(form_data).encode("utf-8")

#生成一个request object（对象）
req=urllib.request.Request(url,form_data,head)
#req.add_header(head)

response=urllib.request.urlopen(req)
html=response.read().decode("utf-8")
print(html)
target=json.loads(html)

print(target)
