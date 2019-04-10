#!usr/bin/env python
# -*- coding:utf -8-*-
# Author: jinjun.gui
# Date: 2019-03-07


import cv2 as cv 
import numpy as np

# recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
# 从摄像头获取到图像
ret, img = cap.read()
# 展示图片,第一个参数表示窗口名称,第二个是要展示的图片.
cv.imshow('windowname', img)
# 停留展示界面,6000毫秒,0表示一直停留着.
cv.waitKey(3000)
# 释放摄像头资源
cap.release()
# 将图片先转换为灰度图片
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray)
faces = detector.detectMultiScale(gray, 1.3, 5)
print(faces)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)




# detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv.VideoCapture(0)

# while True:
#     ret, img = cap.read()
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     faces = detector.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     cv.imshow('frame', img)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv.destroyAllWindows()
