# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:18:18 2016
@author: John
"""
import turtle


def drawsnake(rad, angle, len, neckrad):
    for i in range(len):
        turtle.circle(rad, angle)   # 沿圆形轨迹爬行，rad表示圆形轨迹半径的位置，angle表示沿圆形爬行的弧度值
        turtle.circle(-rad, angle)  # rad为负值，表示半径在小乌龟运行的右侧
    turtle.circle(rad, angle/2)
    turtle.fd(rad)  # turtle.fd() 表示小乌龟向前沿直线爬行，他有一个参数表示爬行距离
    turtle.circle(neckrad+1, 180)
    turtle.fd(rad*2/3)

def main():
    turtle.setup(1300, 800, 0, 0)
    python_size = 30   # 运行轨迹的宽度
    turtle.pensize(python_size)
    turtle.pencolor("red")  # 运行轨迹的颜色
    turtle.seth(0)   # 启动时运行的方向
    drawsnake(30, 60, 5, python_size/2)
main()



