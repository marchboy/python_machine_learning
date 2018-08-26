#-*-coding:utf-8 -*-
# FROM: https://yq.aliyun.com/articles/610135?utm_content=m_1000006277

import imageio
import random
import numpy as np
import matplotlib.pyplot as plt

pic = imageio.imread('E:\\dev_sm.jpg')

"""
print(plt.figure(figsize=(15,15)))
print(plt.imshow(pic))
plt.show()

print('type of the image:', type(pic))
print('shape of the image:{}'.format(pic.shape))

print('Image Height {}, Image Wight {}'.format(pic.shape[0], pic.shape[1]))
print('Dimension of Image {}'.format(pic.ndim))

print('Image size {}'.format(pic.size))
print('Maxmun RGB value in the image {}'.format(pic.max()))
print('Minium RGB value in the image {}'.format(pic.min()))


print(pic[100, 50])
# print(Image([109, 143, 46], dtype=uint8))


# 索引0表示红色通道
# 索引1表示绿色通道
# 索引2表示蓝色通道
print('Value of only R channel {}'.format(pic[100, 50, 0]))
print('Value of only G channel {}'.format(pic[100, 50, 1]))
print('Value of only B channel {}'.format(pic[100, 50, 2]))
# ----------------------R Channel-------------------------
print(plt.title('R Channel'))
print(plt.ylabel('Height {}'.format(pic.shape[0])))
print(plt.xlabel('Width {}'.format(pic.shape[1])))

plt.imshow(pic[:,:,0])
plt.show()
# ----------------------G Channel--------------------------
print(plt.title('G Channel'))
print(plt.ylabel('Height {}'.format(pic.shape[0])))
print(plt.xlabel('Width {}'.format(pic.shape[1])))

plt.imshow(pic[:,:,1])
plt.show()
# ----------------------B Channel--------------------------
print(plt.title('B Channel'))
print(plt.ylabel('Height {}'.format(pic.shape[0])))
print(plt.xlabel('Width {}'.format(pic.shape[1])))

plt.imshow(pic[:,:,2])
plt.show()

# 测试只在一张图像上进行综合处理，方便我们同时查看每个通道的值对图像的影响
pic = imageio.imread('E:\\dev_sm.jpg')
pic[150:350, :, 0] = 255
plt.figure(figsize=(10,10))
print(plt.imshow(pic))
plt.show()

pic[350:550, :, 1] = 255
plt.figure(figsize=(10,10))
print(plt.imshow(pic))
plt.show()

pic[550:750, :, 2] = 255
plt.figure(figsize=(10,10))

pic[100:300, 400:800, [0,1,2]] = 200
print(plt.imshow(pic))
plt.show()



fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
print(fig, '\n', ax)
for c, ax in zip(range(3), ax):
    print(c, ax)
    split_img = np.zeros(pic.shape, dtype='uint8')
    split_img[:,:,c] = pic[:,:,c]
    ax.imshow(split_img)
plt.show()


# Y = 0.299*R + 0.587*G + 0.114*B
pic=imageio.imread('E:\\dev_sm.jpg') 
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic)
plt.figure(figsize=(10,10))
plt.imshow(gray,cmap=plt.get_cmap(name='gray'))
plt.xlabel("Height")
plt.ylabel("Width")
plt.title("Gray Photo")
plt.show()

gray = lambda rgb:np.dot(rgb[...,:3], [0.21, 0.72, 0.07])
gray = gray(pic)
plt.figure(figsize=(10, 10))
plt.imshow(gray, cmap=plt.get_cmap(name='gray'))
plt.show()

print('Type of the image: ', type(gray))
print()
print('Shape of the image: {}'.format(gray.shape))
print('Image Height {}'.format(gray.shape[0]))
print('Image Width {}'.format(gray.shape[1]))
print('Dimension of Image {}'.format(gray.ndim))
print()
print('Image size {}'.format(gray.size))
print('Maximun RGB value in this image {}'.format(gray.max()))
print('Minumum RGB value in this image {}'.format(gray.min()))
print('Random indexes [X, Y]: {}'.format(gray[100, 50]))
"""


pic = imageio.imread('E:\\dev_sm.jpg')
plt.figure(figsize=(10, 10))
# plt.imshow(pic)
# plt.show()

low_pixel = pic < 20
if low_pixel.any() == True:
    print(low_pixel.shape)

print(pic.shape)
print(low_pixel.shape)

pic[low_pixel] = random.randint(25, 225)
plt.figure(figsize=(10, 10))
# plt.imshow(pic)
# plt.show()

total_row, total_col, layers = pic.shape
x, y = np.ogrid[:total_row, :total_col]
cen_x, cen_y = total_row/2, total_col/2

distance_from_the_center = np.sqrt((x-cen_x)**2 + (y-cen_y)**2)
radius = (total_row/2)
circular_pic = distance_from_the_center > radius  #逻辑操作符
pic[circular_pic] = 0
plt.figure(figsize=(10, 10))
plt.imshow(pic)
plt.show()


print(f'Shape of the image {pic.shape}')
print(f'Height {pic.shape[0]} pixels')
print(f'wodth {pic.shape[1]} pixels')

pic = imageio.imread('E:\\dev_sm.jpg')
red_mask = pic[:,:,0] < 180
pic[red_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(pic)

green_mask = pic[:,:,1] < 180
pic[green_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(pic)

blue_mask = pic[:,:,2] < 180
pic[blue_mask] = 0
plt.figure(figsize=(15, 15))
plt.imshow(pic)

final_mask = np.logical_and(red_mask, green_mask, blue_mask)
pic[final_mask] = 40
plt.figure(figsize=(15, 15))
plt.imshow(pic)
plt.show()


# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
# print(fig, '\n', ax)
# for c, ax in zip(range(3), ax):
#     split_img = np.zeros(pic.shape, dtype='uint8')
#     split_img[:,:,c] = pic[:,:,c]
#     ax.imshow(split_img)
# plt.show()


# https://yq.aliyun.com/articles/469057?spm=a2c4e.11153940.blogcont610135.17.16c77861MVQb5g