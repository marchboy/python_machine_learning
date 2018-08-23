#-*-coding:utf-8 -*-
import imageio
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

"""

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
print(fig, '\n', ax)
for c, ax in zip(range(3), ax):
    print(c, ax)
    split_img = np.zeros(pic.shape, dtype='uint8')
    split_img[:,:,c] = pic[:,:,c]
    ax.imshow(split_img)
plt.show()


# Y = 0.299*R + 0.587*G + 0.114*B
pic=imageio.imread('F:/demo_2.jpg') 
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic) 
plt.figure(figsize=(10,10))
plt.imshow(gray,cmap=plt.get_cmap(name='gray'))
plt.show()