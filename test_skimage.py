# -*- coding: UTF-8 -*-

'''
from PIL import Image
import matplotlib.pyplot as plt
# 打开一个jpg图像文件，注意是当前路径:
im = Image.open('tmp/cat.jpg')
# 获得图像尺寸:
w, h = im.size
print('Original image size: %sx%s' % (w, h))
im.thumbnail((w//2, h//2))
print('Resize image to: %sx%s' % (w//2, h//2))
im.show()
'''


from skimage import data, io
import matplotlib.pyplot as plt
img = data.chelsea()
# 查看图片，使用io模块中的imshow方法
# io.imshow(img)
# io.imsave('tmp/cat.jpg', img)
print(img.shape)
R = img[:, :, 0]
io.imshow(R)
io.imsave('tmp/cat.jpg', R)
plt.show()
