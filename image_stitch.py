'''
Author: Wang Taorui
Date: 2023-10-20 14:50:22
LastEditTime: 2023-10-20 16:08:59
LastEditors: Wang Taorui
Description: 
FilePath: /assignment2/image_stitch.py
'''
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def harris_corner_detector(image, threshold=0.01):
    # 计算图像的梯度
    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = dx.T

    Ix = convolve2d(image, dx, mode='same')
    Iy = convolve2d(image, dy, mode='same')

    # 计算 Harris 角点响应函数 R
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    kernel = np.ones((3, 3))
    Sxx = convolve2d(Ix2, kernel, mode='same')
    Syy = convolve2d(Iy2, kernel, mode='same')
    Sxy = convolve2d(Ixy, kernel, mode='same')

    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    R = det_M - 0.04 * (trace_M ** 2)

    # 根据阈值找到角点
    corners = (R > threshold * R.max()).nonzero()

    return corners

# 读取图像
image = plt.imread('image_pairs/image pairs_01_01.jpg')
image = np.mean(image, axis=2)  # 转换为灰度图

# 使用 Harris 角点检测
corners = harris_corner_detector(image)

# 在图像上标记角点
image_with_corners = image.copy()
image_with_corners[corners] = 255  # 将角点设置为白色

# 显示图像和标记的角点
plt.imshow(image_with_corners, cmap='gray')
plt.show()
