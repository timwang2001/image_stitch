'''
Author: Wang Taorui
Date: 2023-10-23 19:39:36
LastEditTime: 2023-10-25 14:52:40
LastEditors: Wang Taorui
Description: 
FilePath: /assignment2/get_homography.py
'''
from cv2implment import get_pointset
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
p=1
img1 = cv.imread('image_pairs/image pairs_01_01.jpg')
img2 = cv.imread('image_pairs/image pairs_01_02.jpg')

pointset,keypoint1,keypoint2 = get_pointset(img1, img2)
# RANSAC算法计算单应性矩阵

src_pts = np.float32([keypoint1[m.queryIdx].pt for m in pointset]).reshape(-1, 1, 2)
tge_pts = np.float32([keypoint2[m.trainIdx].pt for m in pointset]).reshape(-1, 1, 2)
# threshold = 500
# # print(src_pts)
# A = []
# for i in range(len(src_pts)):
#     x, y = src_pts[i][0]
#     x_prime, y_prime = tge_pts[i][0]
#         # Calculate the Euclidean distance between feature points
#     distance = np.sqrt((x - x_prime) ** 2 + (y - y_prime) ** 2)
#     # print(distance)
#     # Check if the distance is less than the threshold
#     if distance < threshold:
#         A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
#         A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

# A = np.array(A)

# Solve the linear system using SVD
# U, S, Vt = np.linalg.svd(A)
# H = Vt[-1].reshape(3, 3)
# H /= H[2,2]
M, mask = cv.findHomography(src_pts, tge_pts, cv.RANSAC, 10)
# print(H)
## get homography matrix done

## do the transformation
# warpimg0 = cv.warpPerspective(img2, np.linalg.inv(H), (img1.shape[1] + img2.shape[1], img2.shape[0]))
# warpimg1 = cv.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))


# Determine the dimensions of the output image
output_width = img1.shape[1] + img2.shape[1]
output_height = max(img1.shape[0], img2.shape[0])

# Create an output image with dimensions to accommodate both images
output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

# Calculate the inverse of the homography matrix
H_inv = np.linalg.inv(M)

# Iterate through the pixels of the output image
for y in range(img2.shape[0]):
    for x in range(img2.shape[1]):
        # Apply the inverse homography to find the corresponding pixel in the output image
        point = np.dot(H_inv, np.array([x, y, 1]))
        output_x = int(point[0] / point[2])
        output_y = int(point[1] / point[2])

        # Check if the output coordinates are within the boundaries of the output image
        if 0 <= output_x < output_width and 0 <= output_y < output_height:
            # Copy the pixel value from img2 to the output image
            output_image[output_y, output_x] = img2[y, x]

warpimg = cv.warpPerspective(img2, np.linalg.inv(M), (output_width,output_height))

# warpimg = output_image.copy()

rows, cols = img1.shape[:2]
print(rows, cols)
print(warpimg.shape)
left = 0
right = cols


# 找到img1和warpimg重叠的最左边界
for col in range(0, cols):
    if img1[:, col].any() and warpimg[:, col].any():
        left = col
    break
# 找到img1和warpimg重叠的最右边界
for col in range(cols - 1, 0, -1):
    if img1[:, col].any() and warpimg[:, col].any():
        right = col
    break
# 图像融合
res = np.zeros([rows, cols, 3], np.uint8)

for row in range(0, rows):
    for col in range(0, cols):
        if not img1[row, col].any():
            res[row, col] = warpimg[row, col]
        elif not warpimg[row, col].any():
            res[row, col] = img1[row, col]
        else:
            # 重叠部分加权平均
            srcimgLen = float(abs(col - left))
            testimgLen = float(abs(col - right))
            alpha = srcimgLen / (srcimgLen + testimgLen)
            res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpimg[row, col] * alpha, 0, 255)


# 裁剪拼接后的图像以删除黑色边界
def crop_black_borders(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(contours[0])
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

warpimg[0:img1.shape[0], 0:img1.shape[1]] = res
# img3 = cv.cvtColor(direct, cv.COLOR_BGR2RGB)
# plt.imshow(img3), plt.show()
warpimg = crop_black_borders(warpimg)
img4 = cv.cvtColor(warpimg, cv.COLOR_BGR2RGB)
# plt.imshow(img4), plt.show()
cv.imwrite('image_pairs/'+'image pairs_0%d.jpg'%p,warpimg)

# cv.waitKey(0)