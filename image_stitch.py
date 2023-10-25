'''
Author: Wang Taorui
Date: 2023-10-20 14:50:22
LastEditTime: 2023-10-25 14:38:10
LastEditors: Wang Taorui
Description: 
FilePath: /assignment2/image_stitch.py
'''
import cv2
import numpy as np

# Load two images to be stitched
image1 = cv2.imread('image_pairs/image pairs_04_01.jpg')
image2 = cv2.imread('image_pairs/image pairs_04_02.jpg')

# Detect keypoints and compute descriptors
detector = cv2.SIFT_create()
keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

# Match keypoints using a descriptor matcher (e.g., FLANN or BFMatcher)
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Get corresponding points in both images
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find the Homography matrix using RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the second image to align with the first
result = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image2.shape[0]))

# Copy the first image to the result
result[0:image1.shape[0], 0:image1.shape[1]] = image1

# Save or display the stitched image
cv2.imwrite('panorama.jpg', result)
cv2.imshow('Stitched Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
