import cv2

img = cv2.imread('picture_0_cut/5_15_0.jpg')
height = img.shape[0]
weight = img.shape[1]

print(height)
print(weight)