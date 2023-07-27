import cv2
import numpy as np

def transSquare(img):
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    new_img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
        new_img[start:fin, :] = tmp
    return(new_img)

img = cv2.imread('picture/picture_0_cut/1_75_0.jpg', cv2.IMREAD_COLOR)
neimg = transSquare(img)

cv2.imshow('image', neimg)
cv2.waitKey(0)