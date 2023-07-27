import cv2
import numpy as np


def make_sharp_kernel(k: int):
  return np.array([
    [-k / 9, -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, k / 9],
    [-k / 9, -k / 9, -k / 9]
  ], np.float32)

img = cv2.imread("picture/noise.png")
kernel = make_sharp_kernel(4)
img = cv2.filter2D(img, -1, kernel).astype("uint8")

# 二値化
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# メディアン
dst = cv2.medianBlur(th, ksize=13)


cv2.imwrite("result/anything/Sharpen_result.png", img)
cv2.imwrite("result/anything/Nichi_result.png", th)
cv2.imwrite("result/anything/median_result.png", dst)
