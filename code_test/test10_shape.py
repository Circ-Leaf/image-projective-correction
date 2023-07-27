import cv2
import numpy as np

# 画像の読み込み
img = cv2.imread('picture_0_cut/2_75_0.jpg')

# グレースケール変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# HSV変換
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

# 色範囲によるマスク生成
img_mask = cv2.inRange(img_hsv, np.array([30,0,0]), np.array([100,32,255]))

# 輪郭抽出
contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 小さい輪郭は誤検出として削除
contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

# 輪郭の描画
cv2.drawContours(img, contours, -1, color=(0, 255, 0), thickness=2)

# イメージの表示
cv2.imshow('img_hsv',img_hsv)
cv2.imshow('img_mask',img_mask)
cv2.imshow('image', img)
cv2.waitKey(0)


