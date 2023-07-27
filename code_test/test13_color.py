#-*- coding:utf-8 -*-
import cv2
import numpy as np

# 青色の検出
def detect_blue_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 青色のHSVの値域1
    hsv_min = np.array([90, 64, 0])
    hsv_max = np.array([150,255,255])

    # 青色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img


# 入力画像の読み込み
img = cv2.imread("picture/blue.png")

# 色検出（赤、緑、青）
blue_mask, blue_masked_img = detect_blue_color(img)

# 結果を出力
cv2.imwrite("blue_mask.png", blue_mask)
cv2.imwrite("blue_masked_img.png", blue_masked_img)