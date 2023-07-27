#-*- coding:utf-8 -*-
import cv2
import numpy as np

# load image (grayscale)
# 入力画像をグレースケールで読み込み
imgWarpColored = cv2.imread(
    "picture/zoom_sample5.jpg")

imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)           # グレースケール化
imgBinarization = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)   # 二値化（適応的二値化処理）
imgBinarization = cv2.bitwise_not(imgBinarization)                      # 色反転
imgMedian = cv2.medianBlur(imgBinarization,5)                           # メディアンフィルタ(3,5...奇数で変更可)


# output
# 結果を出力
cv2.imwrite("result1.png", imgBinarization)
cv2.imwrite("result2.png", imgMedian)
