import cv2
import numpy as np

def detect_area_white(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 指定色のHSVの値域1
    hsv_min = np.array([30,0,0])
    hsv_max = np.array([90,32,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 指定色のHSVの値域2(未使用)
    hsv_min = np.array([30,0,0])
    hsv_max = np.array([90,0,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 指定色領域のマスク（255：赤色、0：赤色以外）    
    mask = mask1 + mask2

    # マスキング処理(ANDで画像合成)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img


# 入力画像の読み込み
img = cv2.imread("resultTF/2_75_0_OCR.jpg")

# 色検出
white_masked_img = detect_area_white(img)

cv2.imwrite("area_white_img.png", white_masked_img)