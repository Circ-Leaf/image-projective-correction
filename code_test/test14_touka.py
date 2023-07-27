import numpy as np
import cv2

img = cv2.imread("blue_masked_img.png" )
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
iter = 5
rect = (39,5,100,250)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img = img*mask2[:,:,np.newaxis]
cv2.imshow("img",img)
cv2.imwrite("grabbed.png", img)

# ---------------------------------

img_bgr = cv2.split(img)
mask2 = mask2*255

# cv2.imwrite("aaa.png",mask2*255)

img_alpha = cv2.merge(img_bgr+[mask2])

cv2.imshow("alpha",img_alpha)
cv2.waitKey(0)
cv2.imwrite("alpha.png",img_alpha)