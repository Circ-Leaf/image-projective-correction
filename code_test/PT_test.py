import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'picture/school.png'                   # 画像のパス
i = cv2.imread(path, 1)               # 画像読み込み
print(i.shape)

"""
#座標確認用
show = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(show)
fig.tight_layout()
plt.show()
plt.close()
"""


# 変換前後の対応点を設定
x1,y1 = 323,1614
x2,y2 = 697,1739
x3,y3 = 266,2055
x4,y4 = 688,2161

p_original = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  
p_trans = np.float32([[0, 0,], [1000, 0], [0, 1000], [1000, 1000]]) 


# 変換マトリクスと射影変換
M = cv2.getPerspectiveTransform(p_original, p_trans)
i_trans = cv2.warpPerspective(i, M, (1000,1000))              #変換後の画像の大きさ

cv2.imwrite("result/PT_out.jpg", i_trans)

#ここからグラフ設定
fig = plt.figure()
ax1 = fig.add_subplot(111)

# 画像をプロット
show = cv2.cvtColor(i_trans, cv2.COLOR_BGR2RGB)
ax1.imshow(show)

#画像を保存
cv2.imwrite("result/PT_result.png", i_trans)


fig.tight_layout()
plt.show()
plt.close()
