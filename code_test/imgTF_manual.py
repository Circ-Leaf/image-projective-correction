import cv2
import numpy as np
import utils
from googleAPI import GoogleAPI

# 初期設定
dirImage  = 'picture/picture_0_cut/'
datImage  = '1_75_0'
estImage  = '.jpg'
pathImage = dirImage + datImage + estImage
heightImg = 640
widthImg  = 640

utils.initializeTrackbars()
count=0

# GoogleDocの初期設定
googleAPI = GoogleAPI(drive_token_path='./auth_keys/drive/token.json', docs_token_path='./auth_keys/docs/token.json',
                drive_credentials_path='./auth_keys/drive/credentials.json', docs_credentials_path='./auth_keys/docs/credentials.json')


while True:

    # 下準備
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))             # 画像のリサイズ
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)   # 空配列（出力がなかったとき用）
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # グレースケール化
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)           # ガウシアンフィルタ
    thres = utils.valTrackbars()                             # Thresholdバーの取得
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])      # Cannyエッジ検出(閾値min,maxはバーで指定)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # 膨張処理（線分強調）
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # 収縮処理（膨張によるノイズ削減のため）


    # 輪郭検出
    imgContours = img.copy()   
    imgBigContour = img.copy() 
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 全輪郭の検出
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)                                     # 輪郭の描画


    # 最大輪郭の探索
    biggest, maxArea = utils.biggestContour(contours)                                   # 最大輪郭（四角形）の頂点を取得
    if biggest.size != 0:
        biggest=utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)                   # 最大輪郭の描画
        imgBigContour = utils.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)                                                      # 部屋識別子の頂点座標（変換前）
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # 部屋識別子の頂点座標（変換後）
        matrix = cv2.getPerspectiveTransform(pts1, pts2)                                # 射影変換の変換行列を生成
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))        # 射影変換

        # 各面の20pixel排除（ズーム）
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # 射影変換後画像のフィルタ処理
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)           # グレースケール化
        imgBinarization = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)   # 二値化（適応的二値化処理）
        imgBinarization = cv2.bitwise_not(imgBinarization)                      # 色反転
        imgMedian = cv2.medianBlur(imgBinarization,3)                           # メディアンフィルタ(奇数で変更可)

        # 出力結果にて表示する画像配列
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgBinarization, imgMedian])

    else:
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])


    # 出力結果の各ラベル
    lables = [["Original","Gray","Binarization","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Binarization","Warp Median"]]

    stackedImage = utils.stackImages(imageArray,0.25,lables)     # 第二引数は表示サイズ
    cv2.imshow("Result",stackedImage)


    # 変換後画像の保存（sキー）
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("result/save/O_color"+str(count)+".jpg",imgWarpColored)
        cv2.imwrite("result/save/O_Binari"+str(count)+".jpg",imgBinarization)
        cv2.imwrite("result/save/O_Median"+str(count)+".jpg",imgMedian)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

    # OCR実行
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print("proceccing c")
        cv2.imwrite("resultTF/" + datImage + "_OCR" + ".jpg",imgWarpColored)
        print(googleAPI.OCR("resultTF/" + datImage + "_OCR" + ".jpg"))

    # 実行終了（qキー）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit()