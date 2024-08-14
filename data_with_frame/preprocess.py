import cv2
import numpy as np

def preprocess_image(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーを使用してノイズを低減
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # エッジ検出
    edged = cv2.Canny(blurred, 50, 200)

    # 輪郭を探す
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭の中で最大のものを探す
    largest_contour = max(contours, key=cv2.contourArea)

    # 輪郭の外接矩形を取得
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 画像を外接矩形にクロップ
    cropped_img = img[y:y+h, x:x+w]

    # 傾き補正（必要な場合）
    # ここに傾き補正コードを追加できます

    # 前処理された画像を保存または返す
    return cropped_img

# 画像のパス
image_path = 'aaa.png'

# 前処理した画像を取得
processed_image = preprocess_image(image_path)

# 処理結果を保存（確認用）
cv2.imwrite('processed_image.png', processed_image)
