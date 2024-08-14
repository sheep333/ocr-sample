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

    # 傾き補正
    corrected_img = deskew(cropped_img)

    # 前処理された画像を保存または返す
    return corrected_img

def deskew(image):
    # グレースケールに変換（傾き補正のために再度グレースケールを使用）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # エッジ検出
    edges = cv2.Canny(gray, 50, 200)

    # 輪郭を探す
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭の最小外接矩形を計算
    min_area_rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))

    # 傾き角度を取得
    angle = min_area_rect[-1]

    # 角度を調整
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 画像の中心を計算
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 回転行列を計算
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 画像を回転させて補正
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

# 画像のパス
image_path = 'aaa.png'

# 前処理した画像を取得
processed_image = preprocess_image(image_path)

# 処理結果を保存（確認用）
cv2.imwrite('processed_image.png', processed_image)
