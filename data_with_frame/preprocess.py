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

def extract_and_save_data_rows(image, min_threshold=0.01, max_threshold=0.9):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二値化
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    h, w = binary_image.shape

    # 各行をスキャンしてデータが含まれているか判定
    saved_image_count = 0
    start = None

    for i in range(h):
        black_pixel_ratio = np.sum(binary_image[i, :] == 255) / w
        
        if min_threshold < black_pixel_ratio < max_threshold:  # 1割以上9割未満の黒いピクセルがある行をデータとみなす
            if start is None:
                start = i  # データがある行の開始位置を記録
        else:
            if start is not None:
                # データがある行を画像として保存
                row_img = image[start:i, :]
                cv2.imwrite(f'data_row_{saved_image_count}.png', row_img)
                saved_image_count += 1
                start = None  # リセット
    
    # 最後のデータ行を保存
    if start is not None:
        row_img = image[start:h, :]
        cv2.imwrite(f'data_row_{saved_image_count}.png', row_img)

    print(f'{saved_image_count} rows with data saved.')


# 画像のパス
image_path = 'aaa.png'

# 前処理した画像を取得
processed_image = preprocess_image(image_path)

# データが含まれている行を抽出して画像として保存
extract_and_save_data_rows(processed_image)
