import cv2
import numpy as np

def preprocess_image_for_lines(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーを適用してノイズを除去（ぼやけの軽減）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # シャープ化のためのカーネルを作成
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # アダプティブ二値化を使用して文字を強調
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)
    
    return binary

def save_non_white_areas_from_binary(binary_image, max_black_threshold=0.9, min_black_threshold=0.1):
    h, w = binary_image.shape
    min_black_pixel_count = int(w * min_black_threshold)  # 黒いピクセルの最小数
    max_black_pixel_count = int(w * max_black_threshold)  # 黒いピクセルの最大数

    start = None
    saved_image_count = 0

    for i in range(h):
        black_pixel_count = np.sum(binary_image[i, :] == 255)
        print("Black pixel count:", black_pixel_count)
        print("min_black_pixel_count:", min_black_pixel_count)
        print("max_black_pixel_count:", max_black_pixel_count)
        
        if min_black_pixel_count < black_pixel_count < max_black_pixel_count:
            if start is None:
                start = i  # 黒いピクセルが1割以上、かつ9割未満の行を記録
        else:
            if start is not None:
                # 黒いピクセルが1割以上、かつ9割未満の範囲を画像として保存
                cropped_img = binary_image[start:i, :]
                cv2.imwrite(f'output/binary_non_white_area_{saved_image_count}.png', cropped_img)
                saved_image_count += 1
                start = None  # 次のセグメントを検出するためにリセット
    
    # 最後のセグメントを保存
    if start is not None:
        cropped_img = binary_image[start:h, :]
        cv2.imwrite(f'binary_non_white_area_{saved_image_count}.png', cropped_img)

    print(f'{saved_image_count} images saved.')

# 画像のパス
image_path = '../input/aaa.png'

# 画像を前処理（binary_imageを取得）
binary_image = preprocess_image_for_lines(image_path)

# 二値化された画像から黒が1割以上かつ9割未満の行が続く部分のみを保存
save_non_white_areas_from_binary(binary_image)
