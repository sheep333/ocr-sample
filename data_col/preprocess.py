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
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    cv2.imwrite('binary.png', binary)
    return binary

def extract_lines_without_horizontal(binary_image, min_line_height=2, min_black_pixel_count=300):
    # 水平方向にピクセルを走査して、行を検出
    h, w = binary_image.shape
    line_starts = []
    line_ends = []
    
    # 画像の各行を走査
    for i in range(h):
        black_pixel_count = np.sum(binary_image[i, :] == 255)  # 黒いピクセルの数をカウント
        print("Black pixel count:", black_pixel_count)
        if black_pixel_count > min_black_pixel_count:
            print(len(line_ends) == 0 or (i - line_ends[-1]))
            if len(line_ends) == 0 or (i - line_ends[-1]) > min_line_height:
                line_starts.append(i)
            line_ends.append(i)
    
    # 行の開始と終了のペアを作成し、不正なペアをフィルタリング
    lines = [(start, end) for start, end in zip(line_starts, line_ends) if start < end]

    # 検出された行の情報をプリントして確認
    print("Filtered lines:", lines)
    
    return lines

def extract_line_data(image, lines):
    line_images = []
    for (start, end) in lines:
        if end > start:  # 有効な行の範囲であることを確認
            line_img = image[start:end, :]
            line_images.append(line_img)
            
            # 行ごとの画像を表示（サイズが有効な場合のみ）
            if line_img.shape[0] > 0 and line_img.shape[1] > 0:
                cv2.imshow('Line', line_img)
                cv2.waitKey(0)
    
    return line_images

# 画像のパス
image_path = '../input/aaa.png'

# 画像を前処理
binary_image = preprocess_image_for_lines(image_path)

# 行を検出
lines = extract_lines_without_horizontal(binary_image)

# 行ごとのデータを抽出
line_images = extract_line_data(binary_image, lines)
