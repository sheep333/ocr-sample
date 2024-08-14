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

def extract_lines_and_save_images(image, binary_image, min_black_pixel_count=50):
    h, w = binary_image.shape
    black_pixel_counts = np.sum(binary_image == 255, axis=1)  # 各行の黒いピクセルの数を計算

    # 黒い行を検出
    black_line_indices = np.where(black_pixel_counts > min_black_pixel_count)[0]

    # 切り出された画像を保存
    for i in range(1, len(black_line_indices)):
        start = black_line_indices[i - 1]
        end = black_line_indices[i]
        
        if end - start > 1:  # 黒い行の間に十分なスペースがある場合
            cropped_img = image[start:end, :]  # 四角で囲んだ領域を切り出す
            cv2.imwrite(f'line_{i}.png', cropped_img)  # 切り出した画像を保存
    
    cv2.destroyAllWindows()

# 画像のパス
image_path = '../input/aaa.png'

# 画像を前処理
binary_image = preprocess_image_for_lines(image_path)

# 元の画像も読み込む
original_image = cv2.imread(image_path)

# 黒い行の間を四角で囲い、その領域を切り出して保存
extract_lines_and_save_images(original_image, binary_image)
