from google.cloud import vision
import io

def detect_text(image_path):
    # クライアントの初期化
    client = vision.ImageAnnotatorClient()

    # 画像ファイルを読み込む
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # テキスト検出を実行
    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(response.error.message))

# 前処理した画像を使ってOCR
detect_text('output/binary_non_white_area_35.png')
