import numpy as np
import cv2


def get_gray_masked_img(img_bgr, threshold=127, save=False):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # グレースケール変換
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 3)     # スムージング
    if save:
        cv2.imwrite('gray.jpg', img_gray)
    _, img_masked = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY) # Hがthreshold以上のピクセルを255, それ以外を0に変換
    return img_masked

def get_hue_masked_img(img_bgr, threshold=127, save=False):
    img_HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) # HSV変換：「色相(Hue)」「彩度(Saturation)」「明度(Value・Brightness)」
    img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)     # スムージング
    img_H, img_S, img_V = cv2.split(img_HSV)           # HSV分離
    if save:
        cv2.imwrite('H.jpg', img_H)
    _, img_masked = cv2.threshold(img_H, threshold, 255, cv2.THRESH_BINARY) # Hがthreshold以上のピクセルを255, それ以外を0に変換
    return img_masked


if __name__ == '__main__':
    img_bgr = cv2.imread('883957.png')

    # 全体の輪郭の検出防止のために一番外側に白線の四角を描く
    height, width, channels = img_bgr.shape[:3]
    img_bgr = cv2.rectangle(
                img_bgr,                # 書き込む画像（上書き）
                (0, 0),                 # 端点1
                (height-1, width-1),    # 端点2
                (255, 255, 255),        # 色
                3                       # 太さ
            )

    # マスク
    img_masked = get_gray_masked_img(img_bgr, 100, save=True) # グレースケールでマスクした画像を取得
    # img_masked = get_hue_masked_img(img_bgr, 127, save=True) # 色相でマスクした画像を取得
    cv2.imwrite('masked.jpg', img_masked)

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(
                            img_masked,             # 2値画像
                            cv2.RETR_LIST,          # 階層構造：LIST = 階層構造を持たない
                            cv2.CHAIN_APPROX_SIMPLE # 近似手法：NONE = 近似なし, SIMPLE = 輪郭の線分の端点のみ保持
                        )                       # ラベル, 輪郭, 階層を返却

    # 必要な輪郭を選んで輪郭描画
    for i in range(len(contours)):

        # 面積が画像全体に対して十分小さい or 面積が画像全体と同等なものは無視
        area = cv2.contourArea(contours[i])
        if area <= width*height*0.0001 or area >= width*height*0.99:
            continue

        # 輪郭描画
        img_cp = np.copy(img_bgr)
        bounded = cv2.drawContours(
                    img_cp,     # 書き込む画像（上書き）
                    contours,   # 輪郭
                    i,          # 書き込む輪郭の番号
                    (255,0,0),  # 色
                    2           # 太さ
                )

        # 輪郭の端点の描画
        points = contours[i].reshape(-1, 2) # point = (x, y)の配列の形式にする
        for x, y in points:
            bounded = cv2.drawMarker(img_cp, (x, y), (0, 0, 255), markerSize=3)

        # 保存
        cv2.imwrite('bounded%d.jpg' % i, bounded)
        points = [(x, y*(-1) + (height - 1)) for x, y in points]
        np.savetxt('extracted_path%d.csv' % i, points, delimiter=',', fmt='%d')
