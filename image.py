# -*- coding:utf-8 -*-
import cv2
import os
import glob
from scipy import ndimage

# 顔認識する対象を決定（検索ワードを入力）
SearchName = ["双葉杏 ssr", "小日向美穂 ssr", "大槻唯 ssr", "渋谷凛 ssr"]
# 画像の取得枚数の上限
ImgNumber = 600
# CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
ImgSize = (250, 250)
input_shape = (250, 250, 3)


def cut_face():
    # アニメ顔のカスケードファイル
    cascade_path = 'lbpcascade_animeface.xml'
    faceCascade = cv2.CascadeClassifier(cascade_path)

    for name in SearchName:
        # 画像データのあるディレクトリ
        input_data_path = "images/Original/" + str(name)
        # 切り抜いた画像の保存先ディレクトリを作成
        os.makedirs("images/Face/" + str(name) + "_face", exist_ok=True)
        save_path = "images/Face/" + str(name) + "_face/"
        # 収集した画像の枚数(任意で変更)
        image_count = ImgNumber
        # 顔検知に成功した数(デフォルトで0を指定)
        face_detect_count = 0

        print("{}の顔を検出し切り取りを開始します。".format(name))
        # 集めた画像データから顔が検知されたら、切り取り、保存する。
        for i in range(image_count):
            img = cv2.imread(input_data_path + '/' + str(i) + '.jpg', cv2.IMREAD_COLOR)
            if img is None:
                print('image' + str(i) + ':NoFace')
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face = faceCascade.detectMultiScale(gray, 1.1, 3)
                if len(face) > 0:
                    for rect in face:
                        x = rect[0]
                        y = rect[1]
                        w = rect[2]
                        h = rect[3]
                        cv2.imwrite(
                            save_path + 'cutted' + str(face_detect_count) + '.jpg', img[y:y + h, x:x + w])
                        face_detect_count = face_detect_count + 1
                else:
                    print('image' + str(i) + ':NoFace')

    print("顔画像の切り取り作業、正常に動作しました。")


def image_inflated():
    for name in SearchName:
        print("{}の写真を増やします。".format(name))
        in_dir = "images/Face/" + name + "_face/*"
        out_dir = "images/FaceEdited/" + name
        os.makedirs(out_dir, exist_ok=True)
        in_jpg = glob.glob(in_dir)
        # img_file_name_list = os.listdir("images/Face/"+name+"_face/")
        for i in range(len(in_jpg)):
            # print(str(in_jpg[i]))
            img = cv2.imread(str(in_jpg[i]))
            # 回転
            for ang in [-10, 0, 10]:
                img_rot = ndimage.rotate(img, ang)
                img_rot = cv2.resize(img_rot, ImgSize)
                fileName = os.path.join(out_dir, str(i) + "_" + str(ang) + ".jpg")
                cv2.imwrite(str(fileName), img_rot)
                # 閾値
                img_thr = cv2.threshold(
                    img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
                fileName = os.path.join(out_dir, str(i) + "_" + str(ang) + "thr.jpg")
                cv2.imwrite(str(fileName), img_thr)
                # ぼかし
                img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
                fileName = os.path.join(
                    out_dir, str(i) + "_" + str(ang) + "filter.jpg")
                cv2.imwrite(str(fileName), img_filter)
    print("画像の水増しに大成功しました！")


if __name__ == '__main__':
    cut_face()
    image_inflated()
    print("完了しました")
