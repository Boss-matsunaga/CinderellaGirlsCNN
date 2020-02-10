# 2割をテストデータに移行
import shutil
import random
import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential

# 顔認識する対象を決定（検索ワードを入力）
SearchName = ["双葉杏 ssr", "小日向美穂 ssr", "大槻唯 ssr", "渋谷凛 ssr"]
# 画像の取得枚数の上限
ImgNumber = 600
# CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
ImgSize = (250, 250)
input_shape = (250, 250, 3)


def main():
    for name in SearchName:
        in_dir = "images/FaceEdited/"+name+"/*"
        in_jpg = glob.glob(in_dir)
        img_file_name_list = os.listdir("images/FaceEdited/"+name+"/")
        # img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる
        random.shuffle(in_jpg)
        os.makedirs('images/test/' + name, exist_ok=True)
        for t in range(len(in_jpg)//5):
            shutil.move(str(in_jpg[t]), "images/test/"+name)

    # 教師データのラベル付け
    X_train = []
    Y_train = []
    for i in range(len(SearchName)):
        img_file_name_list = os.listdir("images/FaceEdited/"+SearchName[i])
        print("{}:トレーニング用の写真の数は{}枚です。".format(
            SearchName[i], len(img_file_name_list)))

        for j in range(0, len(img_file_name_list)-1):
            n = os.path.join("images/FaceEdited/" +
                             SearchName[i]+"/", img_file_name_list[j])
            img = cv2.imread(n)
            if img is None:
                print('image' + str(j) + ':NoImage')
                continue
            else:
                r, g, b = cv2.split(img)
                img = cv2.merge([r, g, b])
                X_train.append(img)
                Y_train.append(i)

    print("")

    # テストデータのラベル付け
    X_test = []  # 画像データ読み込み
    Y_test = []  # ラベル（名前）
    for i in range(len(SearchName)):
        img_file_name_list = os.listdir("images/test/"+SearchName[i])
        print("{}:テスト用の写真の数は{}枚です。".format(
            SearchName[i], len(img_file_name_list)))
        for j in range(0, len(img_file_name_list)-1):
            n = os.path.join(
                "images/test/"+SearchName[i]+"/", img_file_name_list[j])
            img = cv2.imread(n)
            if img is None:
                print('image' + str(j) + ':NoImage')
                continue
            else:
                r, g, b = cv2.split(img)
                img = cv2.merge([r, g, b])
                X_test.append(img)
                Y_test.append(i)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = to_categorical(Y_train)
    y_test = to_categorical(Y_test)

    # モデルの定義
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3),
                     strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    # 分類したい人数を入れる
    model.add(Dense(len(SearchName)))
    model.add(Activation('softmax'))

    # コンパイル
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 学習
    history = model.fit(X_train, y_train, batch_size=70,
                        epochs=50, verbose=1, validation_data=(X_test, y_test))

    # 汎化制度の評価・表示
    score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
    print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

    # acc, val_accのプロット
    plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
    plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.show()

    # モデルを保存
    model.save("MyModel.h5")


if __name__ == "__main__":
    main()
    print("完了しました")
