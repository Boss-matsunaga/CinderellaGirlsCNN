from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import glob

model = load_model('MyModel.h5')
label = ["双葉杏 ssr", "小日向美穂 ssr", "大槻唯 ssr", "渋谷凛 ssr"]

file_path = glob.glob('images/Face/双葉杏 ssr_face/*')
for name in file_path:
    img = img_to_array(load_img(name, target_size=(250, 250)))
    img_nad = img_to_array(img) / 255
    img_nad = img_nad[None, ...]

    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]
    print('name:', pred_label)
    print('score:', pred)
