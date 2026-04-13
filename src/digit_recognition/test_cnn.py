import torch as tc
from PIL import Image
from src.digit_recognition.CNN import digit_recognition
from src.ImageTool import ImageTool
# python -m digit_recognition.test_cnn

with Image.open("./digit_recognition/dataset/images/2-3.png") as img:
    X = ImageTool.RGB_to_gray_encoder(img)
    num = digit_recognition.predict(X)
    proba = digit_recognition.predict_proba(X)[0][num]
    print(num, proba)

