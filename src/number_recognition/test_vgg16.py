import torch as tc
from src.number_recognition.VGG16 import VGG16
from PIL import Image
from src.ImageTool import ImageTool
# python -m number_recognition.test_vgg16
vgg16 = VGG16()
vgg16.load_state_dict(tc.load("./number_recognition/vgg16.pth"))
with Image.open("./number_recognition/images/123-0.png") as img:
    img = img.resize((50, 50))
    img = ImageTool.RGB_to_gray_encoder(img)
    print(vgg16.predict(img), vgg16.predict_proba(img))
    