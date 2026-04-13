import torch as tc
import torchvision.transforms as tf
from src.face_detection.HaveFace import HaveFace
from PIL import Image
from src.ImageTool import ImageTool
# python -m face_detection.test_have_face

transformer = tf.Compose([
    tf.Resize((224, 224)),
    tf.ToTensor(),
    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

have_face = HaveFace()
have_face._linear.load_state_dict(tc.load("./face_detection/have_face_linear.pth"))
have_face._features.face_features.load_state_dict(tc.load("./face_detection/have_face_face_features.pth"))

with Image.open("./face_detection/images/test/face/7.jpg") as img:
    img: tc.Tensor = transformer(img)
    img = img.unsqueeze(0)
    print(have_face.predict(img))
    
with Image.open("./face_detection/images/test/non_face/5.png") as img:
    img: tc.Tensor = transformer(img)
    img = img.unsqueeze(0)
    print(have_face.predict(img))
    