import json
import random
import torch as tc
import torch.nn as nn
import torchvision.transforms as tf
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from src.face_verification.FaceEmbedder import FaceEmbedder
# python -m src.face_verification.test

with open("./src/face_verification/data/info.json", "r") as file:
    infos = json.load(file)

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
transformer = tf.Compose([
    tf.Resize(120),
    tf.CenterCrop(112),
    tf.ToTensor(),
    # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_img(path):
    with Image.open(path).convert("RGB") as img:
        img: tc.Tensor = transformer(img)
        _img = to_pil_image(img)
        _img.show()
        img = img.unsqueeze(0)
        return img.to(device)
    
# face_embedder = FaceEmbedder().to(device)
# face_embedder.eval()
# face_embedder.features_layer.face_features.load_state_dict(tc.load("./src/face_verification/params/face_features.pth"), strict=False)
# face_embedder.linear_layer.load_state_dict(tc.load("./src/face_verification/params/linear.pth"))

def test(num: int = 3):
    infos_len = len(infos)
    num = min(num, len(infos))
    for _ in range(num):
        a = random.randint(0, infos_len - 1)
        b = random.randint(0, infos_len - 1)
        info_a = infos[a]
        info_b = infos[b]
        name_a = info_a["name"]
        name_b = info_b["name"]
        img_a = get_img(info_a["path"])
        img_b = get_img(info_b["path"])
        # print(f"{name_a} -- {name_b} -- {face_embedder.distance(img_a, img_b)}")
        
test()
    

