import torch as tc
import torchvision.transforms as tf
from PIL import Image, ImageDraw
from src.face_detection.faster_rcnn import FasterRCNN

# python -m src.face_detection.test_faster_rcnn
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
faster_rcnn = FasterRCNN()
faster_rcnn = faster_rcnn.eval()

tranform = tf.Compose([
    tf.ToTensor()
])

faster_rcnn.load_state_dict(tc.load(FasterRCNN.PARAM.FACE_DETECTION.PARAMS, map_location=device))

with Image.open("./dataset/images/messi/008.jpg").convert("RGB") as img:
    tensor_img = tranform(img).to(device)
    draw = ImageDraw.Draw(img)
    threshold = 0.2
    with tc.no_grad():
        res = faster_rcnn([tensor_img])[0]
        boxes = res["boxes"]
        labels = res["labels"]
        scores = res["scores"]
    color = ["red", "blue", "green", "white", "black", "yellow"]
    for i in range(4):
        box = boxes[i]
        print(labels[i], scores[i])
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color[i], width=3)

    img.show()
        