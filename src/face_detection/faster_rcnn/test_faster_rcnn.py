import argparse
import torch as tc
import torchvision.transforms as tf
from PIL import Image, ImageDraw
from src.face_detection.faster_rcnn.model import FasterRCNN
# python -m src.face_detection.test_faster_rcnn -m vgg16 -p wider_face -pth ./dataset/football_player/messi/001.png 

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backbone", type=str, default="resnet34")
parser.add_argument("-ds", "--dataset", type=str, default="wider_face")
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-pth", "--path", type=str, default="./dataset/football_player/nhat/001.png")
args = parser.parse_args()

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
faster_rcnn = FasterRCNN(backbone=args.backbone, dataset=args.dataset)
faster_rcnn = faster_rcnn.eval()
faster_rcnn.load_state()
tranform = tf.Compose([
    tf.ToTensor()
])

with Image.open(args.path).convert("RGB") as img:
    tensor_img = tranform(img).to(device)
    draw = ImageDraw.Draw(img)
    with tc.no_grad():
        res = faster_rcnn([tensor_img])[0]
        boxes = res["boxes"]
        labels = res["labels"]
        scores = res["scores"]
    color = ["red", "blue", "green", "white", "black", "yellow"]
    for i in range(len(boxes)):
        box = boxes[i]
        print(box, scores[i])
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color[0], width=5)

    img.show()

# for folder in os.listdir("./dataset/images"):
#     if(folder != "pep"):
#         continue
#     for file in os.listdir("./dataset/images/" + folder):
#         with Image.open("./dataset/images/" + folder + "/" + file).convert("RGB") as img:
#             tensor_img = tranform(img).to(device)
#             draw = ImageDraw.Draw(img)
#             threshold = 0.2
#             with tc.no_grad():
#                 res = faster_rcnn([tensor_img])[0]
#                 boxes = res["boxes"]
#                 labels = res["labels"]
#                 scores = res["scores"]
#             color = ["red", "blue", "green", "white", "black", "yellow"]
#             for i in range(min(len(boxes), 1)):
#                 box = boxes[i]
#                 print(labels[i], scores[i])
#                 draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color[i], width=3)

#             img.show()