import argparse
from PIL import Image, ImageDraw
from src.face_landmark.model import FaceLandmark
#python -m src.face_landmark.test_face_landmark
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-b", "--backbone", type=str, default="resnet18")
parser.add_argument("-r", "--radius", type=int, default=5)
parser.add_argument("-pth", "--path", type=str, default="./dataset/football_player/messi/001.png")
args = parser.parse_args()

device = args.device
backbone = args.backbone
radius = args.radius
path = args.path

model = FaceLandmark(backbone=backbone)
model.to(device=device)
model.load_state(device=device)
model.eval()

with Image.open(path).convert("RGB") as image:
    draw = ImageDraw.Draw(image)
    landmark = model.detect(image)
    for i in range(5):
        draw.ellipse(
            (landmark[2*i] - radius, landmark[2*i + 1] - radius, landmark[2*i] + radius, landmark[2*i + 1] + radius),
            outline="red",   
            width=5,         
            fill="red"      
        )
    image.show()
        