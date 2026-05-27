import argparse
from PIL import Image
from src.face_verification import FaceVerification
from src.utils import *
# python -m src.face_verification.test
parser = argparse.ArgumentParser()
parser.add_argument("-fd", "--face_detection", type=str, default="resnet34")
parser.add_argument("-fl", "--face_landmark", type=str, default="resnet34")
parser.add_argument("-fe", "--face_embedder", type=str, default="resnet34")
args = parser.parse_args()

face_detection = FaceDetectionConfig(algorithm="faster-rcnn", backbone=args.face_detection, dataset="wider_face")
face_landmark = FaceLandmarkConfig(backbone=args.face_landmark)
face_embedder = FaceLandmarkConfig(backbone=args.face_embedder)
face_verification = FaceVerification(
    face_detection = FaceDetectionFactory.create(face_detection),
    face_landmark=FaceLandmarkFactory.create(face_landmark),
    face_embedder=FaceEmbedderFactory.create(face_embedder)
)

for i in range(1):
    for j in range(1):
        with Image.open(f"./dataset/football_player/nhat/002.png").convert("RGB") as image:
            image_1 = image
        with Image.open(f"./dataset/football_player/nhat/003.png").convert("RGB") as image:
            image_2 = image
        print(face_verification.verify(image_1, image_2))
# with Image.open(f"./src/face_verification/1.png").convert("RGB") as image:
# # with Image.open(f"./dataset/football_player/neymar/001.png").convert("RGB") as image:
#     landmark = face_verification.face_landmark.detect(image=image)
#     print(face_verification.is_frontal(image=image, landmark=landmark))
    