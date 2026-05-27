import math
import torch as tc
from PIL import Image
from src.utils import FaceDetectionFactory, FaceEmbedderFactory, FaceLandmarkFactory, FaceLandmarkConfig, FaceDetectionConfig, FaceEmbedderConfig, FaceVerificationFactory
from src.face_landmark import FaceLandmark
from src.face_verification import FaceVerification
from torchvision.transforms.functional import crop
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

face_detection_factory = FaceDetectionFactory()
face_landmark_factory = FaceLandmarkFactory()
face_embedder_factory = FaceEmbedderFactory()
face_verification_factory = FaceVerificationFactory(
    face_detection_factory=face_detection_factory,
    face_landmark_factory=face_landmark_factory,
    face_embedder_factory=face_embedder_factory
)

def detect_face(
    image: Image.Image,
    config: FaceDetectionConfig,
) -> list[dict]:
    model = face_detection_factory.get(config)
    preds = model.detect(image)
    res = []
    for pred in preds:
        bbox = pred[0]
        res.append({
            "score" : pred[1],
            "bbox" : {"x1" : bbox[0], "y1" : bbox[1], "x2" : bbox[2], "y2" : bbox[3]}
        })
    return res

def detect_landmark(
    image: Image.Image,
    config: FaceLandmarkConfig
) -> list[int]:
    image = image.convert("RGB")
    model = face_landmark_factory.get(config=config)
    landmark = model.detect(image=image)
    return landmark

def compare_face(
    images: list[Image.Image],
    face_detection_config: dict,
    face_landmark_config: dict,
    face_embedder_config: dict
) -> tuple[float, float, list[bool]] | str:
    face_detection_config = FaceDetectionConfig(config=face_detection_config)
    face_landmark_config = FaceLandmarkConfig(config=face_landmark_config)
    face_embedder_config = FaceEmbedderConfig(config=face_embedder_config)
    
    model = face_verification_factory.get(face_detection_config, face_landmark_config, face_embedder_config)
    [res_1, res_2], distance = model.verify(images[0], images[1])

    frontals = [res_1[1], res_2[1]]
    percent = 1 / (1 + math.exp(9*(distance - 1)))
    return distance, percent, frontals

def sample_face(
    image: Image.Image,
    face_detection_config: dict,
    face_landmark_config: dict,
) -> tuple[list[dict], bool, Image.Image | None]:
    face_detection_config = FaceDetectionConfig(config=face_detection_config)
    face_landmark_config = FaceLandmarkConfig(config=face_landmark_config)
    
    model = face_verification_factory.get(face_detection_config, face_landmark_config)
    detection, frontal, sample_image = model.sample(sample_image=image)

    return detection, frontal, sample_image


def verify_face(
    sample_image: Image.Image,
    verify_image: Image.Image,
    face_detection_config: dict,
    face_landmark_config: dict,
    face_embedder_config: dict
) -> tuple[list[tuple[list[dict], bool]], float, bool]:
    face_detection_config = FaceDetectionConfig(config=face_detection_config)
    face_landmark_config = FaceLandmarkConfig(config=face_landmark_config)
    face_embedder_config = FaceEmbedderConfig(config=face_embedder_config)

    model = face_verification_factory.get(face_detection_config, face_landmark_config, face_embedder_config)
    
    [res_1, (detection, frontal)], distance = model.verify(sample_image, verify_image)
    percent = 1 / (1 + math.exp(9*(distance - 1)))
    ok = True if distance < 0.95 else False
    return detection, frontal, distance, percent, ok