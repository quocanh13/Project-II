import cv2
import math
import torch as tc
import numpy as np
import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image
from typing import Literal
from torchvision.transforms.functional import crop
from src.face_detection import  FaceDetection
from src.face_landmark import FaceLandmark
from src.face_embedder.model import FaceEmbedder

class FaceVerification(nn.Module):
    def __init__(
        self, 
        face_detection : FaceDetection,
        face_landmark: FaceLandmark,
        face_embedder: FaceEmbedder
    ):
        super().__init__()
        self.face_detection = face_detection.eval()
        self.face_landmark = face_landmark.eval()
        self.face_embedder = face_embedder.eval()
        self.transformer = tf.Compose([
            tf.Resize(224),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
              
    def detect_face(self, image: Image.Image, num):
        return self.face_detection.detect(img=image)
    
    def extract_features(
        self, 
        image: Image.Image
        ) -> tuple[list[tuple[list[int], float]], list[int], bool, tc.Tensor]:
        frontal = True
        with tc.no_grad():
            detection = self.face_detection.detect(image)
            landmark = self.face_landmark.detect(image=image)
            if(FaceLandmark.is_frontal(landmark=landmark) == False):
                frontal = False
            image: Image.Image = FaceLandmark.align_face(image=image, landmark=landmark)
            # image.show()
            detection_final = self.face_detection.detect(image)
            if(len(detection) <= 0):
                detection_final = [([0, 0, image.size[0], image.size[1]], 0)]    
            bbox, _ = detection_final[0]
                
            image = crop(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
            # image.show()
            image:tc.Tensor = self.transformer(img=image)
            image = image.unsqueeze(0)
            features = self.face_embedder(image)
        return detection, landmark, frontal, features[0]

    def sample(
        self, 
        sample_image: Image.Image
    ) -> tuple[list[tuple[dict]], list[int], bool, Image.Image | None]:
        with tc.no_grad():
            detection = self.face_detection.detect(sample_image)
            landmark = self.face_landmark.detect(image=sample_image)
            frontal = FaceLandmark.is_frontal(landmark=landmark)
            t = detection
            detection = []
            for bbox, score in t:
                detection.append({"bbox" : {"x1" : bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}, "score" : score})
            if(len(detection) <= 0 and not frontal):
                sample_image = None
        return detection, landmark, frontal, sample_image

    def verify(
        self, 
        sample_image: Image.Image, 
        verify_image: Image.Image,
    ) -> tuple[list[tuple[dict], list[int], bool], float]:
        res_1 = self.extract_features(sample_image)
        res_2 = self.extract_features(verify_image)
        
        detection_1, landmark_1, frontal_1, features_1 = res_1
        detection_2, landmark_2, frontal_2, features_2 = res_2
        t = detection_1
        detection_1 = []
        for bbox, score in t:
            detection_1.append({"bbox" : {"x1" : bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}, "score" : score})
            
        t = detection_2
        detection_2 = []
        for bbox, score in t:
            detection_2.append({"bbox" : {"x1" : bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}, "score" : score})
        distance = tc.norm(features_1 - features_2, dim=0)
        return [(detection_1, landmark_1, frontal_1), (detection_2, landmark_2, frontal_2)], distance.item()