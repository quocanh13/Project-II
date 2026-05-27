import cv2
import math
import numpy as np
import torchvision
import torch as tc
import torch.nn as nn
import torchvision.transforms as tf
from typing import Literal
from PIL import Image, ImageDraw

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels: int
        pass

class BackboneResnet18(Backbone):
    def __init__(self):
        super().__init__()
        module = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.module = nn.Sequential(*list(module.children())[:-1])
        self.module.requires_grad_(False)
        self.module[6].requires_grad_(True)
        self.module[7].requires_grad_(True)
        self.out_channels = 512
    
    def forward(self, X: tc.Tensor):
        X = self.module(X)
        return X
    
class BackboneResnet34(Backbone):
    def __init__(self):
        super().__init__()
        module = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.module = nn.Sequential(*list(module.children())[:-1])
        self.module.requires_grad_(False)
        self.module[6].requires_grad_(True)
        self.module[7].requires_grad_(True)
        self.out_channels = 512
    
    def forward(self, X: tc.Tensor):
        X = self.module(X)
        return X

class FaceLandmark(nn.Module):
    BACKBONE = {
        "resnet18" : {"class" : BackboneResnet18, "params" : "./params/face_landmark/resnet18.pth"},
        "resnet34" : {"class" : BackboneResnet34, "params" : "./params/face_landmark/resnet34.pth"},
    }
    
    def __init__(self, backbone: Literal["resnet18", "resnet34"] = "resnet18"):
        super().__init__()
        backbone = self.BACKBONE.get(backbone, self.BACKBONE["resnet18"])
        self.params = backbone["params"]
        self.backbone: Backbone = backbone["class"]()
        self.backbone.requires_grad_(True)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )
        self.transformer = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _training_forward(self, X: tc.Tensor) -> tc.Tensor:
        X = self.backbone(X)
        X = self.linear(X)
        return X
    
    def _eval_forward(self, image: Image.Image) -> list[int]:
        w, h = image.size
        image: tc.Tensor = self.transformer(image)
        image = image.unsqueeze(0)
        landmark = self._training_forward(image)[0]
        landmark[[0, 2, 4, 6, 8]] *= w 
        landmark[[1, 3, 5, 7, 9]] *= h
        return landmark.round().int().tolist()
        

    def forward(self, X: tc.Tensor | Image.Image) -> tc.Tensor | list[int]:
        if(self.training):
            return self._training_forward(X)
        else:
            return self._eval_forward(X)
    
    def detect(self, image: Image.Image) -> list[int]:
        self.eval()
        w, h = image.size
        image_tc: tc.Tensor = self.transformer(image)
        image_tc = image_tc.unsqueeze(0)
        landmark = self._training_forward(image_tc)[0]
        landmark[[0, 2, 4, 6, 8]] *= w 
        landmark[[1, 3, 5, 7, 9]] *= h
        landmark = landmark.round().int().tolist()
        # self.show_landmark(image=image, landmark=landmark)
        # lex, ley, rex, rey, nx, ny, lmx, lmy, rmx, rmy = landmark[:]
        # dln = math.sqrt((lex - nx)**2 + (ley - ny)**2)
        # drn = math.sqrt((rex - nx)**2 + (rey - ny)**2)
        # ratio = dln / drn
        
        # print(dln, drn)
        # print(ratio, "ratio")
        return landmark
    
    @staticmethod
    def align_face(image: Image.Image, landmark: list[int]):
        lx, ly, rx, ry = landmark[:4]

        angle = math.degrees(
            math.atan2(ry - ly, rx - lx)
        )

        center = ((lx + rx) / 2,(ly + ry) / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
        aligned = cv2.warpAffine(np.array(image), M, (image.width, image.height))

        return Image.fromarray(aligned)
    
    @staticmethod
    def is_frontal(landmark: list[int]):
        lex, ley, rex, rey, nx, ny, lmx, lmy, rmx, rmy = landmark[:]
        
        dln = math.sqrt((lex - nx)**2 + (ley - ny)**2)
        drn = math.sqrt((rex - nx)**2 + (rey - ny)**2)
        ratio = dln / drn
        print(ratio)
        if ratio < 0.85 or ratio> 1.15:
            return False
        
        return True
    
    @staticmethod
    def show_landmark(image: Image.Image, landmark: list[int], radius = 5):
        t_image = image.copy()
        draw = ImageDraw.Draw(t_image)
        for i in range(5):
            draw.ellipse(
                (landmark[2*i] - radius, landmark[2*i + 1] - radius, landmark[2*i] + radius, landmark[2*i + 1] + radius),
                outline="red",   
                width=5,         
                fill="red"      
            )
        t_image.show()   

    def load_state(self, device: str | None = None):
        if(device == None):
            device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
        self.load_state_dict(tc.load(self.params,  map_location=device))
        
    def save_state(self):
        tc.save(self.state_dict(), self.params)
        