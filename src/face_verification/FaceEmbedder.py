#FaceEmbedder.py
import torch as tc
import torch.nn as nn
import torchvision
import torchvision.models as models
from collections import OrderedDict
from types import SimpleNamespace

class FaceEmbedderResNet50(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        resnet50 = torchvision.models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        resnet50.requires_grad_(False)
        resnet50.eval()
        self.features_layer = nn.Sequential(OrderedDict([
            ("resnet50", resnet50),
        ]))
        self.embedder_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=embedding_dim, bias=True),
            nn.BatchNorm2d(embedding_dim),
            nn.Dropout(0.5)
        )
 
        
    def forward(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self.features_layer(X)
        X = self.embedder_layer(X)
        X = nn.functional.normalize(X, p=2, dim=1)
        return X

    def linear(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self.features_layer(X)
        X = self.embedder_layer(X)
        return X

    def distance(
        self, 
        A: tc.Tensor,
        B: tc.Tensor
    ) -> tc.Tensor: 
        A_vector = self.forward(A)
        B_vector = self.forward(B)
        d = (B_vector - A_vector)**2
        d = tc.sum(d, dim=1)
        return d
    

class FaceEmbedderVGG16(nn.Module):
    class PARAM:
        class FACE_VERIFICATION:
            FACE_FEATURE = "./params/face_verification/face_features.pth"
            LINEAR = "./params/face_verification/linear.pth"
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        vgg16.requires_grad_(False)
        vgg16.eval()
        self.features_layer = nn.Sequential(OrderedDict([
            ("vgg16", vgg16),
            ("face_features", nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            ))
        ]))
        self.embedder_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )
        self.linear_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=embedding_dim, bias=True)
        )
   
    def forward(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self.features_layer(X)
        X = self.embedder_layer(X)
        X = self.linear_layer(X)
        X = nn.functional.normalize(X, p=2, dim=1)
        return X

    def linear(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self.features_layer(X)
        X = self.embedder_layer(X)
        X = self.linear_layer(X)
        return X

    def distance(
        self, 
        A: tc.Tensor,
        B: tc.Tensor
    ) -> tc.Tensor: 
        A_vector = self.forward(A)
        B_vector = self.forward(B)
        d = (B_vector - A_vector)**2
        d = tc.sum(d, dim=1)
        return d
    