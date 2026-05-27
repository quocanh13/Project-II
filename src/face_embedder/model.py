#FaceEmbedder.py
import torch as tc
import torch.nn as nn
import torchvision
import torchvision.models as models
from collections import OrderedDict
from typing import Literal
from abc import ABC, abstractmethod

class FaceEmbedder(nn.Module, ABC):
    def __init__(
        self, 
        params: str
    ):
        super().__init__()
        self.params = params
        self.features_layer: nn.Module
        self.embedder_layer: nn.Module
        
    def load_state(self, device: str | None = None):
        if(device == None):
            device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
        self.load_state_dict(tc.load(self.params,  map_location=device))
        
    def save_state(self):
        tc.save(self.state_dict(), self.params)

    def forward(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self.features_layer(X)
        X = self.embedder_layer(X)
        X = nn.functional.normalize(X, p=2, dim=1)
        return X

    def distance(
        self, 
        A: tc.Tensor,
        B: tc.Tensor
    ) -> tc.Tensor: 
        A_vector = self.forward(A)
        B_vector = self.forward(B)
        d = tc.norm(B_vector - A_vector, dim=1)
        return d

class HardTripletLoss(nn.Module):
    def __init__(
        self, 
        margin = 0.6
        ):
        super().__init__()
        self._margin = margin
    
    def forward(self, embeddings: tc.Tensor, labels: tc.Tensor):
        device = embeddings.device

        dist = tc.cdist(embeddings, embeddings, p=2)

        labels = labels.unsqueeze(1)

        diag_mask = tc.eye(labels.size(0), dtype=tc.bool, device=device)

        pos_mask = (labels == labels.T) & ~diag_mask
        neg_mask = (labels != labels.T)

        hard_pos = dist.masked_fill(~pos_mask, float('-inf'))
        hard_neg = dist.masked_fill(~neg_mask, float('inf'))

        hard_pos = hard_pos.max(dim=1)[0]
        hard_neg = hard_neg.min(dim=1)[0]

        loss = nn.functional.relu(hard_pos - hard_neg + self._margin)
        return loss.mean()

class SemiHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: tc.Tensor, labels: tc.Tensor):
        device = embeddings.device
        N = embeddings.size(0)

        dist = tc.cdist(embeddings, embeddings, p=2)
        dist = tc.clamp(dist, min=1e-12)

        labels = labels.unsqueeze(1)

        diag_mask = tc.eye(N, dtype=tc.bool, device=device)
        pos_mask = (labels == labels.T) & ~diag_mask
        neg_mask = (labels != labels.T)

        pos_dist = dist.masked_fill(~pos_mask, float('-inf'))
        hardest_pos = pos_dist.max(dim=1)[0]   # (N,)

        neg_dist = dist.masked_fill(~neg_mask, float('inf'))

        d_pos = hardest_pos.unsqueeze(1)

        semi_mask = (
            (neg_dist > d_pos) &
            (neg_dist < (d_pos + self.margin))
        )

        semi_neg = neg_dist.clone()
        semi_neg[~semi_mask] = float('inf')

        semi_hard_neg = semi_neg.min(dim=1)[0]

        hard_neg = neg_dist.min(dim=1)[0]

        final_neg = tc.where(
            tc.isinf(semi_hard_neg),
            hard_neg,
            semi_hard_neg
        )

        loss = nn.functional.relu(hardest_pos - final_neg + self.margin)

        return loss.mean()
    
class FaceEmbedderVGG16(FaceEmbedder):
    class PARAMS:
        class FACE_EMBEDDER:
            CELEB_A = "./params/face_embedder/vgg16/celeb_A.pth"
    
    PARAMS_DICT = {
        "celeb_A" : PARAMS.FACE_EMBEDDER.CELEB_A
    }
    
    def __init__(self, embedding_dim: int = 512, params : Literal["celeb_A"] = "celeb_A"):
        super().__init__(self.PARAMS_DICT[params])
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        vgg16.requires_grad_(False)
        for param in vgg16[24:].parameters():
            param.requires_grad = True
        self.features_layer = nn.Sequential(
            vgg16,
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.embedder_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=embedding_dim, bias=True)
        )
    
    def load_params(self, device: str | None = None):
        if(device == None):
            device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
        self.load_state_dict(tc.load(self.params, map_location=device))
        return self
    
    def save_params(self):
        tc.save(self.state_dict(), self.params)
        
    def forward(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self.features_layer(X)
        X = self.embedder_layer(X)
        X = nn.functional.normalize(X, p=2, dim=1)
        return X
    
    def distance(
        self, 
        A: tc.Tensor,
        B: tc.Tensor
    ) -> tc.Tensor: 
        A_vector = self.forward(A)
        B_vector = self.forward(B)
        d = tc.norm(B_vector - A_vector, dim=1)
        return d

class FaceEmbedderResNet18(FaceEmbedder):
    class PARAMS:
        class FACE_EMBEDDER:
            CELEB_A = "./params/face_embedder/resnet18/celeb_A.pth"
    
    PARAMS_DICT = {
        "celeb_A" : PARAMS.FACE_EMBEDDER.CELEB_A
    }
    
    def __init__(self, embedding_dim: int = 512, params : Literal["celeb_A"] = "celeb_A"):
        super().__init__(self.PARAMS_DICT[params])
        resnet18 = torchvision.models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        resnet18.requires_grad_(True)
        self.features_layer = nn.Sequential(OrderedDict([
            ("resnet18", resnet18),
        ]))
        self.embedder_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=embedding_dim, bias=True),
        )

class FaceEmbedderResNet34(FaceEmbedder):
    class PARAMS:
        class FACE_EMBEDDER:
            CELEB_A = "./params/face_embedder/resnet34/celeb_A.pth"
            
    PARAMS_DICT = {
        "celeb_A" : PARAMS.FACE_EMBEDDER.CELEB_A
    }
    
    def __init__(self, embedding_dim: int = 512, params : Literal["celeb_A"] = "celeb_A"):
        super().__init__(self.PARAMS_DICT[params])
        resnet34 = torchvision.models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        resnet34 = nn.Sequential(*list(resnet34.children())[:-1])
        resnet34.requires_grad_(True)
        self.features_layer = nn.Sequential(OrderedDict([
            ("resnet34", resnet34),
        ]))
        self.embedder_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=embedding_dim, bias=True),
        )

class FaceEmbedderResNet50(FaceEmbedder):
    class PARAMS:
        class FACE_EMBEDDER:
            CELEB_A = "./params/face_embedder/resnet50/celeb_A.pth"
    
    PARAMS_DICT = {
        "celeb_A" : PARAMS.FACE_EMBEDDER.CELEB_A
    }
    
    def __init__(self, embedding_dim: int = 512, params : Literal["celeb_A"] = "celeb_A"):
        super().__init__(self.PARAMS_DICT[params])
        resnet50 = torchvision.models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        resnet50.requires_grad_(False)
        for param in resnet50[5].parameters():
            param.requires_grad = True
        for param in resnet50[6].parameters():
            param.requires_grad = True
        for param in resnet50[7].parameters():
            param.requires_grad = True
        self.features_layer = resnet50
        self.embedder_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(in_features=2048, out_features=embedding_dim, bias=True),
        )



    

    