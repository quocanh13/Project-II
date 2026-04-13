import torch as tc
import torch.nn as nn
import torchvision
from collections import OrderedDict

class FaceEmbedder(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        vgg16.requires_grad_(False)
        vgg16.eval()
        self.features = nn.Sequential(OrderedDict([
            ("vgg16", vgg16),
            ("face_features", nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            ))
        ]))
        self.embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=num_class, bias=True)
        )
 
        
    def forward(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self.features(X)
        X = self.embedder(X)
        X = self.linear(X)
        # X = nn.functional.normalize(X, p=2, dim=1)
        return X

