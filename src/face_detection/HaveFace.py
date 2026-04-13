import torch as tc
import torch.nn as nn
import torchvision
from collections import OrderedDict

class HaveFace(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        vgg16.requires_grad_(False)
        self._features = nn.Sequential(OrderedDict([
            ("vgg16", vgg16),
            ("face_features", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ("max_pooling_1", nn.MaxPool2d(2, stride=2))
            ("relu_1", nn.ReLU()),  
        ]))
        self._linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self._output = nn.Sequential(
            nn.Softmax(dim=1)
        )    
        
    def forward(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:    
        X = self._features(X)
        X = self._linear(X)
        return X
    
    def predict(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:
        X = self.forward(X)
        X = self._output(X)
        return X

