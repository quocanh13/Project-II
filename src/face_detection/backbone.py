import torch as tc
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class MiniVGG16(nn.Module):
    MAP = {"0" : "c1", "1" : "c2", "2" : "c3", "3" : "c4", "4" : "c5"}
    IN_CHANNELS_LIST = {"c1" : 64, "c2" : 128, "c3" : 256, "c4" : 512, "c5" : 512}
    def __init__(
        self, 
        featmap_list,
    ):
        super().__init__()
        self._block = nn.ModuleList()
        self._featmap_list = featmap_list
        self._featmap_list = featmap_list
        self.in_channels_list = [self.IN_CHANNELS_LIST[k] for k in featmap_list]
        
        self._block.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )) 
        self._block.append(nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))        
        self._block.append(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
        self._block.append(nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
        self._block.append(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
        self.feature_layer = nn.Sequential(
            *self._block
        )
        
    def forward(
        self, 
        X: tc.Tensor
    ):
        features = OrderedDict()

        x = X
        for i, layer in enumerate(self._block):
            x = layer(x)
            if self.MAP.get(f"{i}", "-1") in self._featmap_list:
                features[self.MAP[f"{i}"]] = x
        return features

class VGG16(nn.Module):
    MAP = {"4" : "c1", "9" : "c2", "16" : "c3", "23" : "c4", "30" : "c5"}
    IN_CHANNELS_LIST = {"c1" : 64, "c2" : 128, "c3" : 256, "c4" : 512, "c5" : 512}
    
    def __init__(self, featmap_list: list[str]):
        super().__init__()
        self.features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.features.requires_grad_(False)
        self._featmap_list = featmap_list
        self.in_channels_list = [self.IN_CHANNELS_LIST[k] for k in featmap_list]
    
    def forward(self, X: tc.Tensor):
        res = OrderedDict()
        for i, layer in enumerate(self.features):
            X = layer(X)
            if(self.MAP.get(f"{i}", "-1") in self._featmap_list):
                res[self.MAP[f"{i}"]] = X
        return res
    
class BackboneMiniVGG16(nn.Module):
    def __init__(self, featmap_list: list[str] = ["c2", "c3", "c4", "c5"]):
        super().__init__()
        self.out_channels=256
        self._backbone = MiniVGG16(featmap_list=featmap_list)
        self._fpn = FeaturePyramidNetwork(in_channels_list=self._backbone.in_channels_list, out_channels=self.out_channels)
    
    def forward(self, X: tc.Tensor):
        X = self._backbone(X)
        X = self._fpn(X)
        return X

class BackboneVGG16(nn.Module):
    def __init__(self, featmap_list: list[str] = ["c2", "c3", "c4", "c5"]):
        super().__init__()
        self.out_channels=256
        self._backbone = VGG16(featmap_list=featmap_list)
        self._fpn = FeaturePyramidNetwork(in_channels_list=self._backbone.in_channels_list, out_channels=self.out_channels)
    
    def forward(self, X: tc.Tensor):
        X = self._backbone(X)
        X = self._fpn(X)
        return X