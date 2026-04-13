import torch as tc
import torch.optim as optim
from typing import Callable
from torch import nn
from torch.utils.data import Dataset

class CNN(nn.Module):
    def __init__(
        self,
        conv_layers: nn.Sequential,
        fcn_layer: nn.Linear,
        output_func: nn.Module | Callable[[tc.Tensor], tc.Tensor]
    ):
        super().__init__()
        self._conv_layers = conv_layers
        self._fcn_layer = fcn_layer
        self._output_func = output_func
        
    def forward(
            self,
            X: tc.Tensor
        ): 
        X = self._conv_layers(X)
        # print(X.shape)
        # exit()
        X = self._fcn_layer(X)
        return X
    
    def predict_proba(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:
        X = self._conv_layers(X)
        X = self._fcn_layer(X)
        X = self._output_func(X)
        return X
    
    def predict(
        self,
        X: tc.Tensor
    ) -> tc.Tensor:
        X = self._conv_layers(X)
        X = self._fcn_layer(X)
        X = self._output_func(X)
        return X.argmax(dim=1)
    
    def predict_all(
        self,
        X: tc.Tensor
    ) -> tuple[int, float]:
        X = self._conv_layers(X)
        X = self._fcn_layer(X)
        X = self._output_func(X)
        index = X.argmax(dim=1)
        return index, X[0][index]
    
    def accuracy(
        self, 
        X: tc.Tensor,
        Y: tc.Tensor
    ) -> tc.Tensor:
        res = self.predict(X)
        res = res == Y
        res = res.float()
        return res.mean()

digit_recognition = CNN(
    conv_layers= nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=1, padding=2), 
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(1)
        ),
    fcn_layer=nn.Linear(in_features=1568, out_features=10),
    output_func=nn.Softmax(dim=1)  
)

digit_recognition_new = CNN(
    conv_layers= nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=1, padding=2), 
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(1)
        ),
    fcn_layer=nn.Linear(in_features=1568, out_features=10),
    output_func=nn.Softmax(dim=1)  
)

digit_recognition_GAP = CNN(
    conv_layers= nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ),
    fcn_layer=nn.Linear(in_features=256, out_features=10),
    output_func=nn.Softmax(dim=1)  
)

digit_recognition.load_state_dict(tc.load("./digit_recognition/params.pth"))
digit_recognition_new.load_state_dict(tc.load("./digit_recognition/params_new.pth"))
digit_recognition_GAP.load_state_dict(tc.load("./digit_recognition/params_gap.pth"))