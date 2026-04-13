import torch as tc
import torch.nn as nn
import torchvision
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._linear = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10, bias=True)
        )
        self._output = nn.Softmax(dim=1)
    def convolve(
        self,
        X: tc.Tensor
    )->tc.Tensor:
        return self._conv(X)

    def forward(
        self, 
        X: tc.Tensor
    ):
        X = self._conv(X)
        X = self._linear(X)
        return X
    
    def predict_proba(
        self, 
        X: tc.Tensor
    ):
        X = self.forward(X)
        X = self._output(X)
        return X
    
    def predict(
        self, 
        X: tc.Tensor
    )->tc.Tensor:
        X = self.forward(X)
        X = self._output(X)
        return X.argmax(dim=1)
     
    def accuracy(
        self, 
        X: tc.Tensor,
        Y: tc.Tensor
    )->tc.Tensor:
        res = self.predict(X)
        res = res == Y
        res = res.float()
        return res.mean()
        

