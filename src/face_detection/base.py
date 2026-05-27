import torch as tc
import torch.nn as nn
from PIL import Image
from abc import ABC, abstractmethod

class FaceDetection(nn.Module, ABC):    
    @abstractmethod
    def load_state(self, device: str | None = None):
        pass
    @abstractmethod
    def save_state(self):
        pass
    
    @abstractmethod
    def detect(
        self,
        img : Image.Image | tc.Tensor,
        num_bbox = -1,
    ) -> list[tuple[list[int], float]]:
        pass
    
