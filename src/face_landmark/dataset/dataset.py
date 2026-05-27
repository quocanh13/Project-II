import json
import torch as tc
from PIL import Image
from torchvision import transforms as tf
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    LANDMARK_PATH = "./src/face_landmark/dataset/json/landmark.json"
    SIZE = 100000
    def __init__(
        self,
        start = 0,
        end = 1000000
    ):
        super().__init__()
        
        if(start > end):
            start = 0
        end = min(end, self.SIZE - 1)
        
        with open(self.LANDMARK_PATH) as file:
            infos = json.load(file)
        
        transformer = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.start = start
        self.end = end
        self.infos = infos
        self.transformer = transformer
    def __len__(self):
        return self.end - self.start + 1
    
    def __getitem__(self, index) -> tuple[tc.Tensor, tc.Tensor]:
        info = self.infos[self.start + index]
        path = info["path"]
        landmark = info["landmark"]
        
        with Image.open(path).convert("RGB") as image:
            w, h = image.size
            image = self.transformer(image)
            
        landmark = tc.tensor(landmark, dtype=tc.float32)
        landmark[[0, 2, 4, 6, 8]] /= w
        landmark[[1, 3, 5, 7, 9]] /= h
        
        return image, landmark
    

        