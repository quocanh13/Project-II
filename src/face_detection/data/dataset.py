import json
import torch as tc
import torchvision.transforms as tf
from random import randint
from PIL import Image
from torch.utils.data import Dataset

class WiderFaceDataset(Dataset):
    IMG_PATH = "./dataset/WIDER_train/images"
    INFO_PATH = "./src/face_detection/data/wider_face_info.json"
    
    def __init__(self, size: int = None, random = False):
        with open(self.INFO_PATH, "r") as info:
            self._info = json.load(info)
            
        if size is None:
            self._size = len(self._info)
        else:
            self._size = min(size, len(self._info))
            
        self._random = random
        self._tranform = tf.Compose([
            tf.ToTensor()
        ])
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        if(self._random):
            index = randint(0, len(self._info) - 1)
            
        info = self._info[index]
        img_path = info["path"]
        bbox = info["bbox"]
        if len(bbox) == 0:
            return None
        bbox = tc.tensor(bbox, dtype=tc.float32)
        label = tc.ones((bbox.shape[0], ), dtype=tc.int64)
        
        with Image.open(img_path).convert("RGB") as img:
            img = self._tranform(img)
        
        return img, {"boxes" : bbox, "labels" : label}
    
class CelebADataset(Dataset):
    IMG_PATH = "./dataset/CelebA/images"
    INFO_PATH = "./src/face_detection/data/celebA_info.json"
    
    def __init__(self, size: int = None, random = False):
        with open(self.INFO_PATH, "r") as info:
            self._info = json.load(info)
            
        if size is None:
            self._size = len(self._info)
        else:
            self._size = min(size, len(self._info))
            
        self._random = random
        self._tranform = tf.Compose([
            tf.ToTensor()
        ])
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        if(self._random):
            index = randint(0, len(self._info) - 1)
            
        info = self._info[index]
        img_path = info["path"]
        bbox = info["bbox"]
        bbox = tc.tensor(bbox, dtype=tc.float32)
        label = tc.ones((bbox.shape[0], ), dtype=tc.int64)
        
        with Image.open(img_path).convert("RGB") as img:
            img = self._tranform(img)
        
        return img, {"boxes" : bbox, "labels" : label}