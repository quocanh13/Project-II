import os
import json
import random
import torch as tc
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

class WebFaceDataset(Dataset):
    PATH = "./dataset/webface_112x112"
    TRAIN_JSON_PATH = "./src/face_verification/data/train_webface.json"
    TEST_JSON_PATH = "./src/face_verification/data/test_webface.json"
    NUM_ID = 100
    
    def __init__(
        self,
        image_size = (112, 112),
        size: int = None,
        train = True,
        random = False
    ):
        super().__init__()
        self._random = random
        self._size = size
        self._img_size = image_size
        
        self.transformer = tf.Compose([
            tf.Resize(image_size),
            tf.RandomHorizontalFlip(p=0.5),
            tf.RandomRotation(10),
            tf.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            tf.ColorJitter(brightness=0.2, contrast=0.2),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if(train):
            with open(self.TRAIN_JSON_PATH, "r") as file:
                self._img_list = json.load(file)
        else:
            with open(self.TEST_JSON_PATH, "r") as file:
                self._img_list = json.load(file)
    
        size = size if size is not None else len(self._img_list)
        self._size = min(len(self._img_list), size)
                
    def __len__(self):
        return self._size
    def __getitem__(self, index) -> tuple[tc.Tensor, int]:
        if(self._random):
            index = random.randint(0, len(self._img_list) - 1)
        # index = random.randint(0, self._size - 1)
        img_data = self._img_list[index]
        img_name = img_data["name"]
        id = img_data["id"]

        with Image.open(self.PATH + f"/id_{id:04}/{img_name}") as img:
            img = self.transformer(img)
            return img, id
        
        
