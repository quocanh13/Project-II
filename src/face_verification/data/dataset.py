# WebFaceDataset.py
import json
import random
import torch as tc
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset
from typing import Literal



class WebFaceDataset(Dataset):
    PATH = "./dataset/webface_112x112"
    TRAIN_JSON_PATH = "./src/face_verification/data/json/train_webface.json"
    TRAIN_INDEX_PATH = "./src/face_verification/data/json/train_webface_index.json"
    TEST_JSON_PATH = "./src/face_verification/data/json/test_webface.json"
    TEST_INDEX_PATH = "./src/face_verification/data/json/test_webface_index.json"
    with open("./src/face_verification/data/json/webface.json") as file:
        info = json.load(file)
        
    NUM_ID = info["num_id"]
    
    def __init__(
        self,
        image_size = 150,
        image_crop = 112,
        size: int = None,
        train = True,
        random = False,
        loss: Literal["cross-entrpy", "triplet"] = "cross-entropy"
    ):
        super().__init__()
        self._random = random
        self._size = size
        self._img_size = image_size
        self._loss: Literal["cross-entrpy", "triplet"] = loss
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
                self._index = json.load(file)
            with open(self.TRAIN_INDEX_PATH, "r") as file:
                self._num_index = json.load(file)
        else:
            with open(self.TEST_JSON_PATH, "r") as file:
                self._index = json.load(file)
            with open(self.TEST_INDEX_PATH, "r") as file:
                self._num_index = json.load(file)
    
        size = size if size is not None else len(self._index)
        self._size = min(len(self._index), size)
                
    def __len__(self):
        return self._size
    
    def get_img(self, name: str, id: int):
        with Image.open(WebFaceDataset.PATH + f"/id_{id:04}/{name}").convert("RGB") as img:
            img = self.transformer(img)
            return img
    
    def _one_img(self, index: int):
        if(self._random):
            index = random.randint(0, len(self._index) - 1)
        # index = random.randint(0, self._size - 1)
        img_data = self._index[index]
        img_name = img_data["name"]
        id = img_data["id"]

        img = self.get_img(img_name, id)
        return img, id

    def _triplet(self, index: int):
        anchor_index = index
        if(self._random):
            anchor_index = random.randint(0, len(self._index) - 1)
        anchor_data = self._index[anchor_index]
        anchor_name = anchor_data["name"]
        anchor_id = anchor_data["id"]
        
        positive_index = anchor_index
        [positive_start, positive_end] = self._num_index[anchor_id]
        if(positive_start < positive_end):
            while(positive_index == anchor_index):
                positive_index = random.randint(positive_start, positive_end)
        
        positive_data = self._index[positive_index]
        positive_name = positive_data["name"]
        
        negative_id = anchor_id
        while(negative_id == anchor_id):
            negative_id = random.randint(0, self.NUM_ID - 1)
        
        negative_index = random.randint(self._num_index[negative_id][0], self._num_index[negative_id][1])
        negative_data = self._index[negative_index]
        negative_name = negative_data["name"]
        
        anchor = self.get_img(anchor_name, anchor_id)
        positive = self.get_img(positive_name, anchor_id)
        negative = self.get_img(negative_name, negative_id)
        
        return anchor, positive, negative

    def __getitem__(self, index) -> tuple[tc.Tensor, int]:
        if(self._loss == "triplet"):
            return self._triplet(index)
        else:
            return self._one_img(index)
        
class CelebADataset(Dataset):
    NUM_ID = 10177
    IDENTITY_PATH = "./src/face_verification/data/json/celebA_identity.json"
    def __init__(
        self,
        num_id = 1000,
        size = 10000
    ):
        self._paths: list[list[str]]
        self._size = size
        self._num_id = min(num_id, self.NUM_ID)
        with open(self.IDENTITY_PATH) as file:
            self._paths = json.load(file)
        self.transformer = tf.Compose([
            tf.ToTensor()
        ])
        
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        anchor = positive = negative = random.randint(0, self._num_id - 1)
        while(negative == anchor):
            negative = random.randint(0, self._num_id - 1)
            
        anchor_index = positive_index = random.randint(0, len(self._paths[anchor]) - 1)
        while(anchor_index == positive_index):
            positive_index = random.randint(0, len(self._paths[positive]) - 1)
        negative_index = random.randint(0, len(self._paths[negative]) - 1)
        
        with Image.open(self._paths[anchor][anchor_index]) as anchor_img:
            anchor_img = self.transformer(anchor_img)
            
        with Image.open(self._paths[positive][positive_index]) as positive_img:
            positive_img = self.transformer(positive_img)
            
        with Image.open(self._paths[negative][negative_index]) as negative_img:
            negative_img = self.transformer(negative_img)
            
        return anchor_img, positive_img, negative_img
        