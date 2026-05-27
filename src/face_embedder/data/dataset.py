# WebFaceDataset.py
import json
import random
import torch as tc
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop
from typing import Literal
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
class WebFaceDataset(Dataset):
    PATH = "./dataset/webface_112x112"
    TRAIN_JSON_PATH = "./src/face_embedder/data/json/train_webface.json"
    TRAIN_INDEX_PATH = "./src/face_embedder/data/json/train_webface_index.json"
    TEST_JSON_PATH = "./src/face_embedder/data/json/test_webface.json"
    TEST_INDEX_PATH = "./src/face_embedder/data/json/test_webface_index.json"
    
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
    IDENTITY_PATH = "./src/face_embedder/data/json/celebA_identity.json"
    IDENTITY_CROP_PATH = "./src/face_embedder/data/json/celebA_identity_crop.json"
    
    def __init__(
        self,
        start_id = 0,
        end_id = 8000,
        size = 5000,
        sampler : Literal["triple", "id"] = "triple",
        detect_face = True,
        transformer : Literal["train", "test"] = "train"
    ):
        self._infos: list[list[str]]
        self._size = size
        self._start_id = start_id
        self._end_id = min(end_id, self.NUM_ID)
        self._sampler = sampler
        self._detect_face = detect_face
        self._transformer = transformer
        with open(self.IDENTITY_CROP_PATH) as file:
            self._infos = json.load(file)
        self.transformers: dict[str, tf.Compose] = {
            "train" : tf.Compose([
                tf.Resize(224),
                tf.CenterCrop(224),
                tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "test" : tf.Compose([
                tf.Resize(224),
                tf.CenterCrop(224),
                tf.ToTensor(),
                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }
        
    def __len__(self):
        return self._size      
    
    def get_one(self, id: int, index: int) -> tc.Tensor:
        info = self._infos[id][index]
        path = info["path"]
        bbox = info["bbox"]
        
        with Image.open(path).convert("RGB") as img:
            if(self._detect_face):
                img = crop(img, bbox[0], bbox[1], bbox[2], bbox[3])
            return self.transformers[self._transformer](img)
    
    def get_by_id(self, id):
        index = random.randint(0, len(self._infos[id]) - 1)
        return self.get_one(id, index), id
    
    def get_triple(self, index):
        anchor = positive = negative = random.randint(self._start_id, self._end_id - 1)
        while(negative == anchor):
            negative = random.randint(self._start_id, self._end_id - 1)
            
        anchor_index = positive_index = random.randint(0, len(self._infos[anchor]) - 1)
        if(len(self._infos[anchor]) > 1):
            while(anchor_index == positive_index):
                positive_index = random.randint(0, len(self._infos[positive]) - 1)
        negative_index = random.randint(0, len(self._infos[negative]) - 1)
        
        anchor_img = self.get_one(anchor, anchor_index)
        positive_img = self.get_one(positive, positive_index)
        negative_img = self.get_one(negative, negative_index)
        
        return anchor_img, positive_img, negative_img

    def __getitem__(self, index):
        if(self._sampler == "triple"):
            return self.get_triple(index)
        else:
            return self.get_by_id(index)
        
class CelebAPKSampler:
    NUM_ID = 10177
    def __init__(self, batch_num = 64, num_id = 9000, P = 64, K = 4):
        self.P = P
        self.K = K
        self.batch_num = batch_num
        self.num_id = min(self.NUM_ID, num_id)
        
    def __iter__(self):
        for i in range(self.batch_num):
            batch = []
            ids = random.sample(range(self.num_id), self.P)
            for id in ids:
                batch.extend([id] * self.K)
            yield batch
            
    def __len__(self):
        return self.batch_num