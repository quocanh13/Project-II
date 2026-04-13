import os
import random
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

TRAIN_FACE_SIZE = 1082
TRAIN_NON_FACE_SIZE = 3899
TEST_FACE_SIZE = 1052
TEST_NON_FACE_SIZE = 1052

class ImageDataset(Dataset):
    def __init__(
        self, 
        img_size: tuple[int, int] = (500, 500),
        num_face: int = 1000,
        num_non_face: int = 100,
        train = True
    ):
        super().__init__()
        self._img_size = img_size
        self._num_face = num_face
        self._num_non_face = num_non_face
        self._face_path = "./face_recognition/images/train/face/"
        self._non_face_path = "./face_recognition/images/train/non_face/"
        self._num = num_face + num_non_face
        self._max_num_face = TRAIN_FACE_SIZE
        self._max_num_non_face = TRAIN_NON_FACE_SIZE
        self._transformer =  tf.Compose([
            tf.Resize(img_size),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if(not train):
            self._face_path = "./face_recognition/images/test/face/"
            self._non_face_path = "./face_recognition/images/test/non_face/"
        
    @staticmethod
    def get_path(folder: str, name: str):
        ext = ["png", "jpg", "jpeg"]
        for i in ext:
            path = f"{folder}{name}.{i}"
            if(os.path.exists(path)): return path
        return ""
     
    def __len__(self):
        return self._num
    
    def __getitem__(self, index):
        num_face = self._num_face
        transformer = self._transformer
        if(index < num_face):
            index = random.randint(0, self._max_num_face)
            path = self.get_path(self._face_path, index)
            label = 1
        else:
            index = random.randint(0, self._max_num_non_face)
            path = self.get_path(self._non_face_path, index)
            label = 0
        with Image.open(path).convert("RGB") as img:
            return transformer(img), label