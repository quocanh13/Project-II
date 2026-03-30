import torch as tc
import numpy as np
from enum import IntEnum
from math import floor
from typing import Callable
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable
# python -m digit_recognition.images.ImageDataset

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class ImageDataset(Dataset):   
    @staticmethod
    def white_black_encoder(img: Image.Image) -> tc.Tensor:
        res: list[list] = []
        for y in range(img.height):
            res.append([])
            for x in range(img.width):
                r, g, b, a = img.getpixel((x, y))
                # print(r, g, b, a)
                if(r == g == b == a == 0):
                    res[y].append(0.0)
                else:
                    res[y].append(1.0)
            # print(res[y])
        return tc.tensor(res)
              
    @staticmethod 
    def white_black_numpy_encoder(img : Image.Image) -> tc.Tensor:
        img = np.array(img)          
        img = np.all(img == 0, axis=-1)
        res = np.where(img, 0.0, 1.0)
        return tc.tensor(res)

    @staticmethod 
    def unique_encoder(img: Image.Image) -> tc.Tensor:
        res: list[list] = []
        for y in range(img.height):
            res.append([])
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                res[y].append(r/256**3 + g/256**2 + b/256)    
        return tc.tensor(res)

    @staticmethod
    def channel_encoder(img: Image.Image) -> tc.Tensor:
        res = np.array(img)
        res = tc.from_numpy(res)
        res = res.permute(2, 0, 1)
        return res
    
    def __init__(
        self, 
        img_data: list[tuple[str, float]], 
        img_encoder: Callable[[Image.Image], tc.Tensor],
        flatten = False,
        rotate_info : tuple[int, int, int] = (-30, 30, 10),
        translate_info : tuple[int, int, int] = (0, 30, 10)
    ):
        """
        Args:
            img_data (list[tuple[str, float]]): lisf[(image_path, label)]
            img_encoder (Callable[[Image.Image], list[list[float]]]): encoder function to encode image's pixels to float
            flatten (bool, optional): Flatten image or not
            rotate (tuple[int, int, int], optional): (start, end, step) (degree)
            translate (tuple[int, int, int], optional): (start, end, step) translates to four directions
        """
        super().__init__()
        self._data: list[tuple[str, float]] = []
        self._img_encoder = img_encoder
        self._data = img_data.copy()
        self._flatten = flatten

        augment_map: list[tuple[int, Direction, int]] = []
        trs, tre, trst = rotate_info
        ros, roe, rost = translate_info
        for ro in range(ros, roe + 1, rost):
            for dir in Direction:
                for tr in range(trs, tre + 1, trst):
                    augment_map.append((ro, dir, tr))
                    
        self._augment_map = augment_map
        self._augment_size = len(augment_map)
    
    @staticmethod
    def translate(img: Image.Image, dir : Direction, tr : int) -> Image.Image:
        """
        Args:
            img (Image): 
            dir (Direction): 
            st (int): The number of pixels translated
        """
        x = 0
        y = 0
        w, h = img.size
        if(dir == Direction.UP):   y = tr*h/100
        elif(dir == Direction.RIGHT): x = -tr*w/100
        elif(dir == Direction.DOWN): y = -tr*h/100
        elif(dir == Direction.LEFT): x = tr*w/100
        else: raise Exception("dir must be Direction enum")
        trans = (1, 0, x, 0, 1, y)
        return img.transform(img.size, Image.Transform.AFFINE, trans)

    def augment(self, img : Image.Image, augment_index: int) -> Image.Image:
        ro, dir, tr = self._augment_map[augment_index]
        img = img.rotate(ro, expand=False, fillcolor="white")
        img = self.translate(img, dir, tr)
        return img
        
    def __len__(self) -> int:
        return len(self._data) * self._augment_size
    
    def __getitem__(self, index) -> tuple[tc.Tensor, float]:
        augment_size = self._augment_size
        img_index = floor(index / augment_size)
        augment_index = index - img_index*augment_size
        img_path, label = self._data[img_index]
        
        with Image.open(img_path) as img:
            img = self.augment(img, augment_index)
            img = self._img_encoder(img)
            if(self._flatten):
                img = img.flatten()
            return img, label
    

