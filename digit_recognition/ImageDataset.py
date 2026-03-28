import torch as tc
from typing import Callable
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable

class ImageDataset(Dataset):
    @staticmethod
    def white_black_encoder(img: Image.Image) -> list[list[float]]:
        res: list[list] = []
        for y in range(img.height):
            res.append([])
            for x in range(img.width):
                r, g, b, a = img.getpixel((x, y))
                # print(r, g, b, a)
                if(r == g == b == a == 0):
                    res[y].append(0)
                else:
                    res[y].append(1.0)
            # print(res[y])
        return res
                    
    @staticmethod 
    def unique_encoder(img: Image.Image) -> list[list[float]]:
        res: list[list] = []
        for y in range(img.height):
            res.append([])
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                res[y].append(r/256**3 + g/256**2 + b/256)    

    @staticmethod
    def img_to_tensor(
        img_path: str,
        img_encoder: Callable[[Image.Image], list[list[float]]]
    ) -> tc.Tensor:
        with Image.open(img_path) as img:
            encoded_img = img_encoder(img)
            return tc.tensor(encoded_img, dtype=float)

    def __init__(
        self, 
        img_data: list[tuple[str, float]], 
        img_encoder: Callable[[Image.Image], list[list[float]]],
        flatten = False
    ):
        """
        Args:
            img_data (list[tuple[str, float]]): list[(image_path, label)]
        """
        super().__init__()
        self._data: list[tuple[str, float]] = []
        self._img_encoder = img_encoder
        self._data = img_data.copy()
        self._flatten = flatten
        
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index) -> tuple[tc.Tensor, float]:
        img_path, label = self._data[index]
        img = ImageDataset.img_to_tensor(img_path, self._img_encoder)
        if(self._flatten):
            img = img.flatten()
        return img, label
    
