import cv2
import random
import numpy as np
import torch as tc
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

class ImageDataset(Dataset):
    FONTS = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        ]
    
    @staticmethod
    def rand_img(
        img_size: tuple[int, int] = (50, 50),
        digit = random.randint(0, 9)
    ) -> tc.Tensor:
        img = np.zeros(img_size, dtype=np.uint8)
        text = str(digit)
        font_face = random.choice(ImageDataset.FONTS)
        img_w, img_h = img_size
        while(1):
            font_scale = random.uniform(0.4, 1.5)
            thickness = random.randint(1, 2)
            (w, h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
            
            if(w < img_w and h < img_h): 
                break

        x = random.randint(0, img_w - w)
        y = random.randint(h, img_h - baseline)
        cv2.putText(
            img, text, (x, y), font_face, font_scale, 255, thickness
        )
        
        # (w, h), _ = cv2.getTextSize(text, fontFace=font_face, fontScale=1, thickness=1)
        # cv2.putText(
        #     img, text, (0, h), color=255, fontFace=font_face, fontScale=1, thickness=1
        # )
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        
        return tc.from_numpy(img).float().unsqueeze(0)/255, digit
    
    def __init__(
        self, 
        size: int,
        img_size : tuple[int, int] = (50, 50),
    ):
        self._size = size
        self._img_size = img_size
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        return self.rand_img(self._img_size)

mnist_transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.5, 1.3)),
    transforms.ToTensor(),
])

mnist_dataset = datasets.MNIST(
    root="./digit_recognition/dataset",
    train=True,  
    transform=mnist_transform
)
indices = list(range(50000))
random.shuffle(indices)
mnist_dataset = Subset(mnist_dataset, indices[0:2000])

