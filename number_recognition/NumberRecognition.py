import torch as tc
import torch.nn as nn
import cv2
import numpy as np
from digit_recognition.CNN import digit_recognition
from digit_recognition.dataset.ImageDataset import ImageDataset
from PIL import Image
# python -m number_recognition.NumberRecognition

class NumberRecognition(nn.Module):
    @staticmethod
    def resize(
        img: np.ndarray,
        w: int,
        h: int
    ):

        # Tính scale giữ tỉ lệ
        scale = 50 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize
        img_resized = cv2.resize(img, (new_w, new_h))

        # Tạo canvas 50x50 (nền đen)
        canvas = np.zeros((50, 50), dtype=np.uint8)

        # Tính offset để căn giữa
        y_offset = (50 - new_h) // 2
        x_offset = (50 - new_w) // 2

        # Đặt ảnh vào giữa
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

        return canvas
    
    @staticmethod
    def selective_search(img: np.ndarray) -> list[tc.Tensor]:
        """
        Args:
            img (np.ndarray): RGB image
        """
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        regions = ss.process()
        regions = sorted(regions, key= lambda r: r[0])
        res:list[tc.Tensor] = []
        for _, (x, y, w, h) in enumerate(regions[:100]): 
            region = img[y:y+h, x:x+w]
            region = np.mean(region, axis=2)
            region = NumberRecognition.resize(region, w, h)
            cv2.imshow("",region)
            cv2.waitKey(0)
            res.append(tc.from_numpy(region).float().unsqueeze(0).unsqueeze(0))
        return res
            
    def forward(
        self,
        img: np.ndarray
    ) -> tc.Tensor:
        regions = self.selective_search(img)
        res = 0
        for region in regions:
            number, proba = digit_recognition.predict_all(region)
            print(number, proba)
    
number_recognition = NumberRecognition()

with Image.open("./number_recognition/images/123-0.png") as img:
    img = np.array(img)
    img = img[:, :, 0:3]
    number_recognition.forward(img)