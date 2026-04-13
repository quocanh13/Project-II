import torch as tc
import numpy as np
from PIL import Image
class ImageTool:
    @staticmethod
    def alpha_to_gray_encoder(img: Image.Image)->tc.Tensor:
        """
        Args:
            img (Image.Image): _description_

        Returns:
            tc.Tensor: batch [1, 1, H, W]
        """
        img = np.array(img)
        img = img[:, :, 3] / 255
        img = tc.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        return img
      
    @staticmethod
    def RGB_to_gray_encoder(img: Image.Image)->tc.Tensor:
        img = np.array(img)
        img = np.mean(img, axis=2) / 255
        img = tc.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        return img
    
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
    def rgba_channel_encoder(img: Image.Image) -> tc.Tensor:
        res = np.array(img)
        res = tc.from_numpy(res)
        res = res.permute(2, 0, 1) / 255
        return res
    
    @staticmethod 
    def alpha_channel_encoder(img: Image.Image) -> tc.Tensor:
        res = np.array(img)
        res = tc.from_numpy(res)
        res = res.permute(2, 0, 1) / 255
        return res[3:4, :, :]
    
# with Image.open("./digit_recognition/dataset/images/1-0.png") as img:
#     ImageTool.gray_encoder(img).show()
    