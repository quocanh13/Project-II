import torch as tc
import torch.nn as nn
from typing import Callable
from datetime import datetime
from PIL import Image
from src.digit_recognition_fnn.FNN import *
from dataset.digit_recognition_fnn.ImageDataset import ImageDataset
from dataset.digit_recognition_fnn.data import get_img_data, get_image_path
# python -m digit_recognition.CNN

"""
    O = (I + 2*P - D*(K - 1) - 1) / S + 1   
"""

class PoolingingFunction:
    @staticmethod
    def gap(input: tc.Tensor) -> tc.Tensor:
        res = tc.mean(input, dim=[2, 3])
        return res
    
    @staticmethod
    def flatten(input: tc.Tensor) -> tc.Tensor:
        res = input.flatten(start_dim=1, end_dim=3)
        return res
class CNN:
    def __init__(
        self, 
        conv_configs : list[tuple[int, int, int, int, int, bool]],
        fc_output_size: int,
        pooling_func: Callable[[tc.Tensor], tc.Tensor] = PoolingingFunction.gap
    ):
        """
        Args:
            conv_configs (list[tuple[int, int, int, int, int, bool]]): list[in_channels, out_channels, kernel_size, stride, padding, bias]
        """
        self._convs: list[nn.Conv2d] = []
        self._fc: NeuralNetwork
        self._pooling_func = pooling_func
        
        i, o, ks, st, p, b  = conv_configs[0]
        
        for i, o, ks, st, p, b  in conv_configs:
            self._convs.append(nn.Conv2d(in_channels=i, out_channels=o, kernel_size=ks, stride=st, padding=p, bias=b))
        # conv_output_size = conv_configs[-1][1]
        conv_output_size = 80000
        self._fc = NeuralNetwork(
            layers = [
                (128, conv_output_size, ActivationFunction.relu),
                (64, 128, ActivationFunction.relu),
                (fc_output_size, 64, ActivationFunction.copy)
            ],
            loss_func=LossFunction.softmax_cross_entropy,
            output_func=ActivationFunction.softmax
        )
        
    def forward(
        self, 
        input: tc.Tensor,
        label: tc.Tensor    
    ):
        input = input.float()
        for conv in self._convs:
            input = ActivationFunction.leaky_relu(conv(input))
        
        input = self._pooling_func(input)
        # print(input.shape)
        # exit()
        loss = self._fc.forward(input, label)
        return loss
        
    def _optimize(self, alpha : float = 0.005): 
        self._fc._optimize(alpha)
        with tc.no_grad():
            for conv in self._convs:
                # print(conv.weight.grad)
                # exit()
                conv.weight.sub_(conv.weight.grad * alpha)
                conv.bias.sub_(conv.bias.grad * alpha)
                conv.weight.grad.zero_()
                conv.bias.grad.zero_()
        
    def train(
        self,
        train_data: Dataset,
        epoch: int = 10,
        alpha: float = 0.001,
    ):
        for _ in range(epoch):
            start = datetime.now()
            batches = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=4)
            for input, label in batches:
                loss = self.forward(input, label)
                loss.backward()
                self._optimize(alpha)
            end = datetime.now()
            print(f"Time : {end - start} -- Loss : {loss}")
    
    def predict(self, input: tc.Tensor) -> tc.Tensor:
        input = input.float()
        for conv in self._convs:
            input = tc.relu(conv(input))
        
        input = self._pooling_func(input)
        return self._fc.predict(input)

    
def main():
    cnn = CNN(
        #in_channels, out_channels, kernel_size, stride, padding, bias
        conv_configs=[          
            (1, 8, 3, 1, 1, True), 
        ],
        fc_output_size=10,
        pooling_func=PoolingingFunction.flatten
    )

    img_dataset = ImageDataset(
        # img_data=[("./digit_recognition/images/images_0/0_0.png", 0), ("./digit_recognition/images/images_1/1_0.png", 1)],
        img_data=get_img_data(), 
        img_encoder=ImageDataset.alpha_channel_encoder, 
        flatten=False,
        rotate_info=(-10, 10, 5),
        translate_info=(0, 32, 2)
    )
    
    cnn.train(
        train_data=img_dataset,
        epoch=5,
        alpha=0.01
    )
    
    for i in range(0, 10):
        for j in range(10, 16):
            with Image.open(get_image_path(f"{i}", j)) as img:
                res = cnn.predict(ImageDataset.alpha_channel_encoder(img).unsqueeze(0)).flatten().tolist()
                max_prob = max(res)
                predict = res.index(max_prob)
                print(f"Label : {i} -- Predict : {predict} - {max_prob} -- {i == predict}" )
                # print(f"Prob : {res}")
    
if __name__ == "__main__":
    main()