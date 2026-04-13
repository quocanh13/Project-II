import torch as tc
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from src.number_recognition.VGG16 import VGG16
from dataset.digit_recognition.ImageDataset import ImageDataset, get_mnist
# python -m number_recognition.train_vgg16

vgg16 = VGG16()
vgg16.load_state_dict(tc.load("./number_recognition/vgg16.pth"))

def train(
    epoch: int = 0,
    lr : float = 0.00001,
):
    mix_dataset = ConcatDataset([ImageDataset(size=3000, img_size=(224, 224)), get_mnist(img_size=(224, 224), data_size=1000)])
    batches = DataLoader(
        dataset=mix_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=vgg16.parameters(), 
        lr=lr
    )
    for _ in range(epoch):
        s = datetime.now()
        for input, label in batches:
            output = vgg16(input)
            loss: tc.Tensor = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        e = datetime.now()
        print(f"Time : {e - s} -- Loss : {loss}", end="")
        tc.save(vgg16.state_dict(), "./number_recognition/vgg16.pth")
        print(" -- Saved")
        
    test_batches = DataLoader(
        dataset=ImageDataset(size=100, img_size=(224, 224)), 
        batch_size=100000,
        shuffle=False 
    )
    for input, label in test_batches: 
        # print(digit_recognition.predict_proba(input))
        print(vgg16.accuracy(input, label))
    test_batches = DataLoader(
        dataset=get_mnist(data_size=100, img_size=(224, 224)),
        batch_size=10000,
        shuffle=True
    )
    for input, label in test_batches: 
        # print(digit_recognition.predict_proba(input))
        print(vgg16.accuracy(input, label))
if __name__ == "__main__":
    train()
        
    