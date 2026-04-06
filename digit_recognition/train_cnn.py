import torch as tc
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from digit_recognition.CNN import digit_recognition, digit_recognition_GAP, digit_recognition_new
from digit_recognition.dataset.ImageDataset import ImageDataset, mnist_dataset
# python -m digit_recognition.train_cnn

def train(
    epoch: int = 5,
    lr : float = 0.000001,
):
    image_train_dataset = ImageDataset(size=2000)
    image_test_dataset = ImageDataset(size=100)
    mix_dataset = ConcatDataset([image_train_dataset, mnist_dataset])
    batches = DataLoader(
        dataset=mix_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=digit_recognition.parameters(), 
        lr=lr
    )
    for _ in range(epoch):
        s = datetime.now()
        for input, label in batches:
            output = digit_recognition(input)
            loss: tc.Tensor = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        e = datetime.now()
        print(f"Time : {e - s} -- Loss : {loss}", end="")
        tc.save(digit_recognition.state_dict(), "./digit_recognition/params.pth")
        print(" -- Saved")
        
    test_batches = DataLoader(
        dataset=image_test_dataset, 
        batch_size=100000,
        shuffle=False 
    )
    for input, label in test_batches: 
        # print(digit_recognition.predict_proba(input))
        print(digit_recognition.accuracy(input, label))

if __name__ == "__main__":
    train()
        
    