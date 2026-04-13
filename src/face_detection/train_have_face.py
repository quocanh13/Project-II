import torch as tc
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from dataset.face_detection.ImageDataset import ImageDataset
from src.face_detection.HaveFace import HaveFace

# python -m face_detection.train_have_face

have_face = HaveFace()
have_face._linear.load_state_dict(tc.load("./face_detection/have_face_linear.pth"))
have_face._features.face_features.load_state_dict(tc.load("./face_detection/have_face_face_features.pth"))

def train(
    epoch: int = 3,
    lr : float = 0.0001,
):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=[
            {'params': have_face._features.face_features.parameters()},
            {'params': have_face._linear.parameters()} # Ví dụ: cho lớp Linear học nhanh hơn
    ], lr=lr)
    batches = DataLoader(
        dataset=ImageDataset(num_face=100, num_non_face=500),
        shuffle=True,
        batch_size=32
    )
    for _ in range(epoch):
        s = datetime.now()
        for input, label in batches:
            output = have_face(input)
            loss: tc.Tensor = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        e = datetime.now()
        print(f"Time : {e - s} -- Loss : {loss}", end="")
        tc.save(have_face._linear.state_dict(), "./face_detection/have_face_linear.pth")
        tc.save(have_face._features.face_features.state_dict(), "./face_detection/have_face_face_features.pth")
        print(" -- Saved")
        
if __name__ == "__main__":
    train()
        
    