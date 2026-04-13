import torch as tc
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from src.face_verification.FaceEmbedder import FaceEmbedder
from src.face_verification.data.WebFaceDataset import WebFaceDataset

# python -m src.face_verification.train_face_embedder
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
face_embedder = FaceEmbedder(num_class=WebFaceDataset.NUM_ID).to(device)
face_embedder.features.face_features.load_state_dict(tc.load("./src/face_verification/params/face_features.pth"), strict=False)
face_embedder.linear.load_state_dict(tc.load("./src/face_verification/params/linear.pth"))

def train(
    epoch: int = 10,
    lr : float = 0.00001,
):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=[
            {"params": face_embedder.features.face_features.parameters()},
            {"params": face_embedder.linear.parameters()}   
        ], 
        lr=lr, weight_decay=0.001)
    batches = DataLoader(
        dataset=WebFaceDataset(),
        shuffle=True,
        batch_size=64,
        num_workers=2
    )
    validation = DataLoader(
        dataset=WebFaceDataset(train=False),
        batch_size=64,
        num_workers=1
    )
    for _ in range(epoch):
        training_loss = 0.0
        validation_loss = 0.0
        s = datetime.now()
        face_embedder.train()
        # print("")
        for i, (input, label) in enumerate(batches):
            input = input.to(device)
            label = label.to(device)
            # print("\033[F\033[K", end="")
            # print(f"{i}/{len(batches)}")
            output = face_embedder(input)
            loss: tc.Tensor = criterion(output, label)
            training_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        face_embedder.eval()
        for i, (input, label) in enumerate(validation):
            input = input.to(device)
            label = label.to(device)
            # print("\033[F\033[K", end="")
            # print(f"{i}/{len(validation)}")
            with tc.no_grad():
                output = face_embedder(input)
                validation_loss += criterion(output, label).item()
                
        e = datetime.now()
        # print("\033[F\033[K", end="")
        print(f"Time : {e - s} -- Training loss : {training_loss / len(batches)} -- Validation loss : {validation_loss / len(batches)}", end="")
        tc.save(face_embedder.features.face_features.state_dict(), "./src/face_verification/params/face_features.pth")
        tc.save(face_embedder.linear.state_dict(), "./src/face_verification/params/linear.pth")
        print(" -- Saved")
        
if __name__ == "__main__":
    train()
        
    