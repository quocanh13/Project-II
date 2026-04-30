import torch as tc
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from src.face_verification.FaceEmbedder import FaceEmbedderVGG16
from src.face_verification.data.dataset import WebFaceDataset, CelebADataset

# python -m src.face_verification.train_face_embedder
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

def train_softmax(
    epoch: int = 10,
    lr : float = 0.00001,
    load_state = True
):
    face_embedder = FaceEmbedderVGG16(embedding_dim=WebFaceDataset.NUM_ID).to(device)
    if(load_state):
        face_embedder.features_layer.face_features.load_state_dict(tc.load(FaceEmbedderVGG16.PARAM.FACE_VERIFICATION.FACE_FEATURE), strict=False)
        face_embedder.linear_layer.load_state_dict(tc.load(FaceEmbedderVGG16.PARAM.FACE_VERIFICATION.LINEAR))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=[
            {"params": face_embedder.features_layer.face_features.parameters()},
            {"params": face_embedder.linear_layer.parameters()}   
        ], 
        lr=lr, weight_decay=0.001)
    batches = DataLoader(
        dataset=WebFaceDataset(loss="cross-entrpy"),
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
        print("")
        for i, (input, label) in enumerate(batches):
            input = input.to(device)
            label = label.to(device)
            print("\033[F\033[K", end="")
            print(f"{i}/{len(batches)}")
            output = face_embedder.linear(input)
            loss: tc.Tensor = criterion(output, label)
            training_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        face_embedder.eval()
        for i, (input, label) in enumerate(validation):
            input = input.to(device)
            label = label.to(device)
            print("\033[F\033[K", end="")
            print(f"{i}/{len(validation)}")
            with tc.no_grad():
                output = face_embedder.linear(input)
                validation_loss += criterion(output, label).item()
                
        e = datetime.now()
        print("\033[F\033[K", end="")
        print(f"Time : {e - s} -- Training loss : {training_loss / len(batches)} -- Validation loss : {validation_loss / len(batches)}", end="")
        tc.save(face_embedder.features_layer.face_features.state_dict(), FaceEmbedderVGG16.PARAM.FACE_VERIFICATION.FACE_FEATURE)
        tc.save(face_embedder.linear_layer.state_dict(), FaceEmbedderVGG16.PARAM.FACE_VERIFICATION.FACE_FEATURE)
        print(" -- Saved")
 
def train_triplet(
    epoch: int = 10,
    lr : float = 0.00001,
    margin: float = 0.5,
    load_state = True,
    print_ec = True
):
    face_embedder = FaceEmbedderVGG16(embedding_dim=512).to(device)
    if(load_state):
        face_embedder.features_layer.face_features.load_state_dict(tc.load(FaceEmbedderVGG16.PARAM.FACE_VERIFICATION.FACE_FEATURE))
        face_embedder.linear_layer.load_state_dict(tc.load(FaceEmbedderVGG16.PARAM.FACE_VERIFICATION.LINEAR))
    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(
        params=[
            {"params": face_embedder.features_layer.face_features.parameters()},
            {"params": face_embedder.linear_layer.parameters()}
        ], 
        lr=lr, weight_decay=0.0001)
    batches = DataLoader(
        # dataset=WebFaceDataset(loss="triplet"),
        dataset=CelebADataset(),
        shuffle=True,
        batch_size=64,
        num_workers=2
    )
    validation = DataLoader(
        dataset=WebFaceDataset(train=False, loss="triplet"),
        batch_size=64,
        num_workers=1
    )
    for _ in range(epoch):
        training_loss = 0.0
        validation_loss = 0.0
        s = datetime.now()
        face_embedder.train()
        if(print_ec): print("")
        for i, (anchor, positive, negative) in enumerate(batches):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            if(print_ec):
                print("\033[F\033[K", end="")
                print(f"{i}/{len(batches)}")
                
            anchor_vector = face_embedder(anchor)
            positive_vector = face_embedder(positive)
            negative_vector = face_embedder(negative)
            
            loss: tc.Tensor = criterion(anchor_vector, positive_vector, negative_vector)
            training_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        face_embedder.eval()
        for i, (anchor, positive, negative) in enumerate(validation):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            if(print_ec):
                print("\033[F\033[K", end="")
                print(f"{i}/{len(validation)}")
            with tc.no_grad():
                anchor_vector = face_embedder(anchor)
                positive_vector = face_embedder(positive)
                negative_vector = face_embedder(negative)
                validation_loss += criterion(anchor_vector, positive_vector, negative_vector).item()
                
        e = datetime.now()
        if(print_ec): print("\033[F\033[K", end="")
        print(f"Time : {e - s} -- Training loss : {training_loss / len(batches)} -- Validation loss : {validation_loss / len(batches)}", end="")
        tc.save(face_embedder.features_layer.face_features.state_dict(), "./src/face_verification/params/face_features.pth")
        tc.save(face_embedder.linear_layer.state_dict(), "./src/face_verification/params/linear.pth")
        print(" -- Saved")
       
if __name__ == "__main__":
    # train_softmax(epoch=3, lr=0.001, load_state=False)
    train_triplet(epoch=10, lr=0.001, load_state=True, print_ec=False, margin=0.5)
    