import numpy as np
import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader
from src.face_verification.FaceEmbedder import FaceEmbedder
from face_verification.data.dataset import WebFaceDataset
# python -m face_verification.test_face_embedder

def test_softmax(train = False):
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    face_embedder = FaceEmbedder(num_class=WebFaceDataset.NUM_ID).to(device)
    face_embedder.eval()
    face_embedder.features_layer.face_features.load_state_dict(tc.load("./src/face_verification/params/face_features.pth"), strict=False)
    face_embedder.linear_layer.load_state_dict(tc.load("./src/face_verification/params/linear.pth"))

    batches = DataLoader(
        dataset=WebFaceDataset(train=train, size=1000, random=True),
        shuffle=False,                  
        batch_size=128
    )

    softmax = nn.Softmax(dim=1)
    accuracy = tc.scalar_tensor(0.0).to(device)
    count = 0

    for X, Y in batches:
        X = X.to(device)
        Y = Y.to(device)
        features = face_embedder(X)
        res: tc.Tensor = softmax(features)
        res = tc.argmax(res, dim=1) == Y
        res = res.float()
        accuracy += tc.mean(res)
        count += 1
        
    print(accuracy / count)


def test_triplet(
    train = False,
    threshold: tuple[float, float, float] = [0.0, 1, 0.05]
):
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    face_embedder = FaceEmbedder().to(device)
    face_embedder.eval()
    face_embedder.features_layer.face_features.load_state_dict(tc.load("./src/face_verification/params/face_features.pth"), strict=False)
    face_embedder.linear_layer.load_state_dict(tc.load("./src/face_verification/params/linear.pth"))

    batches = DataLoader(
        dataset=WebFaceDataset(train=train, size=1000, random=True, loss="triplet"),
        shuffle=False,                  
        batch_size=128
    )
    start, end, step = threshold
    for thres in np.arange(start, end + step/2, step):
        count = 0
        correct = 0

        for i, (anchor, positive, negative) in enumerate(batches):
            with tc.no_grad():
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                d1 = face_embedder.distance(anchor, positive)
                d2 = face_embedder.distance(anchor, negative)
                
                correct += (d1 <= thres).sum().item()
                correct += (d2 > thres).sum().item()
                
                count += 2 * anchor.size(0)
        
        print(f"Training set = {train} -- Threshold : {thres} -- Accuracy : {(correct / count):.3f}")
    
test_triplet(train=True)
test_triplet(train=False)