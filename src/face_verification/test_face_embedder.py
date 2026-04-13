import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader
from src.face_verification.FaceEmbedder import FaceEmbedder
from src.face_verification.data.WebFaceDataset import WebFaceDataset
# python -m face_verification.test_face_embedder

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
face_embedder = FaceEmbedder(num_class=WebFaceDataset.NUM_ID).to(device)
face_embedder.eval()
face_embedder.features.face_features.load_state_dict(tc.load("./src/face_verification/params/face_features.pth"), strict=False)
face_embedder.linear.load_state_dict(tc.load("./src/face_verification/params/linear.pth"))

def test(train = False):
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
    
test(True)
test()