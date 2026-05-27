import argparse
import numpy as np
import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader
from src.face_embedder.model import *
from src.face_embedder.data.dataset import *
# python -m src.face_embedder.test_face_embedder
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="resnet50")
parser.add_argument("-s", "--size", type=int, default=3000)
parser.add_argument("-si", "--start_id", type=int, default=9001)
parser.add_argument("-ei", "--end_id", type=int, default=10000)
args = parser.parse_args()

MODELS : dict[str, type[FaceEmbedder]] = {
    "resnet50"  : FaceEmbedderResNet50,
    "resnet34"  : FaceEmbedderResNet34,
    "resnet18"  : FaceEmbedderResNet18,
    "vgg16"     : FaceEmbedderVGG16
}
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
def test_softmax(train = False):
    face_embedder = MODELS[args.model].to(device)
    face_embedder.eval()
    face_embedder.load_params()

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
    threshold: tuple[float, float, float] = [0.0, 2, 0.01],
    start_id = 9001,
    end_id = 10000,
    size = 1000,
):

    face_embedder = MODELS[args.model]().to(device)
    face_embedder.load_state()
    face_embedder.eval()

    batches = DataLoader(
        dataset=CelebADataset(size=size, start_id=start_id, end_id=end_id, transformer="test"),
        shuffle=False,                  
        batch_size=256
    )
    d1_list = []
    d2_list = []
    d1_mean = d2_mean = 0.0
    count = 0
    for i, (anchor, positive, negative) in enumerate(batches):
        with tc.no_grad():
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            d1 = face_embedder.distance(anchor, positive)
            d2 = face_embedder.distance(anchor, negative)
            d1_list.append(d1) 
            d2_list.append(d2)
            d1_mean += d1.sum().item()
            d2_mean += d2.sum().item()
            count += d1.shape[0]
    d1_mean /= count
    d2_mean /= count
    start, end, step = threshold
    optimal_thres = 0
    acurracy = 0
    for thres in np.arange(start, end + step, step):
        count = 0
        correct = 0 
        pos_correct = 0
        neg_correct = 0
        for i in range(len(d1_list)):
            pos_correct += (d1_list[i] <= thres).sum().item()
            neg_correct += (d2_list[i] > thres).sum().item()
            count += 2 * d1_list[i].numel()
        correct = pos_correct + neg_correct
        if(acurracy < correct / count):
            optimal_thres = thres
            acurracy = correct / count
        
        print(f"Threshold : {thres:.3f} -- Accur : {(correct / count):.3f}", end="")
        print(f" -- Positive : {(pos_correct / count * 2):.3f} -- Negative : {(neg_correct / count * 2):.3f}")
    print(f"Optimal Threshold : {optimal_thres:.3f} -- Accuracy : {acurracy:.5f} -- d1 : {d1_mean} -- d2 : {d2_mean}")

test_triplet(size=args.size, start_id=args.start_id, end_id=args.end_id)