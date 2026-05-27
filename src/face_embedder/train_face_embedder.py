import argparse
import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from typing import Callable
from datetime import datetime
from torch.utils.data import DataLoader
from src.face_embedder.model import *
from src.face_embedder.data.dataset import *
# python -m src.face_embedder.train_face_embedder
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="resnet50")
parser.add_argument("-o", "--optimizer", type=str, default="semi_hard")
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-lr", "--lr", type=float, default=0.001)
parser.add_argument("-ep", "--epoch", type=int, default=5)
parser.add_argument("-s", "--size", type=int, default=10000)
parser.add_argument("-p", "--params", type=str, default="celeb_A")
parser.add_argument("-ls", "--load_state", action="store_true")
parser.add_argument("-P", type=int, default=64)
parser.add_argument("-K", type=int, default=4)
parser.add_argument("-bn", "--batch_num", type=int, default=64)
parser.add_argument("-bs", "--batch_size", type=int, default=128)
parser.add_argument("-mg", "--margin", type=float, default=0.4)
args = parser.parse_args()

MODELS : dict[str, type[FaceEmbedder]] = {
    "resnet50"  : FaceEmbedderResNet50,
    "resnet34"  : FaceEmbedderResNet34,
    "resnet18"  : FaceEmbedderResNet18,
    "vgg16"     : FaceEmbedderVGG16
}

def train_triplet(
    epoch: int = 10,
    lr : float = 0.001,
    margin: float = 0.5,
    batch_size = 128,
    load_state = True,
    model = "resnet50",
    params = "celeb_A",
    size = 5000,
    device = "cuda"
):
    MODEL = MODELS[model]
    face_embedder = MODEL(params=params)

    module = face_embedder
    if(load_state):
        face_embedder.load_state()
    if tc.cuda.device_count() > 1:
        face_embedder = nn.DataParallel(face_embedder)
        module = face_embedder.module
    face_embedder.to(device=device)
    
    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(
        params=[
            {"params": module.parameters()},
        ], 
        lr=lr)
    batches = DataLoader(
        dataset=CelebADataset(sampler="triple", size=size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    scaler = tc.amp.GradScaler(device)
    
    for ep in range(epoch):
        training_loss = 0.0
        s = datetime.now()
        face_embedder.train()
        for i, (anchor, positive, negative) in enumerate(batches):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()
            with tc.amp.autocast(device):
                anchor_vector = face_embedder(anchor)
                positive_vector = face_embedder(positive)
                negative_vector = face_embedder(negative)
            
                loss: tc.Tensor = criterion(anchor_vector.float(), positive_vector.float(), negative_vector.float())
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            training_loss += loss.item()

        e = datetime.now()
        print(f"Epoch : {ep} -- Time : {e - s} -- Training loss : {training_loss / len(batches)}", end="")
        module.save_state()
        print(" -- Saved")

def train_triplet_semi_hard(
    params = "celeb_A",
    model = "resnet50",
    epoch: int = 10,
    lr : float = 0.001,
    margin: float = 0.5,
    load_state = True,
    P = 64,
    K = 4,
    batch_num = 64,
    device = "cuda",
):
    MODEL = MODELS[model]
    face_embedder = MODEL(params=params)    

    module = face_embedder
    if(load_state):
        face_embedder.load_state()
    if tc.cuda.device_count() > 1:
        face_embedder = nn.DataParallel(face_embedder)
        module = face_embedder.module
    face_embedder.to(device=device)
    
    criterion = SemiHardTripletLoss(margin=margin)
    optimizer = optim.Adam(
        params=[
            {"params": module.parameters()},
        ], 
        lr=lr, weight_decay=0.0001)
    batches = DataLoader(
        dataset=CelebADataset(sampler="id"),
        batch_sampler=CelebAPKSampler(P=P, K=K, batch_num=batch_num),
        num_workers=4
    )
    scaler = tc.amp.GradScaler(device)
    
    for ep in range(epoch):
        training_loss = 0.0
        s = datetime.now()
        face_embedder.train()
        for i, (images, labels) in enumerate(batches):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with tc.amp.autocast(device):
                embeddings = face_embedder(images)
                loss: tc.Tensor = criterion(embeddings, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.item()

        e = datetime.now()
        print(f"Epoch : {ep} -- Time : {e - s} -- Training loss : {training_loss / len(batches)}", end="")
        module.save_state()
        print(" -- Saved")

def train_triplet_hard(
    params = "celeb_A",
    model = "resnet50",
    epoch: int = 10,
    lr : float = 0.001,
    margin: float = 0.5,
    load_state = True,
    P = 64,
    K = 4,
    batch_num = 64,
    device = "cuda",
):
    MODEL = MODELS[model]
    face_embedder = MODEL(params=params)    
    
    module = face_embedder
    if(load_state):
        face_embedder.load_state()
    if tc.cuda.device_count() > 1:
        face_embedder = nn.DataParallel(face_embedder)
        module = face_embedder.module
    face_embedder.to(device=device)
    
    criterion = HardTripletLoss(margin=margin)
    optimizer = optim.Adam(
        params=[
            {"params": module.parameters()},
        ], 
        lr=lr)
    batches = DataLoader(
        dataset=CelebADataset(sampler="id"),
        batch_sampler=CelebAPKSampler(P=P, K=K, batch_num=batch_num),
        num_workers=4
    )
    scaler = tc.amp.GradScaler(device)
    
    for ep in range(epoch):
        training_loss = 0.0
        s = datetime.now()
        face_embedder.train()
        for i, (images, labels) in enumerate(batches):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with tc.amp.autocast(device):
                embeddings = face_embedder(images)
                loss: tc.Tensor = criterion(embeddings, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.item()

        e = datetime.now()
        print(f"Epoch : {ep} -- Time : {e - s} -- Training loss : {training_loss / len(batches)}", end="")
        module.save_state()
        print(" -- Saved")

TRAINS : dict[str, Callable] = {
    "triple" : train_triplet,
    "semi_hard" : train_triplet_semi_hard,
    "hard" : train_triplet_hard
}

if __name__ == "__main__":
    train = TRAINS[args.optimizer]
    if(args.optimizer == "triple"):
        train(
            params = args.params, 
            model = args.model, 
            epoch = args.epoch,
            lr = args.lr,
            size = args.size,
            margin = args.margin,
            load_state = args.load_state,
        )
    else:
        train(
            params = args.params, 
            model = args.model, 
            epoch = args.epoch,
            lr = args.lr,
            P = args.P,
            K = args.K,
            margin = args.margin,
            batch_num = args.batch_num,
            load_state = args.load_state,
        )