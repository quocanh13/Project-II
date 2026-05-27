import argparse
import torch as tc
import torch.nn as nn
import torch.amp as amp
from datetime import datetime
from torch.utils.data import DataLoader
from src.face_detection.data.dataset import WiderFaceDataset, CelebADataset
from src.face_detection.faster_rcnn.model import FasterRCNN
# python -m src.face_detection.train_faster_rcnn
# device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="resnet34")
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-lr", "--lr", type=float, default=0.001)
parser.add_argument("-ep", "--epoch", type=int, default=5)
parser.add_argument("-bs", "--batch_size", type=int, default=5)
parser.add_argument("-s", "--size", type=int, default=10000)
parser.add_argument("-ls", "--load_state", action="store_true")
parser.add_argument("-ds", "--dataset", type=str, default="wider_face")
args = parser.parse_args()

scaler = amp.GradScaler("cuda")

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images, targets = zip(*batch)
    return list(images), list(targets)

def train(
    epoch = 3,
    lr = 0.001,
    size = 10000,
    load_state = False,
    model = "resnet34",
    device = "cpu",
    dataset = "wider_face",
    batch_size = 32
):
    model = FasterRCNN(model, dataset=dataset)
    print(model.params)
    if(load_state):
        model.load_state(device=args.device)
    model.to(device=device)
    optimizer = tc.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
    )
    
    if(dataset == "celeb_A"):
        dataset=CelebADataset(size=size, random=False)
    else:
        dataset=WiderFaceDataset(size=size, random=False)
    
    batches = DataLoader(
        dataset=dataset,
        batch_size=56,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    for ep in range(epoch):
        start = datetime.now()
        avg_loss = 0.0
        total_sample = 0
        for images, targets in batches:
            images: list[tc.Tensor] = [img.to(device) for img in images]
            targets = [{k : v.to(device) for k, v in t.items()}  for t in targets]
            with amp.autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_size = len(images)
            avg_loss += losses.item() * batch_size
            total_sample += batch_size
        model.save_state()
        end = datetime.now()
        print(f"Epoch : {ep} -- Time: {end - start} -- Loss: {avg_loss / total_sample}")

print(args.load_state)
train(
    load_state=args.load_state, 
    lr=args.lr, 
    epoch=args.epoch, 
    model=args.model, 
    device=args.device, 
    dataset=args.dataset,
    batch_size=args.batch_size
)