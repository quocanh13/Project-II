import argparse
import torch as tc
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from src.face_landmark.model import FaceLandmark
from src.face_landmark.dataset.dataset import CelebADataset

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-b", "--backbone", type=str, default="resnet18")
parser.add_argument("-lr", "--lr", type=float, default=0.001)
parser.add_argument("-ep", "--epoch", type=int, default=5)
parser.add_argument("-s", "--start", type=int, default=0)
parser.add_argument("-e", "--end", type=int, default=9000)
parser.add_argument("-bs", "--batch_size", type=int, default=128)
parser.add_argument("-ls", "--load_state", action="store_true")
args = parser.parse_args()

device = args.device
backbone = args.backbone
lr = args.lr
epoch = args.epoch
start = args.start
end = args.end
batch_size = args.batch_size
load_state = args.load_state

def train_face_landmark(
    device = device,
    backbone = backbone,
    lr = lr,
    epoch = epoch,
    start = start,
    end = end,
    load_state = load_state,
    batch_size = batch_size
):
    model = FaceLandmark(backbone=backbone)
    model.to(device=device)
    if(load_state):
        model.load_state(device=device)
        
    module = model
    if tc.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        module = model.module
        
    criterion = nn.SmoothL1Loss()
    optimizer = tc.optim.Adam(
        module.parameters(),
        lr=lr
    )
    dataset = CelebADataset(start=start, end=end)
    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    for ep in range(epoch):
        avg_loss = 0.0
        total_size = 0
        start_time = datetime.now()
        for batch in batches:
            image, landmark = batch
            image = image.to(device)
            landmark = landmark.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            res = model(image)
            loss: tc.Tensor = criterion(res, landmark)
            loss.backward()           

            cur_batch_size = image.size(0)
            avg_loss += loss.item()*cur_batch_size
            total_size += cur_batch_size
            
            optimizer.step()
        end_time = datetime.now()
        print(f"Epoch: {ep} -- Time : {end_time - start_time} -- Loss : {avg_loss / total_size}", end="")
        module.save_state()
        print(" -- Saved State")
        

if __name__ == "__main__":
    train_face_landmark()
        