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
parser.add_argument("-s", "--start", type=int, default=0)
parser.add_argument("-e", "--end", type=int, default=9000)
parser.add_argument("-bs", "--batch_size", type=int, default=128)
args = parser.parse_args()

device = args.device
backbone = args.backbone
start = args.start
end = args.end
batch_size = args.batch_size

def test_face_landmark(
    device = device,
    backbone = backbone,
    start = start,
    end = end,
    batch_size = batch_size
):
    model = FaceLandmark(backbone=backbone)
    model.to(device=device)
    model.load_state(device=device)
        
    criterion = nn.SmoothL1Loss()
    dataset = CelebADataset(start=start, end=end)
    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    avg_loss = 0.0
    total_size = 0
    start_time = datetime.now()
    for batch in batches:
        image, landmark = batch
        image = image.to(device)
        landmark = landmark.to(device)
        with tc.no_grad():
            res = model(image)
        loss: tc.Tensor = criterion(res, landmark)
        cur_batch_size = image.size(0)
        avg_loss += loss.item()*cur_batch_size
        total_size += cur_batch_size
        
    end_time = datetime.now()
    print(f"Test Loss -- Time : {end_time - start_time} -- Loss : {avg_loss / total_size}")
        

if __name__ == "__main__":
    test_face_landmark()
        