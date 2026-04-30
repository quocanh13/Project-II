import torch as tc
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from src.face_detection.data.dataset import WiderFaceDataset, CelebADataset
from src.face_detection.faster_rcnn import FasterRCNN
# python -m src.face_detection.train_faster_rcnn

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images, targets = zip(*batch)
    return list(images), list(targets)

def train(
    epoch = 3,
    lr = 0.001,
    load_state = False,
):
    faster_rcnn = FasterRCNN()
    if(load_state):
        faster_rcnn.load_state_dict(tc.load(FasterRCNN.PARAM.FACE_DETECTION.PARAMS, map_location=device))
    faster_rcnn.to(device=device)
    optimizer = tc.optim.SGD(
        faster_rcnn.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    batches = DataLoader(
        dataset=CelebADataset(size=1000, random=False),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    for _ in range(epoch):
        start = datetime.now()
        for images, targets in batches:
            images: list[tc.Tensor] = [img.to(device) for img in images]
            targets = [{k : v.to(device) for k, v in t.items()}  for t in targets]
            loss_dict = faster_rcnn(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        tc.save(faster_rcnn.state_dict(), FasterRCNN.PARAM.FACE_DETECTION.PARAMS)
        end = datetime.now()
        print(f"Time: {end - start} -- Loss: {losses.item()}")
        
train(load_state=True, lr=0.0001, epoch=5)