import math
import torch as tc
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
# python -m face_detection.faster_rcnn

class RegionProposal(nn.Module):
    def __init__(self, anchor: int = 9):
        super().__init__()
        self._backbone = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        self._backbone.requires_grad_(False)
        self._intermediate_layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._object = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=2*anchor, kernel_size=1, stride=1)
        )
        self._bbox = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4*anchor, kernel_size=1, stride=1)
        )
        
    def forward(
        self,
        X: tc.Tensor
    )->tuple[tc.Tensor, tc.Tensor]:
        feature_map = self._backbone.convolve(X)
        feature_map = self._intermediate_layer(feature_map)
        obj: tc.Tensor = self._object(feature_map)
        bbox: tc.Tensor = self._bbox(feature_map)
        return obj, bbox

def generate_anchors(
    feature_size: tuple[int, int],
    stride: int,
    box_size: list[int] = [15, 25, 35],
    ratio: list[float] = [0.5, 1.0, 1.5],
):
    anchors: list[list[list[list[float]]]] = []
    f_w, f_h = feature_size
    for i in range(f_h):
        anchors.append([])
        for j in range(f_w):
            anchors[i].append([])
            for s in box_size:
                for r in ratio:           
                    cx = (j + 0.5)*stride
                    cy = (i + 0.5)*stride
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    anchors[i][j].append([cx - w/2, cy - h/2, cx+ w/2,  cy + h/2])

    return tc.Tensor(anchors)     
          
def calc_single_IoU_scalar(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float]
)->float:
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2
    s1 = (x1_max - x1_min) * (y1_max - y1_min)
    s2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    w = max(0, x_right - x_left)
    h = max(0, y_bottom - y_top)
    
    overlap = w*h
    IoU = overlap / (s1 + s2 - overlap)
    return IoU

def calc_IoU(
    anchors: tc.Tensor,
    GTs: tc.Tensor
):
    anchors = anchors.unsqueeze(dim=1)
    GTs = GTs.unsqueeze(dim=0)
    
    x_left = tc.max(anchors[..., 0], GTs[..., 0])
    y_top = tc.max(anchors[..., 1], GTs[..., 1])
    x_right = tc.min(anchors[..., 2], GTs[..., 2])
    y_bottom = tc.min(anchors[..., 3], GTs[..., 3])
    
    inter_w = (x_right - x_left).clamp(0)
    inter_h = (y_bottom - y_top).clamp(0)
    inter = inter_h * inter_w
    
    s_anchor = (anchors[..., 2] - anchors[..., 0])*(anchors[..., 3] - anchors[..., 1])
    s_GTs = (GTs[..., 2] - GTs[..., 0])*(GTs[..., 3] - GTs[..., 1])
    
    IoUs = inter / (s_anchor + s_GTs - inter)
    return IoUs.flatten()

def create_label(
    anchors: tc.Tensor,
    GTs: tc.Tensor
) -> tc.Tensor:
    IoUs = calc_IoU(anchors, GTs)
    max_iou, _ = IoUs.max(dim=1)
    labels = tc.full_like(max_iou, -1, dtype=tc.int64)
    labels[max_iou >= 0.7] = 1
    labels[max_iou <= 0.3] = 0
    
    best_anchor_per_gt = IoUs.argmax(dim=0)
    labels[best_anchor_per_gt] = 1
    return labels

def train(
    epoch = 5,
    lr = 0.001
):
    faster_rcnn = RegionProposal(anchor=9)
    anchors = generate_anchors(feature_size=(14, 14), stride=16)

    optimizer = optim.Adam(
        params=faster_rcnn.parameters(), 
        lr=lr
    )
    scores: tc.Tensor
    regions: tc.Tensor
    batches = DataLoader()
    for _ in range(epoch):
        s = datetime.now()
        for input, label in batches:
            scores, regions = faster_rcnn(input)
            
            optimizer.zero_grad()
            optimizer.step()
        e = datetime.now()
        print(f"Time : {e - s} -- Loss : {loss}", end="")
        tc.save(faster_rcnn._linear.state_dict(), "./face_recognition/faster_rcnn.pth")
        print(" -- Saved")
        
 
        
# if __name__ == "__main__":
#     train()
