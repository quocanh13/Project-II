import torch as tc
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from datetime import datetime
# python -m face_detection.faster_rcnn

vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
vgg16.named_children()[0:24]

vgg16.out_channels = 512

anchor_generator = AnchorGenerator()
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], 
    output_size=7, 
    sampling_ratio=2
)
faster_rcnn = FasterRCNN(
    backbone=vgg16,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    num_classes=2
)
faster_rcnn = faster_rcnn.eval()
transformer = tf.Compose([
    tf.Resize((224, 224)),
    tf.ToTensor()
])
# with Image.open("./face_recognition/images/test/face/5.png") as img:
#     img = transformer(img)
#     print(faster_rcnn([img]))

