import torch as tc
import torchvision
import torch.nn as nn
from src.face_detection.backbone import BackboneVGG16, BackboneMiniVGG16
from torchvision.models.detection import FasterRCNN as TorchFasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator   
    
class FasterRCNN(nn.Module):
    class PARAM:
        class FACE_DETECTION:
            PARAMS = "./params/face_detection/params.pth"
    
    def __init__(
        self,
        backbone=None,
        anchor_generator=None,
        roi_pooler=None
    ):
        super().__init__()

        if backbone is None:
            backbone = BackboneVGG16(featmap_list=['c2', 'c3', 'c4', 'c5'])

        if anchor_generator is None:
            anchor_generator = AnchorGenerator(
                sizes=((150,), (300,), (500,), (800,)),
                aspect_ratios=((1.0,),) * 4
            )

        if roi_pooler is None:
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['c2', 'c3', 'c4', 'c5'],
                output_size=7,
                sampling_ratio=2
            )
        super().__init__()
        self.model = TorchFasterRCNN(
            backbone=backbone,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            num_classes=2,
            rpn_post_nms_top_n_train=3,
            rpn_pre_nms_top_n_train=5
        )
    
    def forward(
        self,
        images: list[tc.Tensor],
        targets: list[dict] = None
    ):
        return self.model(images, targets)
        
