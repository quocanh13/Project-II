import torch as tc
import torchvision
from typing import Literal
from PIL import Image
from typing import Literal
from torchvision import transforms as tf
from torchvision.models.detection import FasterRCNN as TorchFasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator   
from src.face_detection.faster_rcnn.backbone import FasterRCNNBackbone, BackboneVGG16, BackboneResNet18, BackboneResNet34
from src.face_detection.base import FaceDetection
    
class FasterRCNN(FaceDetection):
    class PARAMS:
        class FACE_DETECTION:
            class VGG16:
                WIDER_FACE = "./params/face_detection/vgg16/wider_face.pth"
                CELEB_A = "./params/face_detection/vgg16/celeb_A.pth"
            class RESNET18:
                WIDER_FACE = "./params/face_detection/resnet18/wider_face.pth"
                CELEB_A = "./params/face_detection/resnet18/celeb_A.pth"
            class RESNET34:
                WIDER_FACE = "./params/face_detection/resnet34/wider_face.pth"
                CELEB_A = "./params/face_detection/resnet34/celeb_A.pth"
                
    BACKBONE = {
        "vgg16"     : {
            "class" : BackboneVGG16,     
            "dataset" : {
                "wider_face" : PARAMS.FACE_DETECTION.VGG16.WIDER_FACE,
                "celeb_A" : PARAMS.FACE_DETECTION.VGG16.CELEB_A
            } 
        },
        "resnet18"  : {
            "class" : BackboneResNet18,  
            "dataset" : {
                "wider_face" : PARAMS.FACE_DETECTION.RESNET18.WIDER_FACE,
                "celeb_A" : PARAMS.FACE_DETECTION.RESNET18.CELEB_A
            } 
        },
        "resnet34"  : {
            "class" : BackboneResNet34,  
            "dataset" : {
                "wider_face" : PARAMS.FACE_DETECTION.RESNET34.WIDER_FACE,
                "celeb_A" : PARAMS.FACE_DETECTION.RESNET34.CELEB_A
            } 
        },
    }
    
    TRANSFORMER = tf.Compose([
        tf.ToTensor()
    ])
    
    def __init__(
        self,
        backbone: Literal["resnet34", "resnet18", "vgg16"] = "resnet18",
        dataset : Literal["wider_face", "celeb_A"] = "wider_face",
        anchor_generator=None,
        roi_pooler=None
    ):
        super().__init__()
        DEFAULT_BACKBONE = "resnet34"
        backbone_info = self.BACKBONE.get(backbone, self.BACKBONE[DEFAULT_BACKBONE])
        self.backbone: FasterRCNNBackbone = backbone_info["class"]()
        self.params = backbone_info["dataset"][dataset]

        if anchor_generator is None:
            anchor_generator = AnchorGenerator(
                sizes=(
                    # (70,), (140,), (280,), (560,)
                    (40,), (90,), (200,), (400,)
                ),
                aspect_ratios=((1.0,),) * 4
            )

        if roi_pooler is None:
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                # featmap_names=['c2', 'c3', 'c4', 'c5'],
                featmap_names=self.backbone.featmap_list,
                output_size=7,
                sampling_ratio=2
            )
        self.model = TorchFasterRCNN(
            backbone=self.backbone,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            num_classes=2,
            box_nms_thresh=0.5,
            box_score_thresh=0.9,
            # rpn_post_nms_top_n_train=3,
            # rpn_pre_nms_top_n_train=5,
            max_size=640,
            min_size=480
        )
    
    def forward(
        self,
        images: list[tc.Tensor],
        targets: list[dict] = None
    ):
        return self.model(images, targets)
    
    def load_state(self, device: str | None = None):
        if(device == None):
            device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
        self.load_state_dict(tc.load(self.params, map_location=device))
    
    def save_state(self) -> None:
        tc.save(self.state_dict(), self.params)
    
    
    def detect(
        self,
        img : Image.Image | tc.Tensor,
        num_bbox = -1,
    ) -> list[tuple[list[int], float]]:
        self.eval()
        
        if(isinstance(img, Image.Image)):
            img = img.convert("RGB")
            img = self.TRANSFORMER(img)
        device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
        img = img.to(device)
        
        with tc.no_grad():
            pred = self.forward([img])[0]
            
        res = []
        if(num_bbox == -1):
            num_bbox = len(pred["boxes"])
        else:
            num_bbox = min(num_bbox, len(pred["boxes"]))
            
        for i in range(0, num_bbox):
            res.append(([it.item() for it in pred["boxes"][i]], pred["scores"][i].item()))
            
        return res
        
        
