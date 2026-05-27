import torch as tc
from typing import Literal
from dataclasses import dataclass
from collections.abc import Iterator
from src.face_detection import FasterRCNN, FaceDetection
from src.face_landmark import FaceLandmark
from src.face_verification import FaceVerification
from src.face_embedder.model import FaceEmbedder, FaceEmbedderResNet18, FaceEmbedderResNet34, FaceEmbedderResNet50, FaceEmbedderVGG16

class FaceDetectionConfig:
    def __init__(
        self,
        algorithm: Literal["faster_rcnn"] ="faster_rcnn",
        backbone: Literal["vgg16", "resnet18", "resnet34"] = "resnet34",
        dataset: Literal["celeb_A", "wider_face"] = "wider_face",
        config: dict[str, str] = None
    ):
        if(config):
            algorithm = config["algorithm"]
            backbone = config["backbone"]
            dataset = config["dataset"]
            
        self.algorithm = algorithm
        self.backbone = backbone
        self.dataset = dataset
        

class FaceLandmarkConfig:
    def __init__(self, backbone: Literal["resnet18", "resnet34"] = "resnet34", config: dict[str, str] = None):
        if(config):
            backbone = config["backbone"]
        self.backbone = backbone
        
class FaceEmbedderConfig:
    def __init__(self, backbone: Literal["vgg16", "resnet18", "resnet34", "resnet50"] = "resnet50", config: dict[str, str] = None):
        if(config):
            backbone = config["backbone"]
        self.backbone: Literal["vgg16", "resnet18", "resnet34", "resnet50"] = backbone
        
class FaceDetectionFactory:
    @staticmethod
    def create(config: FaceDetectionConfig) -> FaceDetection:
        if(config.algorithm == "faster_rcnn"):
            face_detection = FasterRCNN(backbone=config.backbone, dataset=config.dataset)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        face_detection.load_state()
        return face_detection
        
    def __init__(self, eval = True):
        self.models: dict[str, FaceDetection] = {}
        for algorithm in ["faster_rcnn"]:
            for backbone in ["vgg16", "resnet18", "resnet34"]:
                for dataset in ["celeb_A", "wider_face"]:
                    config = FaceDetectionConfig(algorithm, backbone, dataset)
                    model = self.create(config)
                    key = f"{algorithm} {backbone} {dataset}"
                    self.models[key] = model.eval()
    
    @staticmethod
    def get_key(
        config: FaceDetectionConfig
    ) -> str:
        if(config == None):
            config = FaceDetectionConfig("faster_rcnn", "resnet34", "wider_face")
        return f"{config.algorithm} {config.backbone} {config.dataset}"
    
    def get(
        self,         
        config: FaceDetectionConfig | None = None
    ) -> FaceDetection:
        if(config == None):
            config = FaceDetectionConfig("faster_rcnn", "resnet34", "wider_face")
        key = self.get_key(config=config)
        return self.models[key]   
        
    def __iter__(self) -> Iterator[tuple[str, FaceDetection]]:
        models = [(k, v) for k, v in self.models.items()]
        return iter(models)
        
class FaceLandmarkFactory:
    @staticmethod
    def create(
        config: FaceLandmarkConfig,
    ) -> FaceLandmark:
        face_landmark = FaceLandmark(backbone=config.backbone)
        face_landmark.load_state()
        return face_landmark
    
    def __init__(self, eval = True):
        self.models: dict[str, FaceEmbedder] = {}
        for backbone in ["resnet18", "resnet34"]:
            config = FaceLandmarkConfig(backbone=backbone)
            model = self.create(config=config)
            if(eval): model.eval()
            self.models[backbone] = model
          
    @staticmethod
    def get_key(config: FaceLandmarkConfig) -> str:
        if(config == None):
            config = FaceLandmarkConfig("resnet34")
        return f"{config.backbone}"
          
    def get(
        self, 
        config: FaceLandmarkConfig | None = None,
    ) -> FaceLandmark:
        if(config == None):
            config = FaceLandmarkConfig("resnet34")
        key = self.get_key(config=config)
        return self.models.get(key, self.models["resnet34"])
    
    def __iter__(self) -> Iterator[tuple[str, FaceLandmark]]:
        models = [(k, v) for k, v in self.models.items()]
        return iter(models)
     
class FaceEmbedderFactory:
    BACKBONE: dict[str, type[FaceEmbedder]] = {
        "vgg16" : FaceEmbedderVGG16,
        "resnet18" : FaceEmbedderResNet18,
        "resnet34" : FaceEmbedderResNet34,
        "resnet50" : FaceEmbedderResNet50,
    }
    @staticmethod
    def create(
        config: FaceEmbedderConfig
    ) -> FaceEmbedder:
        face_embedder = FaceEmbedderFactory.BACKBONE[config.backbone]()
        face_embedder.load_state()
        return face_embedder
        
    def __init__(self, eval = True):
        self.models: dict[str, FaceEmbedder] = {}
        for k, _ in self.BACKBONE.items():
            model = self.create(FaceEmbedderConfig(backbone=k))
            if(eval): model.eval()
            self.models[k] = model
          
    @staticmethod
    def get_key(config: FaceEmbedderConfig) -> str:
        if(config == None):
            config = FaceEmbedderConfig(backbone="resnet50")
        return f"{config.backbone}"

            
    def get(
        self, 
        config: FaceEmbedderConfig | None = None
        ) -> FaceEmbedder:
        if(config == None):
            config = FaceEmbedderConfig("resnet50")
        key = self.get_key(config=config)
        return self.models[key]
        
    def __iter__(self) -> Iterator[tuple[str, FaceEmbedder]]:
        models = [(k, v) for k, v in self.models.items()]
        return iter(models)    
        
class FaceVerificationFactory:
    @staticmethod
    def create(
        face_detection_config: FaceDetectionConfig | FaceDetection,
        face_landmark_config: FaceLandmarkConfig | FaceLandmark,
        face_embedder_config: FaceDetectionConfig | FaceEmbedder
    ) -> FaceVerification:
        if(isinstance(face_detection_config, FaceDetection)):
            face_detection = face_detection_config
        else:
            face_detection = FaceDetectionFactory.create(face_detection_config)
            
        if(isinstance(face_landmark_config, FaceLandmark)):
            face_landmark = face_landmark_config
        else:
            face_landmark = FaceLandmarkFactory.create(face_landmark_config)
        
        if(isinstance(face_embedder_config, FaceEmbedder)):
            face_embedder = face_embedder_config
        else:
            face_embedder = FaceEmbedderFactory.create(face_embedder_config)
            
        model = FaceVerification(face_detection, face_landmark, face_embedder)
        return model
        
    def __init__(
        self,
        eval = True,
        face_detection_factory: FaceDetectionFactory | None = None,
        face_landmark_factory: FaceLandmarkFactory | None = None,
        face_embedder_factory: FaceEmbedderFactory | None = None,
    ):
        face_detection_factory = face_detection_factory if face_detection_factory != None else FaceDetectionFactory(eval)
        face_landmark_factory = face_landmark_factory if face_landmark_factory != None else FaceLandmarkFactory(eval)
        face_embedder_factory = face_embedder_factory if face_embedder_factory != None else FaceEmbedderFactory(eval)

        self.models: dict[str, FaceVerification] = {}
        
        for face_detection_key, face_detection_model in face_detection_factory:
            for face_landmark_key, face_landmark_model in face_landmark_factory:
                for face_embedder_key, face_embedder_model in face_embedder_factory:
                    self.models[f"{face_detection_key} {face_landmark_key} {face_embedder_key}"] = FaceVerification(
                        face_detection=face_detection_model,
                        face_landmark=face_landmark_model,
                        face_embedder=face_embedder_model
                    )
    
    @staticmethod
    def get_key(
        face_detection_config: FaceDetectionConfig,
        face_landmark_config: FaceLandmarkConfig,
        face_embedder_config: FaceDetectionConfig
    ) -> str:
        face_detection_key = FaceDetectionFactory.get_key(face_detection_config)
        face_landmark_key = FaceLandmarkFactory.get_key(face_landmark_config)
        face_embedder_key = FaceEmbedderFactory.get_key(face_embedder_config)
        return f"{face_detection_key} {face_landmark_key} {face_embedder_key}"
        
    
    def get(
        self, 
        face_detection_config: FaceDetectionConfig | None = None,
        face_landmark_config: FaceLandmarkConfig | None = None,
        face_embedder_config: FaceEmbedderConfig | None = None
    ) -> FaceVerification:
        key = self.get_key(face_detection_config, face_landmark_config, face_embedder_config)
        return self.models[key]
        
    def __iter__(self) -> Iterator[tuple[str, FaceEmbedder]]:
        models = [(k, v) for k, v in self.models.items()]
        return iter(models)    
          
