export const FACE_DETECTION_BACKBONE = ["vgg16", "resnet18", "resnet34"] as const
export const FACE_DETECTION_ALGORITHM = ["faster_rcnn"] as const
export const FACE_DETECTION_DATASET = ["wider_face", "celeb_A"] as const

export const FACE_LANDMARK_BACKBONE = ["resnet18", "resnet34"] as const
export const FACE_EMBEDDER_BACKBONE = ["resnet18", "resnet34", "resnet50"] as const

export type ServerResponse<Result> = {
    error : boolean,
    message : string | undefined
    result : Result | undefined
}

export type Toast = {
    id: number,
    title: string,
    message: string,
    closeTime: number | undefined,
    error: boolean
}

interface Bbox{
    x1 : number, y1 : number,
    x2 : number, y2 : number
}

export type DetectionResult = {
    bbox : Bbox,
    score : number
}

export type LandmarkResult = {
    landmark : number[]
}

export type ComparisonResult = {
    distance : number,
    percent : number,
    frontals: boolean[]
}

export type VerificationResult = {
    detection: DetectionResult[],
    frontal: boolean,
    ok: boolean,
    distance : number,
    percents: number,
    landmark: number[]
}

export type FaceDetectionBackbone = typeof FACE_DETECTION_BACKBONE[number];
export type FaceDetectionAlgorithm = typeof FACE_DETECTION_ALGORITHM[number];
export type FaceDetectionDataset = typeof FACE_DETECTION_DATASET[number];

export type FaceLandmarkBackbone = typeof FACE_LANDMARK_BACKBONE[number];

export type FaceEmbedderBackbone = typeof FACE_EMBEDDER_BACKBONE[number];

export type FaceDetectionConfig = {
    algorithm : FaceDetectionAlgorithm,
    backbone: FaceDetectionBackbone,
    dataset: FaceDetectionDataset,
}

export type FaceLandmarkConfig = {
    backbone: FaceLandmarkBackbone,
}

export type FaceEmbedderConfig = {
    backbone: FaceEmbedderBackbone,
}
