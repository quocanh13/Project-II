import { create } from "zustand"
import { type ComparisonResult, type FaceDetectionConfig, type FaceLandmarkConfig, type FaceEmbedderConfig, FACE_DETECTION_BACKBONE } from "../types"

interface FaceComparisonStore{
    images: [File | null, File | null],
    setImages : (images : [File | null, File | null]) => void,
    getValidImages : ()=>[File, File] | null

    faceDetectionConfig : FaceDetectionConfig,
    setFaceDetectionConfig : (faceDetectionConfig : FaceDetectionConfig) => void,

    faceLandmarkConfig: FaceLandmarkConfig,
    setFaceLandmarkConfig: (faceLandmarkConfig: FaceLandmarkConfig) => void,

    faceEmbedderConfig: FaceEmbedderConfig,
    setFaceEmbedderConfig: (faceEmbedderConfig: FaceEmbedderConfig) => void,

    comparisonResult : ComparisonResult | null,
    setComparisonResult : (comparisonResult : ComparisonResult | null) => void,
}

export const useFaceComparisonStore = create<FaceComparisonStore>((set, get) => ({
    images : [null, null] as [File | null, File | null],
    setImages(images : [File | null, File | null]) {
        set({images})
    },

    getValidImages : () : [File, File] | null =>{
        const {images} = get()
        for(let i = 0; i < images.length; i++){
            if(images[i] == null) 
                return null
        }
        return images as [File, File]
    },

    faceDetectionConfig : {algorithm: "faster_rcnn", backbone : "resnet34", dataset: "wider_face"},
    setFaceDetectionConfig : (faceDetectionConfig : FaceDetectionConfig,) => {
        set({ faceDetectionConfig })
    },

    faceLandmarkConfig: {backbone: "resnet34"},
    setFaceLandmarkConfig: (faceLandmarkConfig: FaceLandmarkConfig) => {
        set({ faceLandmarkConfig})
    },

    faceEmbedderConfig: {backbone: "resnet50"},
    setFaceEmbedderConfig: (faceEmbedderConfig: FaceEmbedderConfig) => {
        set({ faceEmbedderConfig })
    },

    comparisonResult : null,
    setComparisonResult : (comparisonResult : ComparisonResult | null) => {
        set({ comparisonResult })
    }
}))