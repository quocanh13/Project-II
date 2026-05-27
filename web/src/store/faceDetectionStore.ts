import { create } from "zustand"
import defaultResultImageURL from "../assets/default/result_image.png"
import type { FaceDetectionAlgorithm, FaceDetectionBackbone, FaceDetectionDataset, DetectionResult } from "../types"
import { FACE_DETECTION_BACKBONE, FACE_DETECTION_ALGORITHM, FACE_DETECTION_DATASET } from "../types"


interface FaceDetectionStore{
    image : File | null,
    setImage : (image : File | null) => void,

    imageID: number | null,
    setImageID : (imageID: number) => void,

    imageURL : string | null,
    setImageURL : (url : string | null) => void,

    detectionResult : DetectionResult[] | null,
    setDetectionResult : (detectionResult : DetectionResult[] | null) => void,

    resultImageURL : string,
    setResultImageURL : (image : string) => void,

    bboxColor : string,
    setBboxColor : (bboxColor: string) => void,

    bboxLineWidth : number,
    setBboxLineWidth : (bboxLineWidth: number) => void,

    backbone: FaceDetectionBackbone,
    setBackbone : (backbone : string) => void,

    algorithm: FaceDetectionAlgorithm,
    setAlgorithm: (algorithm: string) => void,

    dataset: FaceDetectionDataset,
    setDataset: (dataset: string) => void,

    numBbox: string,
    setNumBbox : (numBbox : string) => void,
}

export const useFaceDetectionStore = create<FaceDetectionStore>((set) => ({
    image : null,
    setImage : (image: File | null) =>{
        set({image})
    },

    imageURL : null,
    setImageURL : (url: string | null) => {
        set({imageURL : url})
    },

    imageID: null,
    setImageID : (imageID: number) => {
        set({imageID})
    },

    detectionResult : null,
    setDetectionResult : (detectionResult : DetectionResult[] | null) => {
        set({detectionResult})
    },

    resultImageURL : defaultResultImageURL,
    setResultImageURL : (resultImageURL: string) =>{
        set({resultImageURL})
    },

    bboxColor : "red",
    setBboxColor : (bboxColor: string) =>{
        set({bboxColor})
    },

    bboxLineWidth : 5,
    setBboxLineWidth : (bboxLineWidth: number) => {
        if(bboxLineWidth < 0)
            bboxLineWidth = 0
        set({bboxLineWidth})
    },

    backbone: "resnet34",
    setBackbone : (backbone : string) => {
        for(let i = 0; i < FACE_DETECTION_BACKBONE.length; i++){
            if(backbone == FACE_DETECTION_BACKBONE[i]){
                set({backbone})
                return
            }
        }
        set({backbone : "resnet34"})
        console.warn(`Backbone "${backbone}" is not supported`);
    },

    algorithm: "faster_rcnn",
    setAlgorithm: (algorithm: string) => {
        for(let i = 0; i < FACE_DETECTION_ALGORITHM.length; i++){
            if(algorithm == FACE_DETECTION_ALGORITHM[i]){
                set({algorithm})
                return
            }
        }
        set({algorithm : "faster_rcnn"})
        console.warn(`Algorithm "${algorithm}" is not supported`);
    },

    dataset: "wider_face",
    setDataset: (dataset: string) => {
        for(let i = 0; i < FACE_DETECTION_DATASET.length; i++){
            if(dataset == FACE_DETECTION_DATASET[i]){
                set({dataset})
                return
            }
        }
        set({dataset : "wider_face"})
        console.warn(`Dataset "${dataset}" is not supported`);
    },

    numBbox: "1",
    setNumBbox : (numBbox : string) => {
        if(numBbox == ""){
        }
        else if(isNaN(Number(numBbox.slice(0, -1)))){
            numBbox = numBbox.slice(0, -1)
        } else {
            let num = Number(numBbox)
            num = Math.min(100, num)
            num = Math.max(1, num)
            numBbox = String(num)
        }
        set({numBbox})
    }
}))