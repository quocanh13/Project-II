import { create } from "zustand";
import type { FaceDetectionConfig, FaceLandmarkConfig, FaceEmbedderConfig, VerificationResult } from "../types";

interface FaceVerificationStore{
    mode: "sample" | "verify",
    setMode : (mode: string)=>void,
    getMode : () => string,

    done: boolean,
    getDone : ()=>boolean,
    setDone : (done : boolean) => void,

    frameCount: number,
    getFrameCount: ()=> number,
    increaseFrameCount: () => void,

    verificationResult: VerificationResult | undefined,
    setVerificationResult: (verificationResult: VerificationResult) => void,

    faceDetectionConfig : FaceDetectionConfig,
    setFaceDetectionConfig : (faceDetectionConfig : FaceDetectionConfig) => void,

    faceLandmarkConfig: FaceLandmarkConfig,
    setFaceLandmarkConfig: (faceLandmarkConfig: FaceLandmarkConfig) => void,

    faceEmbedderConfig: FaceEmbedderConfig,
    setFaceEmbedderConfig: (faceEmbedderConfig: FaceEmbedderConfig) => void,
}

export const useFaceVerificationStore = create<FaceVerificationStore>((set, get) => ({
    mode: "sample",
    setMode(mode: string){
        if(mode == "sample" || mode == "verify"){
            set({mode})
        } else {
            set({mode : "verify"})
        }
    },

    done: true,
    getDone : () => get().done,
    setDone : (done : boolean) => {
        set({done})
    },

    getMode : ()=>{
        return get().mode
    },

    frameCount: 0,
    getFrameCount: ()=> {
        return get().frameCount
    },
    increaseFrameCount: () => {
        set(state => ({
            frameCount : state.frameCount >= 20 ? 0 : state.frameCount + 1
        }))
    },

    verificationResult: undefined,
    setVerificationResult: (verificationResult: VerificationResult) => {
        set({verificationResult})
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
}))