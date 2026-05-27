import { create } from "zustand"
import defaultResultImageURL from "../assets/default/result_image.png"
import type { LandmarkResult } from "../types"


interface FaceLandmarkStore{
    image : File | null,
    setImage : (image : File | null) => void,

    imageURL : string | null,
    setImageURL : (url : string | null) => void,

    landmarkResult : LandmarkResult | null,
    setLandmarkResult : (landmarkResult : LandmarkResult | null) => void,

    resultImageURL : string,
    setResultImageURL : (image : string) => void,

    backbone: string,
    setBackbone : (backbone : string) => void,

    radius: number,
    setRadius: (radius: number) => void
}

export const useFaceLandmarkStore = create<FaceLandmarkStore>((set) => ({
    image : null,
    setImage : (image: File | null) =>{
        set({image})
    },

    imageURL : null,
    setImageURL : (url: string | null) => {
        set({imageURL : url})
    },

    landmarkResult : null,
    setLandmarkResult : (landmarkResult : LandmarkResult | null) => {
        set({landmarkResult})
    },

    resultImageURL : defaultResultImageURL,
    setResultImageURL : (resultImageURL: string) =>{
        set({resultImageURL})
    },

    backbone: "resnet34",
    setBackbone : (backbone : string) => {
        set({backbone})
    },

    radius: 10,
    setRadius: (radius: number) => {
        if(radius < 0) radius = 0
        set({radius})
    },
}))