import type { ComparisonResult, FaceDetectionConfig, FaceEmbedderConfig, FaceLandmarkConfig, ServerResponse } from "../types";

const BASE_URL = import.meta.env.VITE_BASE_URL;

export async function compareFace(
    faceDetection : FaceDetectionConfig,
    faceLandmark: FaceLandmarkConfig,
    faceEmbedder: FaceEmbedderConfig,
    images: [File, File]
) : Promise<ServerResponse<ComparisonResult>> {
    const configs = {
        model_config: {
            face_detection: faceDetection, 
            face_landmark : faceLandmark, 
            face_embedder: faceEmbedder
        }}
    const formData = new FormData()
    formData.append("image", images[0])
    formData.append("image", images[1])
    formData.append("configs", JSON.stringify(configs))
    const raw = await fetch(
        `${BASE_URL}/api/face_comparison`,
        {
            body: formData,
            method : "POST",
        }
    )

    const res: Promise<ServerResponse<ComparisonResult>>  = await raw.json()
    return res
}
