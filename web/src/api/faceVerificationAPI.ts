import type { FaceDetectionConfig, FaceEmbedderConfig, FaceLandmarkConfig, ServerResponse, VerificationResult } from "../types";

const BASE_URL = import.meta.env.VITE_BASE_URL;

export async function sampleFace(
    faceDetection : FaceDetectionConfig,
    faceLandmark: FaceLandmarkConfig,
    image: File | Blob
) : Promise<ServerResponse<VerificationResult>> {
    const configs = {
        model_config: {
            face_detection: faceDetection, 
            face_landmark : faceLandmark, 
        }}
    const formData = new FormData()
    formData.append("image", image)
    formData.append("configs", JSON.stringify(configs))
    const raw = await fetch(
        `${BASE_URL}/api/sample_face`,
        {
            body: formData,
            method : "POST",
        }
    )

    const res: Promise<ServerResponse<VerificationResult>>  = raw.json()
    return res
}

export async function verifyFace(
    faceDetection : FaceDetectionConfig,
    faceLandmark: FaceLandmarkConfig,
    faceEmbedder: FaceEmbedderConfig,
    image: File | Blob
) : Promise<ServerResponse<VerificationResult>> {
    const configs = {
        model_config: {
            face_detection: faceDetection, 
            face_landmark : faceLandmark, 
            face_embedder : faceEmbedder
        }}
    const formData = new FormData()
    formData.append("image", image)
    formData.append("configs", JSON.stringify(configs))
    const raw = await fetch(
        `${BASE_URL}/api/verify_face`,
        {
            body: formData,
            method : "POST",
        }
    )

    const res: Promise<ServerResponse<VerificationResult>>  = raw.json()
    return res
}