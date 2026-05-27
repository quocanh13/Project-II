import { type FaceLandmarkConfig, type LandmarkResult, type ServerResponse } from "../types"

const BASE_URL = import.meta.env.VITE_BASE_URL;

export async function detectLandmark(
    image: File, 
    model_config : FaceLandmarkConfig, 
) : Promise<ServerResponse<LandmarkResult>>  {
    const formData = new FormData()
    formData.append("image", image)
    formData.append("configs", JSON.stringify({model_config}))

    const raw_res = await fetch(
        `${BASE_URL}/api/face_landmark`,
        {
            body: formData,
            method : "POST",
        }
    )

    let res = await raw_res.json()

    return res
}

