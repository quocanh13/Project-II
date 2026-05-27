import { type ServerResponse, type DetectionResult, type FaceDetectionConfig} from "../types"

const BASE_URL = import.meta.env.VITE_BASE_URL;

export async function detectFace(
    image: File, 
    model_config: FaceDetectionConfig, 
    num_bbox : number = 1
) : Promise<ServerResponse<DetectionResult[]>>  {
    const configs = {model_config, num_bbox}
    const formData = new FormData()
    formData.append("image", image)
    formData.append("configs", JSON.stringify(configs))

    const raw_res = await fetch(
        `${BASE_URL}/api/face_detection`,
        {
            body: formData,
            method : "POST",
        }
    )

    let res: ServerResponse<DetectionResult[]>  = await raw_res.json()
    return res
}

