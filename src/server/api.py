import io
import json
from PIL import Image
from flask import jsonify, request, Blueprint
from src.server.models import detect_face, compare_face, detect_landmark, sample_face, verify_face
from src.utils.factory import FaceLandmarkConfig, FaceDetectionConfig
# python -m src.server.api

api = Blueprint("api", __name__)
sample_image = None

@api.route('/api/face_detection', methods=['POST'])
def detect_face_api():
    image_file = request.files.get("image")
    configs = request.form.get("configs")
    configs = json.loads(configs)

    if image_file is None:
        return jsonify({
            "error": True,
            "message" : "You need to upload image"
        }), 400

    image = Image.open(io.BytesIO(image_file.read()))
    model_config = configs["model_config"]
    print(configs)
    res = detect_face(
        image=image, 
        config=FaceDetectionConfig(model_config["algorithm"], model_config["backbone"], model_config["dataset"]), 
    )
    return jsonify({
        "error" : False,
        "result" : res
    })
    
@api.route('/api/face_landmark', methods=['POST'])
def detect_landmark_api():
    image = request.files.get("image")
    configs = request.form.get("configs")
    configs = json.loads(configs)

    image = Image.open(io.BytesIO(image.read()))
    model_config = configs["model_config"]
    
    landmark = detect_landmark(
        image=image, 
        config=FaceLandmarkConfig(backbone=model_config["backbone"])
    )
    
    return jsonify({
        "error" : False,
        "result" : {
            "landmark" : landmark
        }
    })
    
@api.route('/api/face_comparison', methods=['POST'])
def compare_face_api():
    images = request.files.getlist("image")
    configs = request.form.get("configs")
    configs = json.loads(configs)

    if images is None or len(images) < 2:
        return jsonify({
            "error": True,
            "message" : "You need to upload image"
        }), 400

    model_config = configs["model_config"]

    images[0] = Image.open(io.BytesIO(images[0].read()))
    images[1] = Image.open(io.BytesIO(images[1].read()))
    images[0] = images[0].convert("RGB")
    images[1] = images[1].convert("RGB")
    res = compare_face(images, model_config["face_detection"], model_config["face_landmark"], model_config["face_embedder"])
    if(isinstance(res, str)):
        return jsonify({
            "error" : True,
            "message" : res
        })
    
    distance, percent, frontals = res
    
    return jsonify({
        "error" : False,
        "result" : {"distance" : distance, "percent" : percent, "frontals" : frontals}
    })
    
    
@api.route('/api/sample_face', methods=['POST'])
def sample_face_api():
    global sample_image
    
    image = request.files.get("image")
    configs = request.form.get("configs")
    configs = json.loads(configs)

    if image is None:
        return jsonify({
            "error": True,
            "message" : "You need to upload image"
        }), 400

    model_config = configs["model_config"]

    image = Image.open(io.BytesIO(image.read()))
    image = image.convert("RGB")
    res = sample_face(image, model_config["face_detection"], model_config["face_landmark"])
    detection, landmark, frontal, image = res
    
    ok = False
    if(image != None):
        ok = True
        sample_image = image.convert("RGB")
    
    return jsonify({
        "error" : False,
        "result" : {"detection" : detection, "frontal" : frontal, "ok" : ok, "landmark" : landmark}
    })
    
@api.route('/api/verify_face', methods=['POST'])
def verify_face_api():
    global sample_image
    
    verify_image = request.files.get("image")
    configs = request.form.get("configs")
    configs = json.loads(configs)

    if verify_image is None:
        return jsonify({
            "error": True,
            "message" : "You need to upload image"
        }), 400

    model_config = configs["model_config"]

    verify_image = Image.open(io.BytesIO(verify_image.read()))
    verify_image = verify_image.convert("RGB")
    res = verify_face(sample_image, verify_image, model_config["face_detection"], model_config["face_landmark"], model_config["face_embedder"])
    detection, landmark, frontal, distance, percent, ok = res
    
    return jsonify({
        "error" : False,
        "result" : {
            "detection" : detection, 
            "frontal" : frontal, 
            "distance" : distance,
            "ok" : ok, 
            "percent" : percent,
            "landmark" : landmark
        }
    })