import json
import regex

def create_landmark():
    root = "./dataset/CelebA/images"
    res = []
    with open("./dataset/CelebA/list_landmarks_celeba.txt") as file:
        next(file); next(file)
        for line in file:
            p = regex.split(r"\s+", line.strip())
            res.append({
                "path" : f"{root}/{p[0]}",
                "landmark" : [int(i) for i in p[1:]]
            })
    with open("./src/face_landmark/dataset/json/landmark.json", "w") as file:
        json.dump(res, file, indent=2)
        
create_landmark()