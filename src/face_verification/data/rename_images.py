import os
import json

def rename():
    path = "./dataset/images/"
    folders = os.listdir(path)
    for i, folder in enumerate(folders):
        files = [f for f in os.listdir(path + f"/{folder}")]
        for i, file in enumerate(files):
            os.rename(path + f"/{folder}" + f"/{file}", path + f"/{folder}" + f"/b{i}.jpg")
        
        files = [f for f in os.listdir(path + f"/{folder}")]
        for i, file in enumerate(files):
            os.rename(path + f"/{folder}" + f"/{file}", path + f"/{folder}" + f"/{i:03}.jpg")
            
def create_json():
    path = "./dataset/images/"
    folders_path = "./dataset/images/dataset/"
    folders = os.listdir(folders_path)
    res = []
    for i, folder in enumerate(folders):
        files_path = folders_path + f"{folder}/"
        files = os.listdir(files_path)
        for file in files:
            file_path = files_path + f"{file}"
            res.append({"name" : f"{folder}_{file.split(".")[0]}", "path" : file_path})
    with open(path + "info.json", "w") as file:
        json.dump(res, file, indent=3)
    