import os
import json

def rename():
    path = "./dataset/football_player/"
    folders = os.listdir(path)
    for i, folder in enumerate(folders):
        files = [f for f in os.listdir(path + f"/{folder}")]
        for i, file in enumerate(files):
            os.rename(path + f"/{folder}" + f"/{file}", path + f"/{folder}" + f"/b{i}.png")
        
        files = [f for f in os.listdir(path + f"/{folder}")]
        for i, file in enumerate(files):
            os.rename(path + f"/{folder}" + f"/{file}", path + f"/{folder}" + f"/{i:03}.png")
            
def create_json():
    folders_path = "./dataset/football_player/"
    folders = os.listdir(folders_path)
    res = []
    for i, folder in enumerate(folders):
        files_path = folders_path + f"{folder}/"
        files = os.listdir(files_path)
        for file in files:
            file_path = files_path + f"{file}"
            res.append({"name" : f"{folder}_{file.split(".")[0]}", "path" : file_path})
    with open("./src/face_verification/data/json/football_player.json", "w") as file:
        json.dump(res, file, indent=3)

rename()
create_json()