import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

def create_json_wider_face():
    path = "./dataset/WIDER_train/images"
    res = []
    with open("./dataset/WIDER_train/info/wider_face_train_bbx_gt.txt", "r") as info_txt:
        lines = [line.strip() for line in info_txt]
    
    i = 0
    while(i < len(lines)):
        img_path = path + f"/{lines[i]}"
        i += 1
        
        num_faces = int(lines[i])
        if(num_faces == 0): i += 1
        i += 1
        
        bboxes = []
        for _ in range(num_faces):
            parts = list(map(int, lines[i].split()))
            x1, y1, w, h = parts[:4]
            x2 = x1 + w
            y2 = y1 + h
            
            if(w > 0 and h > 0):
                bboxes.append([x1, y1, x2, y2])
            i += 1
            
        res.append({
            "path" : img_path,
            "bbox" : bboxes
        })
        
    with open("./src/face_detection/data/wider_face_info.json", "w") as file:
        json.dump(res, file, indent=2)

def create_json_celebA():
    root = "./dataset/CelebA/images"
    res = []
    with open("./dataset/CelebA/list_bbox_celeba.txt", "r") as info_txt:
        lines = [line.strip() for line in info_txt]
        
    lines = lines[2:]
    for i, line in enumerate(lines):
        if(i > 10000): 
            with open("./src/face_detection/data/celebA_info.json", "w") as file:
                json.dump(res, file, indent=2) 
            break  
        part = re.split(r"\s+", line)
        path = f"{root}/{part[0]}"
        
        x = int(part[1])
        y = int(part[2])
        w = int(part[3])
        h = int(part[4])
        bbox = [x, y, x + w, y + h]
        res.append({
            "path" : path,
            "bbox" : [bbox]
        })
create_json_celebA()
# with open("./src/face_detection/data/celebA_info.json", "r") as file:
#     w_list = [0 for _ in range(5000)]
#     h_list = [0 for _ in range(5000)]
#     w_avg = 0
#     h_avg = 0
#     info = json.load(file)
#     size = len(info)
#     for i in info:
#         x1, y1, x2, y2 = i["bbox"][0]
#         w = x2 - x1
#         h = y2 - y1
#         w_list[w] += 1
#         h_list[h] += 1
#         w_avg += w
#         h_avg += h
#     print(w_avg/size, h_avg/size)
        

        
