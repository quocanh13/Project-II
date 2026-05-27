#rename.py
import re
import os
import shutil
import json
import torch as tc
from torchvision.transforms.functional import crop
from PIL import Image
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
# python -m src.face_verification.data.rename

def rename():
    path = "./dataset/webface_112x112"
    folders = os.listdir(path)
    for i, folder in enumerate(folders):
        files = [f for f in os.listdir(path + f"/{folder}")]
        for i, file in enumerate(files):
            os.rename(path + f"/{folder}" + f"/{file}", path + f"/{folder}" + f"/b{i}.jpg")
        
        files = [f for f in os.listdir(path + f"/{folder}")]
        for i, file in enumerate(files):
            os.rename(path + f"/{folder}" + f"/{file}", path + f"/{folder}" + f"/{i:03}.jpg")
    
    # path = "./face_verification/dataset/webface_112x112"
    # folders = os.listdir(path)
    # folders =  [f for f in folders]
    # for i, folder in enumerate(folders):
    #     os.rename(f"./face_verification/dataset/webface_112x112/{folder}", f"./face_verification/dataset/webface_112x112/c{i}")
        
    # folders = os.listdir(path)
    # folders =  [f for f in folders]
    # for i, folder in enumerate(folders):
    #     os.rename(f"./face_verification/dataset/webface_112x112/{folder}", f"./face_verification/dataset/webface_112x112/id_{i:04}")
        
def remove():
    path = "./face_recognition/images/train/face"
    files = os.listdir(path)

    for i, file in enumerate(files):
            os.remove(path + f"/{i}.jpg")
            if(i >= 4000): break

def copy():
    des = "./face_recognition/images/train/non_face"
    folders = os.listdir("C:/Users/quocanh/Downloads/archive/animals/animals")
    for folder in folders:
        src = "C:/Users/quocanh/Downloads/archive/animals/animals" + f"/{folder}"
        i = 0
        files = os.listdir(src)
        for file in files:
            shutil.copy(src + f"/{file}", des + f"/{file}")
            i += 1
            if(i >= 20): break

def create_json():
    num_id = 100
    train = []
    test = []
    train_index = []
    test_index = []
    path = "./dataset/webface_112x112"
    folders = os.listdir(path)
    folders = sorted(folders, key=lambda x: int(x.split("_")[1]))
    start_train = end_train = start_test = end_test = -1
    for i, folder in enumerate(folders):
        if(i >= num_id):
            break
        id = int(folder.split("_")[1])
        files = [f for f in os.listdir(path + f"/{folder}")]
        k = 0.7*len(files)
        start_train = end_train + 1
        start_test = end_test + 1
        for j in range(0, len(files) - 1):
            file_data = {"name" : files[j], "id" : id}
            if(j < k):
                train.append(file_data)
                end_train += 1
            else:
                test.append(file_data)
                end_test += 1
        train_index.append([start_train, end_train])
        test_index.append([start_test, end_test])
    with open("./src/face_verification/data/train_webface.json", "w") as file:
        json.dump(train, file, indent=2)
    with open("./src/face_verification/data/test_webface.json", "w") as file:
        json.dump(test, file, indent=2)
        
    with open("./src/face_verification/data/train_webface_index.json", "w") as file:
        json.dump(train_index, file, indent=2)
    with open("./src/face_verification/data/test_webface_index.json", "w") as file:
        json.dump(test_index, file, indent=2)
    with open("./src/face_verification/data/webface.json", "w") as file:
        json.dump({"num_id" : num_id}, file, indent=2)

# def create_celebA_identity():
    # faster_rcnn = FasterRCNN().eval()
    # faster_rcnn.load_state_dict(tc.load(FasterRCNN.PARAMS.FACE_DETECTION.RESNET34.PARAMS, map_location=device))    
    # faster_rcnn.to(device)
    
    # txt_pth = "./dataset/CelebA/identity_CelebA.txt"
    # root = "./dataset/CelebA/images_align"
    # json_pth = "./src/face_verification/data/json/celebA_identity.json"
    # res = [[] for _ in range(10177)]
    # count = 0
    # with open(txt_pth) as txt:
    #     for i, line in enumerate(txt):
    #         part = re.split(r"\s", line)
    #         path = f"{root}/{part[0]}"
    #         with Image.open(path).convert("RGB") as img:
    #             pred = faster_rcnn.predict(img)
    #             if(len(pred) == 0):
    #                 continue
    #             else:
    #                 bbox, _ = pred[0]
    #                 bbox = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
    #         res.append({"path" : path, "bbox" : bbox})
    #         print(f"\r{count}", end="")
    #         count += 1
    # with open(json_pth, "w") as file:
    #     json.dump(res, file, indent=2)
# import torch as tc
# from facenet_pytorch import MTCNN
# from PIL import Image
# import json
# import re
# import os

# with open("./src/face_verification/data/json/celebA_identity.json", "r+") as file:
#     celebA_id: list[list] = json.load(file)

# with open("./src/face_verification/data/json/celebA_identity_crop_.json", "r") as file:
#     celebA_id_crop = json.load(file)

# res = []
# index = 0
# for id in range(len(celebA_id)):
#     temp = []
#     for path in range(len(celebA_id[id])):
#         if(index >= 202306):
#             break
#         if(celebA_id[id][path] == celebA_id_crop[index]["path"]):
#             temp.append(celebA_id_crop[index])
#             index += 1
#     res.append(temp)
    
# with open("./src/face_verification/data/json/celebA_identity_crop.json", "r") as file:
#     f = json.load(file)
#     for i in f[10000]:
#         print(i)
    # print(f[2880])

# remove()
# copy()           
# rename()
# create_json()
# create_celebA_identity()