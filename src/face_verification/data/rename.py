import os
import shutil
import json

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
    train = []
    test = []
    path = "./dataset/webface_112x112"
    folders = os.listdir(path)
    for i, folder in enumerate(folders):
        if(i >= 100):
            break
        id = int(folder.split("_")[1])
        files = [f for f in os.listdir(path + f"/{folder}")]
        k = 0.7*len(files)
        for j in range(0, len(files) - 1):
            file_data = {"name" : files[j], "id" : id}
            if(j < k):
                train.append(file_data)
            else:
                test.append(file_data)
    with open("./src/face_verification/data/train_webface.json", "w") as file:
        json.dump(train, file, indent=2)
    with open("./src/face_verification/data/test_webface.json", "w") as file:
        json.dump(test, file, indent=2)

# remove()
# copy()           
# rename()
create_json()


