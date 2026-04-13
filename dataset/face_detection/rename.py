import os
import shutil

def rename():
    path = "./face_recognition/images/train/face"
    files = os.listdir(path)
    for i, file in enumerate(files):
        [_, ext] = file.split(".")
        os.rename(f"./face_recognition/images/train/face/{file}", f"./face_recognition/images/train/face/a{i}.{ext}")
        
    files = os.listdir(path)    
    for i, file in enumerate(files):
        [_, ext] = file.split(".")
        os.rename(f"./face_recognition/images/train/face/{file}", f"./face_recognition/images/train/face/{i}.{ext}")
        
    # path = "./face_recognition/images/train/non_face"
    # files = os.listdir(path)
    # for i, file in enumerate(files):
    #     [_, ext] = file.split(".")
    #     os.rename(f"./face_recognition/images/train/non_face/{file}", f"./face_recognition/images/train/non_face/a{i}.{ext}")
        
    # files = os.listdir(path)    
    # for i, file in enumerate(files):
    #     [_, ext] = file.split(".")
    #     os.rename(f"./face_recognition/images/train/non_face/{file}", f"./face_recognition/images/train/non_face/{i}.{ext}")

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

# remove()
# copy()           
rename()


