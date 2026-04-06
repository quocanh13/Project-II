from PIL import Image
import os

def rename():
    for i in range(6, 11):
        os.rename(
            f"./digit_recognition/images/images_9/pixil-frame-0 ({i}).png",
            f"./digit_recognition/images/images_9/9_{i+5}.png"
        )

# rename()

img_data_info = {
    "0" : (0, 9, 0), "1" : (0, 9, 1), "2" : (0, 9, 2),
    "3" : (0, 9, 3), "4" : (0, 9, 4), "5" : (0, 9, 5),
    "6" : (0, 9, 6), "7" : (0, 9, 7), "8" : (0, 9, 8), "9" : (0, 9, 9),
}

img_test_info = {
    "0" : (10, 15, 0), "1" : (10, 15, 1), "2" : (10, 15, 2),
    "3" : (10, 15, 3), "4" : (10, 15, 4), "5" : (10, 15, 5),
    "6" : (10, 15, 6), "7" : (10, 15, 7), "8" : (10, 15, 8), "9" : (10, 1, 9),
}

def get_img_data(train = True) -> list[tuple[str, int]]:
    data_info = img_test_info
    if(train): data_info = img_data_info
    img_data = []
    for i in data_info:
        s, e, l = data_info[i]
        for j in range(s, e + 1):
            img_data.append((get_image_path(i, j), l))
    return img_data

def get_image_path(type : str, index : int) -> Image:
    return f"./digit_recognition_fnn/images/images_{type}/{type}_{index}.png"

