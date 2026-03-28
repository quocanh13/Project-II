from PIL import Image
import os

def rename():
    for i in range(12, 21):
        os.rename(
            f"./digit_recognition/images/images_6/pixil-frame-0 ({i}).png",
            f"./digit_recognition/images/images_6/6_{i-12}.png"
        )

# rename()

img_data_info = {
    "0" : (0, 9, 0), "1" : (0, 9, 1), "2" : (0, 9, 2),
    "3" : (0, 9, 3), "4" : (0, 9, 4), "5" : (0, 9, 5),
    "6" : (0, 9, 6), "7" : (0, 9, 7), "8" : (0, 9, 8), "9" : (0, 9, 9),
}

def get_img_data() -> list[tuple[str, int]]:
    img_data = []
    for i in img_data_info:
        s, e, l = img_data_info[i]
        for j in range(s, e + 1):
            img_data.append((get_image_path(i, j), l))
    return img_data

def get_image_path(type : str, index : int) -> Image:
    return f"./digit_recognition/images/images_{type}/{type}_{index}.png"