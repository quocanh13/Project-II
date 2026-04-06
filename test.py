import cv2
import numpy as np
from PIL import Image


with Image.open("./numbers-12.png") as img:
    img = np.array(img.convert("RGB"), dtype=np.uint8)
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    print(f"Số vùng đề xuất: {len(rects)}")


    output = img.copy()

    for i, (x, y, w, h) in enumerate(rects[:100]): 
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 1)
        print(x, y, x+w, y+h)

    cv2.imshow("Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

