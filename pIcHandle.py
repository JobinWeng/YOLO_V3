import os
from PIL import Image

for i in range(3):
    for j in range(3):
        img = Image.open(os.path.join(r"F:\jobin\yolo数据","{}-{}.jpg".format(i,j)))
        w,h =img.size
        sidelen = max(w,h)

        _w = (sidelen-w)/2
        _h = (sidelen-h)/2

        img =img.crop((-_w, -_h, w+_w, h+_h))
        img = img.resize((416,416))
        img.save(os.path.join(r"F:\jobin\yolo数据\数据","{}-{}.jpg".format(i,j)))


