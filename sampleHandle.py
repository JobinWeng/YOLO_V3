import os
import json
import k_means
import numpy as np

LABEL_PATH = r"F:\jobin\YOLO_DATA\outputs"

files = os.listdir(LABEL_PATH)

labels = []
areas = []
for file in files:
    f = open(os.path.join(LABEL_PATH,file))
    text = json.load(f)
    label = {}
    fileName = text["path"].strip().split('\\')

    label['fileName'] = fileName[-1]
    label['object'] = []

    objects = text['outputs']['object']

    for i, object in enumerate(objects):
        text = {}
        cla = object['name']
        position = object['bndbox']['xmin'],object['bndbox']['ymin'],object['bndbox']['xmax'],object['bndbox']['ymax']
        x_c = (position[0]+position[2])//2
        y_c = (position[1]+position[3])//2
        w = position[2]-position[0]
        h = position[3]-position[1]

        area=  w, h
        areas.append(area)

        text['position']=x_c,y_c,w,h
        text['cls'] = object['name']
        label['object'].append(text)
        labels.append(label)

areas = np.array(areas)
# print("areas",areas)

anchor = k_means.kmeans(areas,3)
print("anchor:",anchor)
# sortArea = sorted(areas,key=lambda x:x[0])
# sortArea_0 = sortArea[:4]
# sortArea_1 = sortArea[4:8]
# sortArea_2 = sortArea[8:]
# print("sortArea_0",sortArea_0)
# print("sortArea_1",sortArea_1)
# print("sortArea_2",sortArea_2)



# with open(os.path.join(r"F:\YOLO\yolo数据\outputs",'lables.json'), 'w') as f:
#     json.dump(labels, f)






