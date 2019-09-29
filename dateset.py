from torch.utils.data import Dataset
from PIL import Image,ImageDraw
import numpy as np
import torchvision
import cfg
import json
import math
import os

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self):
        with open(cfg.LABEL_FILE,'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        datas = {}
        img = Image.open(os.path.join(cfg.IMG_PATH,self.labels[index]["fileName"]))
        img_data = transforms(img)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            datas[feature_size] = np.zeros(shape=(feature_size,feature_size,3,6),dtype=np.float32)

            for label in self.labels[index]["object"]:
                #分类
                for i, cl in enumerate(cfg.LABEL_CLASS):
                    if cl == label["cls"]:
                        cls = i
                        break
                #索引和偏移量
                cx_offset, cx_index = math.modf(label["position"][0] * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(label["position"][1] * feature_size / cfg.IMG_WIDTH)



                #图片查看
                # img_draw =ImageDraw.Draw(img)
                # img_draw.rectangle((label["position"][0]-label["position"][2]//2, label["position"][1]-label["position"][3]//2,
                #                     label["position"][0]+label["position"][2]//2, label["position"][1]+label["position"][3]//2), outline='red')
                # img.show()

                #3个建议框
                for i,anchor in enumerate(anchors):
                    w, h = label["position"][2],label["position"][3]
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w,p_h = w/anchor[0] ,h/anchor[1]
                    box_area = w*h

                    # 计算置信度(同心框的IOU(交并))
                    inter = np.minimum(w,anchor[0])*np.minimum(h,anchor[1])
                    conf = inter/(box_area+anchor_area-inter)

                    datas[feature_size][int(cy_index),int(cx_index),i] = np.array([
                        cx_offset, cy_offset, np.log(p_w), np.log(p_h), conf, cls
                    ])

                    # print("feature_size",feature_size,datas[feature_size][int(cy_index),int(cx_index),i])


        return datas[13],datas[26],datas[52],img_data.to(cfg.device)


if __name__ == '__main__':
    dataset = MyDataset()
    for i in range(10):
        print(dataset[i][0][...,0].shape)
