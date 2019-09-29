import torch

IMG_PATH = r'../YOLO_DATA'
LABEL_FILE = r'../YOLO_DATA/lables.json'
MODEL_PATH = './model/yolo.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_HIGHT = 416
IMG_WIDTH = 416

LABEL_CLASS=['car','cat','dog']

CLASS_NUM = len(LABEL_CLASS)

ANCHORS_GROUP = {
    13: [[350, 150], [250, 250], [150, 350]],
    26: [[300, 120], [168, 170], [120, 300]],
    52: [[160, 100], [100, 100], [100, 160]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}