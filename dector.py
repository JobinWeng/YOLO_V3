import torch
import torch.nn as nn
import cfg
from utils import nms
import torchvision
from PIL import Image, ImageDraw
import NET

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class Detector(nn.Module):
    def __init__(self,netPath = cfg.MODEL_PATH):
        super().__init__()

        self.net = NET.MainNet(cfg.CLASS_NUM).to(cfg.device)
        self.net.load_state_dict(torch.load(netPath))
        self.net.eval()

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parase(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parase(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parase(idxs_52, vecs_52, 8, anchors[52])

        boxes_all = torch.cat([boxes_13, boxes_26, boxes_52],dim=0)

        last_boxes = []
        #0: 第几张图片
        #1：第几个框
        #2：框的坐标
        for n in range(input.size(0)):
            n_boxes=[]
            boxes_n = boxes_all[boxes_all[:,6] == n]
            for cls in range(cfg.CLASS_NUM):
                boxes_c = boxes_n[boxes_n[:,5] == cls]
                if boxes_c.size(0) > 0:
                    n_boxes.extend(nms(boxes_c, 0.3))
                else:
                    pass

            last_boxes.append(torch.stack(n_boxes))

        return last_boxes


    def _filter(self, output, thresh):
        output = output.permute(0,2,3,1)
        output = output.reshape(output.shape[0],output.shape[1],output.shape[2],3,-1)

        output = output.cpu()

        #计算置信度损失
        torch.sigmoid_(output[...,4])
        torch.sigmoid_(output[...,0:2])

        mask = output[...,4] > thresh
        idxs = mask.nonzero()
        vecs = output[mask]

        return idxs, vecs

    def _parase(self, idxs, vecs, t, anchors):
        if idxs.size(0) == 0:
            return torch.Tensor([])

        anchors = torch.Tensor(anchors)

        n = idxs[:,0]  #所属图片
        a = idxs[:,3]  #建议框
        c = vecs[:,4]  #置信度

        cls = torch.argmax(vecs[:,5:],dim=1)

        cy = (idxs[:,1].float() + vecs[:,1])*t
        cx = (idxs[:,2].float() + vecs[:,0])*t

        w = anchors[a,0] * torch.exp(vecs[:,2])
        h = anchors[a,1] * torch.exp(vecs[:,3])

        w0_5, h0_5 = w/2 ,h/2

        x1, y1, x2, y2 = cx - w0_5, cy - h0_5, cx + w0_5, cy + h0_5

        return torch.stack([x1, y1, x2, y2, c, cls.float(), n.float()],dim=1)


if __name__ == '__main__':
    detect = Detector()
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

    img = Image.open(r"F:\jobin\YOLO_DATA\2-0.jpg")
    imgData = transforms(img)

    with torch.no_grad():
        boxes = detect(imgData.unsqueeze(dim=0).to(device),0.65,cfg.ANCHORS_GROUP)
        imDraw = ImageDraw.Draw(img)

        for box in boxes:
            box = box.data.numpy()
            print(box)
            for data in box:
                imDraw.rectangle((data[0],data[1],data[2],data[3]), outline='red')
                imDraw.text((data[0], data[1]), "con:{},cls:{}".format(str(data[4]),cfg.LABEL_CLASS[int(data[5])]))
        img.show()









