import torch


# IOU
# (x1,y1,x2,y2,c) 格式
def iou(box, boxes, mode="inter"):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter)
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)


# 非极大值抑制
# (x1,y1,x2,y2,c) 格式
def nms(boxes, thresh, mode='inter'):
    keep_boxes = []
    if boxes.size(0) == 0:
        return keep_boxes
    args = boxes[:, 4].argsort(descending=True)
    sort_boxes = boxes[args]

    while len(sort_boxes) > 0:
        _box = sort_boxes[0]
        keep_boxes.append(_box)

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]
            _iou = iou(_box, _boxes, mode)
            sort_boxes = _boxes[_iou < thresh]
        else:
            break

    return keep_boxes
