import numpy as np

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    # rows行k列，每行对应一个boxes，每列是boxes与各个质心的距离
    distances = np.empty((rows, k))

    last_clusters = np.zeros((rows,))

    np.random.seed()
    # 在boxes中随机选取k个作为质心clusters
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    print("distances",distances.shape)
    print("rows",rows)
    print("clusters",clusters.shape)

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        # 找到与boxes距离最近的clusters的索引
        nearest_clusters = np.argmin(distances, axis=1)
        print("nearest_clusters",nearest_clusters)

        # 当两次聚类结果相同时结束
        if (last_clusters == nearest_clusters).all():
            break

        # 分别取w和h的平均值更新clusters
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters
