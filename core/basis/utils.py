import numpy as np


def iou(box1: np.ndarray, box2: np.ndarray):
    overlap_a = np.maximum(box1[0], box2[0])
    overlap_b = np.minimum(box1[1], box2[1])
    overlap = np.prod(np.maximum(overlap_b - overlap_a, 0))
    box1_area = np.prod(box1[1] - box1[0])
    box2_area = np.prod(box2[1] - box2[0])
    union_area = box1_area + box2_area - overlap
    res = overlap / union_area
    return res


def kargmax(cls, params, k=1):
    # argmax random on tie using reservoir sampling
    res = 0
    maximum = float('-inf')
    k = 0
    for i, p in enumerate(params):
        if p > maximum:
            maximum = p
            k = 1
            res = i
        elif p == maximum:
            k += 1
            if np.random.random() < 1 / k:
                res = i
    return res
# def iou(coor1: tuple, coor2: tuple, size: tuple):
#     # Assign variable names to coordinates for clarity
#     box1_x1, box1_y1, box1_x2, box1_y2 = coor1[0], coor1[1], coor1[0] + size[0], coor1[1] + size[1]
#     box2_x1, box2_y1, box2_x2, box2_y2 = coor2[0], coor2[1], coor2[0] + size[0], coor2[1] + size[1]
#
#     xi1 = max(box1_x1, box2_x1)
#     yi1 = max(box1_y1, box2_y1)
#     xi2 = min(box1_x2, box2_x2)
#     yi2 = min(box1_y2, box2_y2)
#     inter_width = max(xi2 - xi1, 0)
#     inter_height = max(yi2 - yi1, 0)
#     inter_area = inter_width * inter_height
#
#     box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
#     box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
#     union_area = box1_area + box2_area - inter_area
#     res = inter_area / union_area
#
#     return res
