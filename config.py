import os.path


# single level anchor box config for VOC and COCO
ANCHOR_SIZE = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]

ANCHOR_SIZE_COCO = [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]]

# multi level anchor box config for VOC and COCO

# yolo_v3
MULTI_ANCHOR_SIZE = [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],
                     [78.4, 201.92], [178.24, 178.56], [129.6, 294.72],
                     [331.84, 194.56], [227.84, 325.76], [365.44, 358.72]]

MULTI_ANCHOR_SIZE_COCO = [[12.48, 19.2], [31.36, 46.4],[46.4, 113.92],
                          [97.28, 55.04], [133.12, 127.36], [79.04, 224.],
                          [301.12, 150.4 ], [172.16, 285.76], [348.16, 341.12]]

# tiny yolo_v3
TINY_MULTI_ANCHOR_SIZE = [[34.01, 61.79], [86.94, 109.68], [93.49, 227.46],
                          [246.38, 163.33], [178.68, 306.55], [344.89, 337.14]]

TINY_MULTI_ANCHOR_SIZE_COCO = [[15.09, 23.25], [46.36, 61.47],[68.41, 161.84],
                               [168.88, 93.59], [154.96, 257.45], [334.74, 302.47]]

IGNORE_THRESH = 0.5

mnist_ab = {
    'num_classes': 10,
    'lr_epoch': (150, 200), # (60, 90, 160),
    'max_epoch': 250,
    'min_dim': [80, 80],
    'name': 'MNIST',
}

voc_ab = {
    'num_classes': 20,
    'lr_epoch': (150, 200), # (60, 90, 160),
    'max_epoch': 250,
    'min_dim': [416, 416],
    'name': 'VOC',
}

coco_ab = {
    'num_classes': 80,
    'lr_epoch': (150, 200), # (60, 90, 160),
    'max_epoch': 260,
    'min_dim': [608, 608],
    'name': 'COCO',
}