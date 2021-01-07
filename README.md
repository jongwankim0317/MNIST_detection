# [PYTORCH] YOLO (You Only Look Once)

## Introduction
Here is my pytorch implementation of the model described in the paper **YOLO9000: Better, Faster, Stronger** [paper](https://arxiv.org/abs/1612.08242). 

# Data generation
implement data/mnist/generate_data_xml.py 
num-train-images", default=50000, type=int
num-test-images", default=10000, type=int

# Train
implement /src/tiny_yolo_net.py

# mAP (mean Average Precision)
This code will evaluate the performance of your neural net for object recognition.
In practice, a **higher mAP** value indicates a **better performance**
## Citation

This project was developed for the following paper, please consider citing it:

"""code Adapted from:
    @ longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @ rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    @ SidHard https://github.com/SidHard/eval_mAP
    @ adapted the official Matlab code into Python
"""
