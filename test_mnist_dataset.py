"""
Adapted from:
    @ Viet Nguyen's utils.py
    @ author: Viet Nguyen <nhviet1009@gmail.com>

    modified by Jongwan Kim (jongwankim@snu.ac.kr)
"""
import time
import os
import argparse
import glob
import shutil
import cv2
import numpy as np
from src.utils import *
import pickle
from src.tiny_yolo_net import Yolo
from PIL import Image
from src.mnist_dataset import mnistDataset
from torch.utils.data import DataLoader
from src.loss import YoloLoss
import datetime
from tensorboardX import SummaryWriter


CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=80, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--test_set", type=str, default="test",
                        help="For both VOC2007 and 2012, you could choose 3 different datasets: train, trainval and val. Additionally, for VOC2007, you could also pick the dataset name test")
    parser.add_argument("--year", type=str, default="2020")
    parser.add_argument("--data_path", type=str, default="/home/jongwan0317/dataset/detection_MNISTtest/MNISTdevkit", help="the root folder of dataset")
    parser.add_argument("--pretrained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pretrained_model_path", type=str, default="./trained_models")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=5000, help="The number of images per batch")

    args = parser.parse_args()
    return args

def test(opt):
    input_list_path = os.path.join(opt.data_path, "MNIST{}".format(opt.year), "ImageSets/Main/{}.txt".format(opt.test_set))
    image_ids = [id.strip() for id in open(input_list_path)]
    output_folder = os.path.join(opt.output, "MNIST{}_{}".format(opt.year, opt.test_set))
    colors = pickle.load(open("src/pallete", "rb"))
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    pre_path = os.path.join(opt.pretrained_model_path, "epochs100, b512")
    premodel_path = os.path.join(opt.pretrained_model_path, "epochs100, b512", "model.pt")

    if torch.cuda.is_available():
        if os.path.isdir(pre_path):
            print("using pre-trained model")
            model = torch.load(premodel_path)
            # model.load_state_dict(prestate_path)
        else:
            print("There is a problem with loading pretrained model")


    model.eval()

    for id in image_ids:
        image_path = os.path.join(opt.data_path, "MNIST{}".format(opt.year), "PNGImages", "{}.png".format(id))
       # image = cv2.imread(image_path)
       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height = 80
        width = 80
        image_pil = Image.open(image_path, 'r')
        image = np.array(image_pil)
        image = image.reshape(opt.image_size, opt.image_size, 1)
#        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))

        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) == 0:
            continue
        else:
            predictions = predictions[0]
        output_image = cv2.imread(image_path)
        for pred in predictions:
            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            color = colors[CLASSES.index(pred[5])]
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 1)
            text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 0.5, 1)[0]
            cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0], ymin + text_size[1] + 1), color, -1)
            cv2.putText(
                output_image, pred[5] + ' : %.2f' % pred[4],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                (255, 255, 255), 1)
            print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))

        f = open("/home/jongwan0317/dataset/detection_MNISTtest/MNISTdevkit/MNIST2020/txt/{}.txt".format(id), 'a')
        data = '{} {} {} {} {} {}'.format(pred[5], pred[4], xmin, ymin, xmax, ymax)
        f.write(data)
        f.close()

        cv2.imwrite("{}/{}.png".format(output_folder, id), output_image)



if __name__ == "__main__":
    opt = get_args()
    test(opt)
