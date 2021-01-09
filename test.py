"""
Adapted from:
    @ Viet Nguyen's utils.py
    @ author: Viet Nguyen <nhviet1009@gmail.com>
    modified by Jongwan Kim (jongwankim@snu.ac.kr)
"""
import time
import datetime
import os
import os.path as osp
import argparse
import shutil
import cv2
import numpy as np
from src.utils import *
import pickle
from src.model import Yolo
from PIL import Image
import datetime


CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--model_num", type=int, default=1, help='User defines the model number')
    parser.add_argument("--image_size", type=int, default=80, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--test_set", type=str, default="test")
    parser.add_argument("--year", type=str, default="2020")
    parser.add_argument("--data_path", type=str, default="/home/jongwan0317/dataset/MNIST2020/MNIST2020_test/MNISTdevkit", help="the root folder of dataset")
    parser.add_argument("--pretrained_model_type", type=str, choices=["model", "state"], default="model")
    parser.add_argument("--pretrained_model_path", type=str, default="./trained_models/model/")
    parser.add_argument("--pretrained_state_path", type=str, default="./trained_models/state")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=100, help="The number of images per batch")

    args = parser.parse_args()
    return args


def recent_file(input_path):
    each_file_path_and_gen_time = []
    for each_file_name in os.listdir(input_path):
        each_file_path = os.path.join(input_path, each_file_name)
        each_file_gen_time = os.path.getctime(each_file_path)
        each_file_path_and_gen_time.append((each_file_path, each_file_gen_time))
    most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]
    return most_recent_file

def test(opt):
    input_list_path = os.path.join(opt.data_path, "MNIST{}".format(opt.year), "ImageSets/Main/{}.txt".format(opt.test_set))
    image_ids = [id.strip() for id in open(input_list_path)]
    output_folder = os.path.join(opt.output, "MNIST{}_{}".format(opt.year, opt.model_num))
    colors = pickle.load(open("src/pallete", "rb"))
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}


    each_file_path_and_gen_time = []
    for each_file_name in os.listdir(opt.pretrained_model_path):
        each_file_path = opt.pretrained_model_path + each_file_name
        each_file_gen_time = os.path.getctime(each_file_path)
        each_file_path_and_gen_time.append(
            (each_file_path, each_file_gen_time)
        )
    most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]


    if torch.cuda.is_available():
        if opt.pretrained_model_type == "model":
            print("Using pre-trained model")
            _model = recent_file(opt.pretrained_model_path)
            model = torch.load(_model)
            print("Load model from : ", _model)
        elif opt.pretrained_model_type == "state":
            print("Using pre-trained model_state")
            model = Yolo(10)
            _state = recent_file(opt.pretrained_state_path)
            model.load_state_dict(torch.load(_state))
        else:
            print("There is no pretrained model..You had better train a model before testing")

    else:
        print("You had batter use GPU")



    model.eval()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_result = osp.join(opt.data_path, "MNIST2020", "prediction_model{}".format(opt.model_num), "{}".format(timestamp))

    if not osp.exists(save_result):
        os.makedirs(save_result)

    for id in image_ids:
        image_path = os.path.join(opt.data_path, "MNIST{}".format(opt.year), "PNGImages", "{}.png".format(id))
        #image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height = 80
        width = 80
        image_pil = Image.open(image_path, 'r')
        image = np.array(image_pil)
        image = image.reshape(opt.image_size, opt.image_size, 1)
        #image = cv2.resize(image, (opt.image_size, opt.image_size))
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
            xmin = int(xmin + (xmax-xmin) / 2)
            xmax = int(xmax + (xmax-xmin) / 2)
            ymin = int(ymin + (ymax - ymin) / 2)
            ymax = int(ymax + (ymax - ymin) / 2)

            color = colors[CLASSES.index(pred[5])]
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 1)
            text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 0.5, 1)[0]
            cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0], ymin + text_size[1]), color, -1)
            cv2.putText(
                output_image, pred[5] + ' : %.2f' % pred[4],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)

            print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
        data = '{} {} {} {} {} {}'.format(pred[5], pred[4], xmin, ymin, xmax, ymax)
        file_path = osp.join(save_result, "{}.txt".format(id))
        f = open(file_path, "w")
        f.write(data)
        f.close()
        cv2.imwrite("{}/{}.png".format(output_folder, id), output_image)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
