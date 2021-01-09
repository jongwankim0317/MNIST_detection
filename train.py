"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import os
import argparse
import torch.nn as nn
import numpy as np
import shutil
import time
import datetime
from src.dataset import MNISTdataset
from src.utils import *
from src.loss import YoloLoss
from src.model import Yolo
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

classes = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=80,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="The number of images per batch")
    parser.add_argument("--train_dataset_size", type=int, default=50000,
                        help="The number of train_dataset")
    parser.add_argument("--test_dataset_size", type=int, default=10000,
                        help="The number of test_dataset")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=10,
                        help="Number of epoches between testing phases")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--noobject_scale", type=float, default=0.5)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--coord_scale", type=float, default=5.0)
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=50,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. "
                             "Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="train")
    parser.add_argument("--test_set", type=str, default="test")
    parser.add_argument("--data_path", type=str, default="/home/jongwan0317/dataset/MNIST2020/MNIST2020_train/MNISTdevkit/MNIST2020/",
                        help="the root folder of dataset")
    parser.add_argument("--pretrained_model_type", type=str, choices=["model", "state", "None"], default="state")
    parser.add_argument("--pretrained_model_path", type=str, default="./trained_models/model")
    parser.add_argument("--pretrained_state_path", type=str, default="./trained_models/state")
    parser.add_argument("--log_path", type=str, default="./tensorboard/yolo_mnist")
    parser.add_argument("--save_dir", type=str, default="./trained_models")

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

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    learning_rate_schedule = {"0": 1e-3, "5": 1e-4,
                              "80": 1e-5, "110": 1e-5}
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    training_set = MNISTdataset(opt.data_path, opt.train_set, opt.image_size, opt.train_dataset_size)
    training_generator = DataLoader(training_set, **training_params)

    testing_set = MNISTdataset(opt.data_path, opt.test_set, opt.image_size, opt.test_dataset_size)
    test_generator = DataLoader(testing_set, **test_params)


    if torch.cuda.is_available():
        if opt.pretrained_model_type == "model":
            print("Using pre-trained model")
            _model = recent_file(opt.pretrained_model_path)
            model = torch.load(_model)
            print("Load model from : ", _model)
        elif opt.pretrained_model_type == "state":
            print("Using pre-trained model_state")
            model = Yolo(training_set.num_classes)
            _state = recent_file(opt.pretrained_state_path)
            model.load_state_dict(torch.load(_state))
            print("Load model_state from : ", _state)
        else:
            model = Yolo(training_set.num_classes)
            print("Making a new model")
    else:
        print("You had batter use GPU")

    # The following line will re-initialize weight for the last layer, which is useful
    # when you want to retrain the model based on my trained weights. if you uncomment it,
    # you will see the loss is already very small at the beginning.

    start = time.time()
    nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    log_paths = os.path.join(opt.log_path, timestamp)
    print('=> Log data will be saved to {}'.format(log_paths))

    if os.path.isdir(log_paths):
        shutil.rmtree(log_paths)
    os.makedirs(log_paths)
    writer = SummaryWriter(log_paths)

    if torch.cuda.is_available():
        writer.add_graph(model.cpu(), torch.rand(opt.batch_size, 1, opt.image_size, opt.image_size))
        model.cuda()
    else:
        writer.add_graph(model, torch.rand(opt.batch_size, 1, opt.image_size, opt.image_size))
    criterion = YoloLoss(training_set.num_classes, model.anchors, opt.reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=opt.momentum, weight_decay=opt.decay)
    best_loss = 1e10
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(opt.num_epoches):
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]
        for iter, batch in enumerate(training_generator):
            image, label = batch
            if torch.cuda.is_available():
                image = Variable(image.cuda(), requires_grad=True)
            else:
                image = Variable(image, requires_grad=True)
            optimizer.zero_grad()
            logits = model(image)
            loss, loss_coord, loss_conf, loss_cls = criterion(logits, label)
            loss.backward()
            optimizer.step()
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                loss_coord,
                loss_conf,
                loss_cls))
            writer.add_scalar('Train/Total_loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Coordination_loss', loss_coord, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Confidence_loss', loss_conf, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Class_loss', loss_cls, epoch * num_iter_per_epoch + iter)

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            loss_coord_ls = []
            loss_conf_ls = []
            loss_cls_ls = []
            for te_iter, te_batch in enumerate(test_generator):
                te_image, te_label = te_batch
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_image = te_image.cuda()
                with torch.no_grad():
                    te_logits = model(te_image)
                    batch_loss, batch_loss_coord, batch_loss_conf, batch_loss_cls = criterion(te_logits, te_label)
                loss_ls.append(batch_loss * num_sample)
                loss_coord_ls.append(batch_loss_coord * num_sample)
                loss_conf_ls.append(batch_loss_conf * num_sample)
                loss_cls_ls.append(batch_loss_cls * num_sample)
            te_loss = sum(loss_ls) / testing_set.__len__()
            te_coord_loss = sum(loss_coord_ls) / testing_set.__len__()
            te_conf_loss = sum(loss_conf_ls) / testing_set.__len__()
            te_cls_loss = sum(loss_cls_ls) / testing_set.__len__()
            print("Epoch: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss,
                te_coord_loss,
                te_conf_loss,
                te_cls_loss))
            writer.add_scalar('Test/Total_loss', te_loss, epoch)
            writer.add_scalar('Test/Coordination_loss', te_coord_loss, epoch)
            writer.add_scalar('Test/Confidence_loss', te_conf_loss, epoch)
            writer.add_scalar('Test/Class_loss', te_cls_loss, epoch)

            model.train()

            model_save_dir = os.path.join(opt.save_dir, "model")
            state_save_dir = os.path.join(opt.save_dir, "state")
            print('=> Training model will be saved to {}'.format(model_save_dir))
            print('=> Training model state will be saved to {}'.format(state_save_dir))

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            if not os.path.exists(state_save_dir):
                os.makedirs(state_save_dir)

            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch

#           torch.save(model, model_save_dir + os.sep + 'model_{}.pt'.format(timestamp))
#           torch.save(model.state_dict(), state_save_dir + os.sep + 'model_state_dict_{}.pt'.format(timestamp))
#           torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
#                           opt.save_dir + os.sep + 'all_{}.tar'.format(timestamp))

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break

    writer.export_scalars_to_json(log_paths + os.sep + "all_logs.json")
    writer.close()
    print("time :", time.time() - start)

    torch.save(model, model_save_dir + os.sep + 'model_{}.pt'.format(timestamp))
    torch.save(model.state_dict(), state_save_dir + os.sep + 'model_state_dict_{}.pt'.format(timestamp))
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
               opt.save_dir + os.sep + 'all_{}.tar'.format(timestamp))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
