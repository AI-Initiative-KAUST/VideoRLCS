# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import sys
import time
import argparse
import datetime
import copy
import itertools
from tqdm import *
from utils import *

from dataset import Pong_dataset, collate_fn
from train import train_detector,test_detector
from model import Detector, Predictor

parser = argparse.ArgumentParser(description='PyTorch Subgoal Training')
parser.add_argument('--l1_weight', default=1e-2, type=float,
                    help='L1 regularization')
# parser.add_argument('--var_weight', default=1e-2, type=float,
#                     help='Var regularization')
parser.add_argument('--classify_weight', default=1, type=float,
                    help='formal classification loss')
parser.add_argument('--reverse_weight', default=2, type=float,
                    help='reverse classification loss')
parser.add_argument('--data_dir', default='./Datasets/Pong_Dataset_single_round/', type=str,
                    help='training dir')
parser.add_argument('--max_Epoch',default=60, type=int, help='The maximum Epochs for learn')
parser.add_argument('--training_num',default=21500, type=int, help='The maximum training samples')
parser.add_argument('--batch_size', default=64, type=int, help='The batch_size for training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
parser.add_argument("--save_dir",type=str, default='./Weight/')
parser.add_argument("--frame_length",default=12, type=int, help='the frame length')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',)
parser.add_argument('--random_length', default=50, type=int,
                    help='random length  (default: 50)',)
parser.add_argument('--save_model_every_n_steps',default=1, type=int, help='The frequency for saving model')
args = parser.parse_args()


if __name__ == '__main__':
    make_dir(args)

    train_loader = DataLoader(
            Pong_dataset(path = args.data_dir,sample_num=args.training_num,split='train'),
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            drop_last = False,
            collate_fn=collate_fn
    )
    test_loader = DataLoader(
                    Pong_dataset(path = args.data_dir,sample_num=2000,split='test'),
                    batch_size = args.batch_size,
                    shuffle = False,
                    num_workers = 2,
                    pin_memory = True,
                    drop_last = False,
                    collate_fn=collate_fn
    )
    detector = Detector()
    predictor = Predictor()

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        detector.cuda()
        predictor.cuda()
        criterion.cuda()

    optimizer_detector = torch.optim.Adam(detector.parameters(), args.lr,
                            weight_decay=args.weight_decay)

    optimizer_predictor = torch.optim.Adam(predictor.parameters(), args.lr,
                            weight_decay=args.weight_decay)


    for epoch in range(args.max_Epoch):
        train_detector(args,
                                train_loader,
                                criterion,
                                optimizer_detector,
                                optimizer_predictor,
                                epoch,
                                detector,
                                predictor,
                                )
        acc_masked, acc_r,acc_normal = test_detector(args,
                                test_loader,
                                detector,
                                predictor,
                                )

        if epoch % args.save_model_every_n_steps == 0:
            save_checkpoint(
                state = detector.state_dict(), is_best=False, step = epoch, args=args,name='detector_'
            )


            save_checkpoint(
                state = predictor.state_dict(), is_best=False, step = epoch, args=args,name='predictor_'
            )
