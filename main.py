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

from train import train_epoch,test_epoch
from Dataset.GridWorld import gridworld_positive_negative, collate_fn
from Models.model import Detector,Predictor

parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')
parser.add_argument('--l1_weight', default=1e-2, type=float,
                    help='Compactness regularization')
parser.add_argument('--classify_weight', default=1, type=float,
                    help='formal classification loss')
parser.add_argument('--reverse_weight', default=2, type=float,
                    help='reverse classification loss')
parser.add_argument('--train_dir', default='./GridWorld/toy_dataset/train', type=str,
                    help='training dir')
parser.add_argument('--test_dir', default='./GridWorld/toy_dataset/test', type=str,
                    help='test dir')
parser.add_argument('--max_Epoch',default=20, type=int, help='The maximum Epochs for learn')
parser.add_argument('--batch_size', default=64, type=int, help='The batch_size for training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
parser.add_argument("--save_dir",type=str, default='./Weight/toy_dataset/test')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',)
parser.add_argument('--save_model_every_n_steps',default=1, type=int, help='The frequency for saving model')
args = parser.parse_args()


if __name__ == '__main__':
    make_dir(args)

    train_loader = DataLoader(
            gridworld_positive_negative(path = args.train_dir,frame_unit_length = -1),
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            drop_last = False,
            collate_fn=collate_fn,
    )
    test_loader = DataLoader(
                    gridworld_positive_negative(path = args.test_dir,frame_unit_length = -1),
                    batch_size = args.batch_size,
                    shuffle = False,
                    num_workers = 2,
                    pin_memory = True,
                    drop_last = False,
                    collate_fn = collate_fn,
    )
    critical_state_detector = Detector()
    return_predictor = Predictor()




    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        critical_state_detector.cuda()
        return_predictor.cuda()
        criterion.cuda()

    optimizer_detector = torch.optim.Adam(critical_state_detector.parameters(), args.lr,
                            weight_decay=args.weight_decay)

    optimizer_predictor = torch.optim.Adam(return_predictor.parameters(), args.lr,
                            weight_decay=args.weight_decay)


    for epoch in range(args.max_Epoch):
        train_epoch(args,
                                train_loader,
                                criterion,
                                optimizer_detector,
                                optimizer_predictor,
                                epoch,
                                critical_state_detector,
                                return_predictor,
                                )
        acc_masked, acc_r,acc_normal = test_epoch(args,
                                test_loader,
                                critical_state_detector,
                                return_predictor,
                                )
        if epoch % args.save_model_every_n_steps == 0:
            save_checkpoint(
                state = critical_state_detector.state_dict(), is_best=False, step = epoch, args=args,name='detector_'
            )


            save_checkpoint(
                state = return_predictor.state_dict(), is_best=False, step = epoch, args=args,name='predictor_'
            )
