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

from dataset import Atari_dataset, collate_fn
from train import train_epoch,test_epoch
from model import Detector,Predictor

parser = argparse.ArgumentParser(description='PyTorch Subgoal Training')
parser.add_argument('--l1_weight', default=1e-2, type=float,
                    help='L1 regularization')
# parser.add_argument('--var_weight', default=1e-2, type=float,
#                     help='Var regularization')
parser.add_argument('--classify_weight', default=1, type=float,
                    help='formal classification loss')
parser.add_argument('--reverse_weight', default=2, type=float,
                    help='reverse classification loss')
parser.add_argument('--data_dir', default='/ibex/scratch/liuh0g/Datasets/Pong_Dataset_single_round/', type=str,
                    help='training dir')
parser.add_argument('--max_Epoch',default=60, type=int, help='The maximum Epochs for learn')
parser.add_argument('--training_num',default=21500, type=int, help='The maximum training samples')
parser.add_argument('--batch_size', default=64, type=int, help='The batch_size for training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
parser.add_argument("--save_dir",type=str, default='./Weight/toy_dataset/test')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',)
parser.add_argument('--random_length', default=0, type=int,
                    help='random length  (default: 0)',)
parser.add_argument('--sample_length', default=200, type=int,
                    help='sample length  (default: 200)',)
parser.add_argument('--num_workers', default=4, type=int,
                    help='num workers  (default: 4)',)
parser.add_argument('--norm_scale', default=1000, type=float,
                    help='norm scale for accumulative rewards  (default: 1000)',)
parser.add_argument('--orthogonal_weight', default=5e-4, type=float,
                    help='orthogonal for mask  (default: 5e_4)',)
parser.add_argument('--save_model_every_n_steps',default=1, type=int, help='The frequency for saving model')
args = parser.parse_args()


if __name__ == '__main__':
    make_dir(args)

    train_loader = DataLoader(
            Atari_dataset(  path = args.data_dir,
                            sample_num=args.training_num,
                            split='train',
                            sample_length=args.sample_length,
                            norm_scale=args.norm_scale,
                            random_length=args.random_length),
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = args.num_workers,
            pin_memory = False,
            drop_last = False,
            collate_fn=collate_fn
    )
    test_loader = DataLoader(
                    Atari_dataset(path = args.data_dir,
                                  sample_num=2000,
                                  split='test',
                                  sample_length=args.sample_length,
                                  norm_scale=args.norm_scale,
                                  random_length=0),
                    batch_size = args.batch_size,
                    shuffle = False,
                    num_workers = args.num_workers,
                    pin_memory = False,
                    drop_last = False,
                    collate_fn=collate_fn
    )
    critical_state_detector = Detector()
    return_predictor = Predictor()

    criterion =  nn.SmoothL1Loss()
    if torch.cuda.is_available():
        critical_state_detector.cuda()
        return_predictor.cuda()
        criterion.cuda()

    optimizer_subgoal = torch.optim.Adam(critical_state_detector.parameters(), args.lr,
                            weight_decay=args.weight_decay)

    optimizer_trajectory = torch.optim.Adam(return_predictor.parameters(), args.lr,
                            weight_decay=args.weight_decay)


    for epoch in range(args.max_Epoch):
        train_epoch(args,
                                train_loader,
                                criterion,
                                optimizer_subgoal,
                                optimizer_trajectory,
                                epoch,
                                critical_state_detector,
                                return_predictor,
                                )
        acc_masked, acc_r,acc_normal = test_epoch(args,
                        test_loader,
                        criterion,
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
