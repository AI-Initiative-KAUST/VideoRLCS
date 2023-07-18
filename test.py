# -*- coding: utf-8 -*-
import sys
from Dataset.data_generation import SimpleKeyCorridor,collect_positive_data
from Models.model import Detector
import numpy as np
import torch
from utils import tensor_to_np
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import *

parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')

parser.add_argument('--model_path', default='./', type=str,
                    help='the path for the pretrained model weight')

parser.add_argument('--test_num', default=100, type=int,
                    help='Test Number (Environments)')

parser.add_argument('--tole_thres', default=1, type=int,
                    help='the tole_thres for calculating F1 score')
args = parser.parse_args()



output_list = []
ground_truth_list = []
for repeat in tqdm(range(args.test_num)):
    frames_partial, frames_full, label = collect_positive_data(True)

    frames_partial = np.array(frames_partial)
    
    length = torch.tensor([frames_partial.shape[0]])

    test_data = frames_partial
    test_data = np.transpose(test_data,[0,3,1,2])
    test_data = test_data / 255.
    test_data = torch.tensor(test_data).float()
    test_data = pad_sequence([test_data], batch_first=True, padding_value=0)

    detector = Detector()
    state_dict = torch.load(args.model_path)
    detector.load_state_dict(state_dict)
    output = detector(test_data,length)
    output = tensor_to_np(output)[0]


    output_list.append(output)
    ground_truth_list.append(label)


def evaluate(outputs,labels,threshold,tolerance=2):
    tp = 1e-8
    fp = 1e-8
    fn = 1e-8
    tn = 1e-8
    for ind,output in enumerate(outputs):
        label = labels[ind]
        for ind_in_traj in range(len(label)):
            lower_ind = max(0,ind_in_traj-tolerance)
            upper_ind = min(len(label),ind_in_traj+tolerance+1)
            if output[ind_in_traj]<threshold:
                if label[ind_in_traj] == 0: 
                    tn += 1
                else:
                    fn += 1
            else:
                if np.mean(label[lower_ind:upper_ind]) > 0: 
                    tp += 1
                else:
                    fp += 1
    return tp/(tp+fp), tp/(tp+fn)

F1_optimal = 0
for t in tqdm(range(11)):
    precision,recall = evaluate(output_list,ground_truth_list,t/10.,args.tole_thres)
    F1 = 2*precision*recall/(precision+recall)
    if F1 > F1_optimal:
        F1_optimal = F1
print(F1_optimal)