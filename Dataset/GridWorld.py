# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
from torch.nn.utils.rnn import pad_sequence
import os



class gridworld_positive_negative(Dataset):
    """docstring for gridworld_positive_negative."""

    def __init__(self, path,frame_unit_length=12):
        super(gridworld_positive_negative, self).__init__()
        self.item_list = []
        self.label_list = []
        for root_path, ds, fs in os.walk(path):
            for f in fs:
                if f.endswith('.npy'):
                    full_name = os.path.join(root_path,f)
                    if 'negative' in full_name:
                        label = 1
                    elif 'positive' in full_name:
                        label = 0
                    else:
                        break
                    frames = np.load(full_name)
                    if frame_unit_length >0:
                        for ind in range(frames.shape[0]-frame_unit_length):
                            self.item_list.append(frames[ind:ind+frame_unit_length])
                            self.label_list.append(label)
                    else:
                        self.item_list.append(frames)
                        self.label_list.append(label)

    def __getitem__(self,index):
        frames, label = self.item_list[index], self.label_list[index]
        frames = np.transpose(frames,[0,3,1,2])
        frames = frames / 256.
        return frames, label
    def __len__(self):
        return len(self.item_list)


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    video_list = []
    label_list = []
    offsets_list = []
    for video, label in data:
        video_list.append(torch.tensor(video))
        offsets_list.append(len(video))
        label_list.append(label)
    video_tensor = pad_sequence(video_list, batch_first=True, padding_value=0)
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    offsets_tensor = torch.tensor(offsets_list)
    return video_tensor, offsets_tensor, label_tensor