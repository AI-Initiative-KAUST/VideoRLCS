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


class Pong_dataset(Dataset):
    """docstring for toy_dataset."""

    def __init__(self, path, sample_num=21500,split='train'):
        super(Pong_dataset, self).__init__()
        self.item_list = []
        self.label_list = []
        if split == 'train':
            path = os.path.join(path,'train')
        if split == 'test':
            path = os.path.join(path,'test')

        for i in range(sample_num):
            frames = os.path.join(path,str(i)+'.npy')
            labels = os.path.join(path,str(i)+'_r.npy')
            self.item_list.append(frames)
            self.label_list.append(labels)

    def __getitem__(self,index):
        frames, label = self.item_list[index], self.label_list[index]
        frames = np.load(frames)
        label = int(np.load(label)[0]) + 1
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