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

class Atari_dataset(Dataset):
    """docstring for toy_dataset."""

    def __init__(self, path, sample_num=20000,split='train',sample_length=400,norm_scale=1000,random_length=30):
        super(Atari_dataset, self).__init__()
        self.item_list = []
        self.label_list = []
        self.length = sample_length
        self.norm_scale = norm_scale
        self.random_length = random_length
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
        length = self.length + int(np.random.rand(1)[0]-0.5)*self.random_length 
        frames, label = self.item_list[index], self.label_list[index]
        frames = np.load(frames)
        init_index = np.random.randint(0, len(frames)-1-length)
        frames = frames[init_index:init_index+length] / 255. 
        frames[:,:,:10,:] = np.zeros_like(frames[:,:,:10,:])
        label = np.load(label)
        label = np.sum(label[init_index:init_index+length])/ self.norm_scale
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
    label_tensor = torch.tensor(label_list, dtype=torch.float64)
    offsets_tensor = torch.tensor(offsets_list)
    return video_tensor, offsets_tensor, label_tensor