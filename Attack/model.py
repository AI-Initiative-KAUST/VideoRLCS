import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

class FeatureExtractor(nn.Module):
    """docstring for FeatureExtractor."""

    def __init__(self, input_channel=1,output_channel=256):
        super(FeatureExtractor, self).__init__()
        self.conv_1 = nn.Conv2d(input_channel, 32, kernel_size=3,stride=2)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(32, 64, (3,3), stride=1)
        # self.instance_norm_1 = nn.InstanceNorm2d(64)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(64, 128, (3,3), stride=(2,2))
        self.relu_3 = nn.ReLU(inplace=True)
        self.conv_4 = nn.Conv2d(128, 128, (3,3), stride=(1,1))
        self.instance_norm_2 = nn.InstanceNorm2d(128)
        self.relu_4 = nn.ReLU(inplace=True)
        self.conv_5 = nn.Conv2d(128, output_channel, (2,2), stride=(1,1))
        self.relu_5 = nn.ReLU(inplace=True)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,x):
        b,f,c,h,w = x.size()
        x = x.contiguous().view(b*f,c,h,w)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        # x = self.instance_norm_1(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.conv_4(x)
        x = self.instance_norm_2(x)
        x = self.relu_4(x)
        x = self.conv_5(x)
        x = self.relu_5(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        _,c_out = x.size()
        x = x.contiguous().view(b,f,c_out)
        return x

class Detector_Head(nn.Module):
    """docstring for Detector_Head."""

    def __init__(self, input_channel=256,max_length=200,mode='detector'):
        super(Detector_Head, self).__init__()
        self.max_length = max_length
        self.rnn = nn.LSTM(256,128,bidirectional=True,batch_first=True)
        if mode == 'detector':
            self.linear_1 = nn.Linear(256,1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear_1 = nn.Linear(max_length * 256,3)
        self.mode = mode
    def forward(self, x, lengths):
        x = pack_padded_sequence(x,lengths=lengths,batch_first=True, enforce_sorted=True)
        h0 = torch.zeros(2, x.batch_sizes[0], 128) # 2 for bidirection
        c0 = torch.zeros(2, x.batch_sizes[0], 128)
        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        x,_ = self.rnn(x,(h0,c0))
        x,_ = pad_packed_sequence(x,padding_value=0,total_length=self.max_length, batch_first=True)
        if self.mode == 'detector':
            b,f,c = x.size()
            x = x.contiguous().view(b*f,c)
            x = self.linear_1(x)
            x = self.sigmoid(x)
            x = x.view(b,f)
            x = pack_padded_sequence(x,lengths=lengths,batch_first=True, enforce_sorted=True)
            x,_ = pad_packed_sequence(x,padding_value=0,total_length=lengths[0], batch_first=True)
            return x
        else:
            b,f,c = x.size()
            x = x.contiguous().view(b,f*c)
            x = self.linear_1(x)
            return x


class Detector(nn.Module):
    """docstring for Detector."""

    def __init__(self):
        super(Detector, self).__init__()
        self.feature_network = FeatureExtractor()
        self.head = Detector_Head(mode='detector')
    def forward(self,x,length):
        out = self.feature_network(x)
        return self.head(out,length)


class Predictor(nn.Module):
    """docstring for Predictor."""

    def __init__(self):
        super(Predictor, self).__init__()
        self.feature_network = FeatureExtractor()
        self.head = Detector_Head(mode='predictor')
    def forward(self,x,length):
        out = self.feature_network(x)
        return self.head(out,length)
