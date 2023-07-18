# -*- coding: utf-8 -*-
from Dataset.data_generation import SimpleKeyCorridor,collect_positive_data
from Models.model import Detector
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import tensor_to_np,make_dir, display_frames_as_gif
from PIL import ImageFont, ImageDraw, Image
from torch.nn.utils.rnn import pad_sequence
import argparse



parser = argparse.ArgumentParser(description='PyTorch Deep State Identifier')

parser.add_argument('--model_path', default='./', type=str,
                    help='the path for the pretrained model weight')
parser.add_argument('--save_dir', default='./tmp_test', type=str,
                    help='the path to save the visualization result')
args = parser.parse_args()



#generate a unseen environment 
frames_partial, frames_full = collect_positive_data()

frames_partial = np.array(frames_partial)
length = torch.tensor([frames_partial.shape[0]])

confidents_matrix = np.zeros((len(frames_partial)))
confidents_count = np.zeros((len(frames_partial)))


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

frame_visual = []
for ind, conf in enumerate(output):
    frame = Image.fromarray(np.uint8(frames_full[ind]))
    draw = ImageDraw.Draw(frame)
    draw.text((10, 10),  "{:.4f}".format(conf),font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf',size = 25), fill = (255, 255, 255))
    frame = np.array(frame)
    frame_visual.append(frame)
display_frames_as_gif(frame_visual,dir=args.save_dir)
