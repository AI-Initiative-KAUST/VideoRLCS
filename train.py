import torch
from torch.autograd import Variable
import sys
import os
from tqdm import *
import numpy as np
import random
from utils import save_checkpoint, tensor_to_np, accuracy


def train_epoch(args,
                train_loader,
                criterion,
                optimizer_detector,
                optimizer_predictor,
                epoch,
                critical_state_detector,
                return_predictor):
    param_groups = optimizer_detector.param_groups[0]
    curr_lr = param_groups["lr"]
    critical_state_detector.train()
    return_predictor.train()
    msg_dict = {}
    for i, data in enumerate(train_loader):
        (frames,length,labels) = data
        frames = frames.float()
        if torch.cuda.is_available():
            frames = frames.cuda()
            labels = labels.cuda()



        optimizer_detector.zero_grad()

        mask = critical_state_detector(frames,length)

        loss_l1 =  args.l1_weight * torch.linalg.norm(mask,ord=1)

        bs,f_len = mask.size()
        mask = mask.view(bs,f_len,1,1,1)
        masked_frames = mask * frames

        output = return_predictor(masked_frames,length)

        loss_classify = args.classify_weight * criterion(output,labels)
        acc_mask = accuracy(output,labels)[0]


        reverse_mask = torch.ones_like(mask) - mask
        reverse_frames = reverse_mask * frames

        output_r = return_predictor(reverse_frames,length)

        confused_label = torch.ones_like(output_r)*0.5
        loss_classify_r = args.reverse_weight * criterion(output_r,confused_label)
        acc_r = accuracy(output_r,labels)[0]


        loss_total = loss_l1 + loss_classify + loss_classify_r


        loss_total.backward()
        optimizer_detector.step()

        optimizer_predictor.zero_grad()

        output = return_predictor(frames,length)

        loss_classify_traj = criterion(output,labels)
        acc_clean = accuracy(output,labels)[0]
        loss_classify_traj.backward()
        optimizer_predictor.step()
        if i == 0:
            msg_dict['L1_Norm'] = tensor_to_np(loss_l1)
            msg_dict['LR'] = curr_lr
            msg_dict['Cls_Goal'] = tensor_to_np(loss_classify)
            msg_dict['Cls_Acc_Goal'] = tensor_to_np(acc_mask)[0]
            msg_dict['Cls_R_Goal'] = tensor_to_np(loss_classify_r)
            msg_dict['Cls_Acc_R_Goal'] = tensor_to_np(acc_r)[0]
            msg_dict['Cls_Traj'] = tensor_to_np(loss_classify_traj)
            msg_dict['Cls_Acc_Traj'] = tensor_to_np(acc_clean)[0]
        else:
            msg_dict['L1_Norm'] = 0.1 * tensor_to_np(loss_l1) + 0.9 * msg_dict['L1_Norm']
            msg_dict['Cls_Goal'] = 0.1 * tensor_to_np(loss_classify) + 0.9 * msg_dict['Cls_Goal']
            msg_dict['Cls_R_Goal'] = 0.1 * tensor_to_np(loss_classify_r) + 0.9 * msg_dict['Cls_R_Goal']
            msg_dict['Cls_Traj'] = 0.1 * tensor_to_np(loss_classify_traj) + 0.9 * msg_dict['Cls_Traj']
            msg_dict['Cls_Acc_Traj'] = 0.1 * tensor_to_np(acc_clean)[0] + 0.9 * msg_dict['Cls_Acc_Traj']
            msg_dict['Cls_Acc_R_Goal'] = 0.1 * tensor_to_np(acc_r)[0] + 0.9 * msg_dict['Cls_Acc_R_Goal']
            msg_dict['Cls_Acc_Goal'] = 0.1 * tensor_to_np(acc_mask)[0] + 0.9 * msg_dict['Cls_Acc_Goal']
        epoch_msg = 'Epoch = {} '.format(epoch)
    for k in msg_dict.keys():
        epoch_msg += (k + ' : {:.4f} '.format(msg_dict[k]))
    print(epoch_msg)


def test_epoch(args,test_loader,critical_state_detector,return_predictor):
    critical_state_detector.eval()
    return_predictor.eval()
    acc_mask_avg = 0
    acc_r_avg = 0
    acc_clean_avg = 0
    l1_regular_list = []
    var_list = []
    for steps, data in enumerate(test_loader):
        (frames,length,labels) = data
        frames = frames.float()
        if torch.cuda.is_available():
            frames = frames.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            mask = critical_state_detector(frames,length)

            l1_regular_list.append(tensor_to_np(torch.linalg.norm(mask,ord=1)))
            var_list.append(tensor_to_np(torch.var(mask)))
            bs,f_len = mask.size()
            mask = mask.view(bs,f_len,1,1,1)
            masked_frames = mask * frames

            output = return_predictor(masked_frames,length)


            acc_mask = accuracy(output,labels)[0]

            reverse_mask = torch.ones_like(mask) - mask
            
            reverse_frames = reverse_mask * frames
            
            output_r = return_predictor(reverse_frames,length)

            acc_r = accuracy(output_r,labels)[0]

            output = return_predictor(frames, length)

            acc_clean = accuracy(output,labels)[0]

        acc_mask_avg += tensor_to_np(acc_mask)[0]
        acc_r_avg += tensor_to_np(acc_r)[0]
        acc_clean_avg += tensor_to_np(acc_clean)[0]
    var_list = np.array(var_list)
    l1_regular_list = np.array(l1_regular_list)
    epoch_msg = ' ==> Test '
    epoch_msg += 'acc clean : {:4f} '.format(acc_clean_avg/(steps+1.0))
    epoch_msg += 'acc masked: {:4f} '.format( acc_mask_avg/(steps+1.0))
    epoch_msg += 'acc reversed: {:4f} '.format(acc_r_avg/(steps+1.0))
    epoch_msg += 'L1 Average: {:4f} '.format(np.mean(l1_regular_list))
    epoch_msg += 'L1 Std: {:4f} '.format(np.std(l1_regular_list))
    epoch_msg += 'Var Average: {:4f} '.format(np.mean(var_list))
    print(epoch_msg)
    return acc_mask_avg/(steps+1.0), acc_r_avg/(steps+1.0), acc_clean_avg/(steps+1.0)
