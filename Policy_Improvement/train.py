import torch
from torch.autograd import Variable
import sys
import os
from tqdm import *
import numpy as np
import random
from utils import save_checkpoint, tensor_to_np, accuracy
import torch.nn.functional as F


def train_epoch(args,
                train_loader,
                criterion,
                optimizer_subgoal,
                optimizer_trajectory,
                epoch,
                critical_state_detector,
                return_predictor):
    param_groups = optimizer_subgoal.param_groups[0]
    curr_lr = param_groups["lr"]
    critical_state_detector.train()
    return_predictor.train()
    msg_dict = {}
    for i, data in enumerate(train_loader):
        (frames,length, labels) = data

        frames = frames.float()
        labels = labels.float()
        if torch.cuda.is_available():
            frames = frames.cuda()
            labels = labels.cuda()

        labels = labels.unsqueeze(1)
        optimizer_subgoal.zero_grad()
        mask = critical_state_detector(frames,length)
        l1_norm = torch.mean(mask)

        # Orthogonal can help to stablize training (Tricks)
        Ort_loss_matrix = torch.abs(mask.mm(mask.t()))
        Ort_loss_matrix = (torch.ones_like(Ort_loss_matrix) - torch.eye(Ort_loss_matrix.size(0)).cuda()) * Ort_loss_matrix
        orthogonal_loss = torch.mean(torch.triu(Ort_loss_matrix, diagonal=1))
        orthogonal_loss = args.orthogonal_weight * orthogonal_loss


        loss_l1 =  args.l1_weight * torch.linalg.norm(mask,ord=1)
        bs,f_len = mask.size()
        mask = mask.view(bs,f_len,1,1,1)
        masked_frames = mask * frames[:,:f_len,:,:,:]

        # print(mask)

        output = return_predictor(masked_frames,length)
        loss_classify = args.classify_weight * criterion(output,labels)


        reverse_mask = torch.ones_like(mask) - mask
        reverse_frames = reverse_mask * frames[:,:f_len,:,:,:]


        output_r = return_predictor(reverse_frames,length)


        loss_classify_r =  args.reverse_weight * criterion(output_r,torch.zeros_like(output_r))


        loss_total = loss_l1 + loss_classify + loss_classify_r + orthogonal_loss

        loss_total.backward()
        optimizer_subgoal.step()

        optimizer_trajectory.zero_grad()

        output = return_predictor(frames,length)
        loss_classify_traj = criterion(output,labels)
        loss_classify_traj.backward()
        optimizer_trajectory.step()
        if i == 0:
            # msg_dict['Orth'] = tensor_to_np(loss_l1)
            msg_dict['L1_Norm'] = tensor_to_np(l1_norm)
            msg_dict['Orthogonal'] = tensor_to_np(orthogonal_loss)
            msg_dict['LR'] = curr_lr
            msg_dict['Cls_Goal'] = tensor_to_np(loss_classify)
            msg_dict['Cls_R_Goal'] = tensor_to_np(loss_classify_r)
            msg_dict['Cls_Traj'] = tensor_to_np(loss_classify_traj)
        else:
            # msg_dict['Orth'] = 0.1 * tensor_to_np(loss_l1) + 0.9 * msg_dict['Orth']
            msg_dict['L1_Norm'] = 0.1 * tensor_to_np(l1_norm) + 0.9 * msg_dict['L1_Norm']
            msg_dict['Orthogonal'] = 0.1 * tensor_to_np(orthogonal_loss) + 0.9 * msg_dict['Orthogonal']
            msg_dict['Cls_Goal'] = 0.1 * tensor_to_np(loss_classify) + 0.9 * msg_dict['Cls_Goal']
            msg_dict['Cls_R_Goal'] = 0.1 * tensor_to_np(loss_classify_r) + 0.9 * msg_dict['Cls_R_Goal']
            msg_dict['Cls_Traj'] = 0.1 * tensor_to_np(loss_classify_traj) + 0.9 * msg_dict['Cls_Traj']
        epoch_msg = 'Epoch = {} '.format(epoch)
        for k in msg_dict.keys():
            epoch_msg += (k + ' : {:.4f} '.format(msg_dict[k]))
        print(epoch_msg)


def test_epoch(args,test_loader,criterion,critical_state_detector,return_predictor):
    critical_state_detector.eval()
    return_predictor.eval()
    loss_mask_avg = 0
    loss_r_avg = 0
    loss_clean_avg = 0
    l1_regular_list = []
    var_list = []
    for steps, data in enumerate(test_loader):

        (frames,length, labels) = data
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

            loss_mask = criterion(output,labels)

            reverse_mask = torch.ones_like(mask) - mask
            reverse_frames = reverse_mask * frames

            output_r = return_predictor(reverse_frames,length)

            loss_r = criterion(output_r,labels)

            output = return_predictor(frames,length)
            loss_clean = criterion(output,labels)
        loss_mask_avg += tensor_to_np(loss_mask)
        loss_r_avg += tensor_to_np(loss_r)
        loss_clean_avg += tensor_to_np(loss_clean)
    var_list = np.array(var_list)
    l1_regular_list = np.array(l1_regular_list)
    epoch_msg = ' ==> Test '
    epoch_msg += 'loss clean : {:4f} '.format(loss_clean_avg/(steps+1.0))
    epoch_msg += 'loss masked: {:4f} '.format( loss_mask_avg/(steps+1.0))
    epoch_msg += 'loss reversed: {:4f} '.format(loss_r_avg/(steps+1.0))
    epoch_msg += 'L1 Average: {:4f} '.format(np.mean(l1_regular_list))
    epoch_msg += 'L1 Std: {:4f} '.format(np.std(l1_regular_list))
    epoch_msg += 'Var Average: {:4f} '.format(np.mean(var_list))
    print(epoch_msg)
    return loss_mask_avg/(steps+1.0), loss_r_avg/(steps+1.0), loss_clean_avg/(steps+1.0)
