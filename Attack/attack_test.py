# -*- coding: utf-8 -*-
import numpy as np
import torch
import os, gym, time, glob, sys
import matplotlib.pyplot as plt
from utils import tensor_to_np
from PIL import ImageFont, ImageDraw, Image
from torch.nn.utils.rnn import pad_sequence
from model import Detector
from a3c import NNPolicy_CROSS,NNPolicy,prepro
import torch.nn.functional as F
from tqdm import *

# Test the result and release the pretrained model.
def adversarial_attack(model,env,seed,critical_states,top_k,render=False):
    critical_states = np.array(critical_states)
    epr = 0
    frames_full = []
    index_states = (-critical_states).argsort()[:top_k]
    act_set = np.array([0, 1, 2, 3, 4, 5])
    Done = False
    iter_step = 0
    env.seed(seed)
    init_state = prepro(env.reset())
    state = torch.tensor(init_state).cuda()
    hx = torch.zeros(1, 256).cuda()
    if render:
        frames_full.append(env.render(mode='rgb_array'))
    episode_length = 0
    while not Done:
        iter_step +=1
        value, logit, hx = model((state.view(1,1,80,80), hx.detach()))
        logp = F.log_softmax(logit, dim=-1)
        action = logp.max(1)[1].cpu().data.numpy()[0]
        if iter_step in index_states:
            act_set_1 = act_set[act_set!=action]
            action = np.random.choice(act_set_1)
        state, reward, done, _ = env.step(action)
        if render:
            frames_full.append(env.render(mode='rgb_array'))
        state_proc = prepro(state)
        state = torch.tensor(state_proc).cuda()
        epr += reward
        Done = done or epr!=0 or iter_step >= 199
    return epr, frames_full

def detect_CS(model,detector,env,seed,render=False):
    env.seed(seed)
    epr = 0
    init_state = prepro(env.reset())
    traj = []
    traj.append(init_state)
    state = torch.tensor(init_state).cuda()
    hx = torch.zeros(1, 256).cuda()
    frames_full = []
    if render:
        frames_full.append(env.render(mode='rgb_array'))
    traj = []
    Done = False
    episode_length = 0
    while not Done:
        value, logit, hx = model((state.view(1,1,80,80), hx.detach()))
        logp = F.log_softmax(logit, dim=-1)
        action = logp.max(1)[1].cpu().data.numpy()[0]
        state, reward, done, _ = env.step(action)
        if render:
            frames_full.append(env.render(mode='rgb_array'))
        state_proc = prepro(state)
        traj.append(state_proc)
        state = torch.tensor(state_proc).cuda()
        epr += reward
        # Done = done or epr!=0
        episode_length+=1
        Done = done or episode_length >= 199 or epr!=0
    traj = np.array(traj)
    length = torch.tensor([traj.shape[0]])
    test_data = torch.tensor(traj)
    test_data = pad_sequence([test_data], batch_first=True, padding_value=0).cuda()
    output = detector(test_data,length)
    output = tensor_to_np(output)[0]
    return output, epr, frames_full



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Subgoal Adv Test')
    parser.add_argument('--top-k', default=30, type=int,
                        help='Top K Num for subgoals')
    parser.add_argument('--model_path', default='./', type=str,
                        help='the model for testing')
    parser.add_argument('--test_num', default=500, type=int,
                        help='the test number')
    parser.add_argument('--seed', default=0, type=int,
                            help='random seed')
    parser.add_argument('--cross_step',default=0,type=int,
                            help='set as 1 to start cross-step test')
    parser.add_argument('--cross_seed',default=0,type=int,
                        help='set as 1 to start cross-seed test')
    parser.add_argument('--cross_architecture',default=0,type=int,
                        help='set as 1 to start cross-arch test')
    parser.add_argument('--random_baseline',default=0,type=int,
                        help='set as 1 to start cross-polict test')
    parser.add_argument('--training_data',default=0,type=int,
                        help='set as 0:single policy training set as 1: multiple policies training set.')
    args = parser.parse_args()
    top_k = args.top_k
    if args.cross_architecture == 1:
        model = NNPolicy_CROSS(channels=1, memsize=256, num_actions=gym.make('Pong-v4').action_space.n)
        model.load_model_by_name('./Attack_Model/Cross_Arch/model.40.tar')
    else:
        model = NNPolicy(channels=1, memsize=256, num_actions=gym.make('Pong-v4').action_space.n)
        if args.cross_seed == 1:
            model.load_model_by_name('./Attack_Model/Cross_Seed/model.40.tar')
        elif args.cross_step == 1:
            model.load_model_by_name('./Attack_Model/Cross_Step/model.40.tar')
        else:
            if args.training_data == 1:
                model.load_model_by_name('./Attack_Model/In_Policy_Dataset_1/model.80.tar')
            else:
                 model.load_model_by_name('./Attack_Model/In_Policy_Dataset_0/model.80.tar')

    model.cuda()


    detector = Detector()
    state_dict = torch.load(args.model_path)
    detector.load_state_dict(state_dict)
    detector.cuda()

    seed = args.seed
    seed_list = range(seed,seed+args.test_num)


    env = gym.make('Pong-v4')

    originial_win = []
    adv_win = []
    change_rate = 0
    win_change_rate = 0
    for i in tqdm(seed_list):
        env.seed(i) ; torch.manual_seed(i)
        cs_states, epr_ori, frames_full = detect_CS(model,detector,env,i)
        if args.random_baseline == 1:
            cs_states = np.random.rand(len(cs_states))
        epr_adv, frames_full_adv = adversarial_attack(model,env,i,cs_states,top_k)
        originial_win.append(epr_ori)
        adv_win.append(epr_adv)
        if epr_ori != epr_adv:
            change_rate +=1
            if epr_ori == 1:
                win_change_rate += 1
    originial_win = np.array(originial_win)
    adv_win = np.array(adv_win)
    win_rate = np.sum(originial_win[originial_win==1])
    win_adv_rate = np.sum(adv_win[adv_win==1])
    print("===============(Consider failure cases)==============")
    print(win_rate)
    print(win_adv_rate)
    
    print("===============(ONLY consider win cases)==============")
    print(win_change_rate)
    print(win_rate)
