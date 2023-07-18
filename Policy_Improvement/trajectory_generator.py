import argparse
import datetime
import os
import pprint
import random

import numpy as np
import torch
from atari_network import DQN
from atari_wrapper import make_atari_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer, to_numpy
from tianshou.policy import RainbowPolicy
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.data import Batch
from tqdm import *


parser = argparse.ArgumentParser()
parser.add_argument("--frames-stack", type=int, default=4)
parser.add_argument("--scale-obs", type=int, default=0)
parser.add_argument("--task", type=str, default="SeaquestNoFrameskip-v4")
parser.add_argument("--test-num", type=int, default=1)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--training-num", type=int, default=1)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--n-step", type=int, default=3)
parser.add_argument("--target-update-freq", type=int, default=500)
parser.add_argument("--noisy-std", type=float, default=0.1)
parser.add_argument("--num-atoms", type=int, default=51)
parser.add_argument("--v-min", type=float, default=-10.)
parser.add_argument("--v-max", type=float, default=10.)
parser.add_argument("--no-dueling", action="store_true", default=False)
parser.add_argument("--no-noisy", action="store_true", default=False)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--eps-test", type=float, default=0.005)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--path", type=str, default='./')
parser.add_argument("--sample_numebers", type=int, default=10000)
parser.add_argument("--random_thres",type=float, default=0.1)
parser.add_argument("--model_dir", type=str, default='./')
args = parser.parse_args()

if not os.path.isdir(args.path):
    os.makedirs(args.path)


def get_all_pth_files(dir_path):
  """Get all .pth files in a given directory.

  Args:
    dir_path: The path to the directory to search.

  Returns:
    A list of all .pth files in the directory.
  """
  pth_files = []
  for file_name in os.listdir(dir_path):
    if file_name.endswith(".pth"):
      pth_files.append(os.path.join(dir_path, file_name))

  return pth_files


env, train_envs, test_envs = make_atari_env(
    args.task,
    args.seed,
    args.training_num,
    args.test_num,
    scale=args.scale_obs,
    frame_stack=args.frames_stack,
)

args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n


SeaquestNoFrameskip_dqn_model_list = get_all_pth_files(args.model_dir)

net = DQN(*args.state_shape, args.action_shape, args.device).to(args.device)

optim = torch.optim.Adam(net.parameters(), lr=args.lr)
# define policy
policy = DQNPolicy(
    net,
    optim,
    args.gamma,
    args.n_step,
    target_update_freq=args.target_update_freq
)


# frames_full = []
# steep = 0
for i in tqdm(range(args.sample_numebers)):
    if i % 100 == 0:
        model_path = random.choice(SeaquestNoFrameskip_dqn_model_list)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    if random.random() > args.random_thres:
        policy.set_eps(args.eps_test)
    else:
        eps_test = random.random()
        policy.set_eps(eps_test)
    data = Batch(
        obs={},
        act={},
        rew={},
        terminated={},
        truncated={},
        done={},
        obs_next={},
        info={},
        policy={}
    )
    rval = test_envs.reset()

    data.obs = rval

    rew_final = 0
    traj = []
    traj.append(rval[0])
    rewards = []
    rewards.append([0])
    while True:
        with torch.no_grad():
            result = policy(data, None)
            act = to_numpy(result.act)
            data.update(policy=policy, act=act)
            action_remap = policy.map_action(data.act)
            result = test_envs.step(action_remap)
            obs_next, rew, done, info = result
            traj.append(obs_next[0])
            rew_final += rew
            rewards.append(rew)
            data.update(
                obs_next=obs_next,
                rew=rew,
                done=done,
                info=info
            )
            data.obs = data.obs_next
            # if steep >=10:
            #     break
            if done:
                break
    traj = np.array(traj)
    np.save(os.path.join(args.path,str(i)+'.npy'),traj)
    rewards = np.array(rewards)
    np.save(os.path.join(args.path,str(i)+'_r.npy'),rewards)
