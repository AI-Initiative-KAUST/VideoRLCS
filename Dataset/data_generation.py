# -*- coding: utf-8 -*-
from Minigrid.minigrid import minigrid_env
from Minigrid.minigrid.envs.keycorridor import KeyCorridorEnv
import numpy as np
import random
import sys
sys.path.append('../')
class SimpleKeyCorridor(KeyCorridorEnv):
    """docstring for SimpleKeyCorridor."""

    def __init__(self, num_rows=3, obj_type="ball", room_size=6, **kwargs):
        super().__init__(num_rows=num_rows,obj_type="ball",room_size=room_size, **kwargs)
        self.height = (self.room_size - 1) * self.num_rows + 1
        self.width = (self.room_size - 1) * self.num_cols + 1
        self.max_steps = np.inf
        self.reset_memory()
    def get_obj(self,obj_type="key"):
        obj_pos_list = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid.get(x,y) == None:
                    continue
                else:
                    if obj_type == self.grid.get(x,y).type:
                        obj_pos_list.append((x,y))
        else:
            return obj_pos_list
    def get_agent(self):
        return self.agent_pos
    def get_door_from_key(self,door_pos_list,key_pos):
        x_key,y_key = key_pos
        key_color = self.grid.get(x_key,y_key).color
        target_door_pos_list = []
        for door_pos in door_pos_list:
            x,y = door_pos
            if self.grid.get(x,y).color == key_color:
                if self.grid.get(x,y).is_locked:
                    target_door_pos_list.append((x,y))
        return target_door_pos_list
    def reset(self, *, seed=None, options=None):
        obs,_ = super().reset(seed=seed)
        self.start_pos = self.get_agent()
        return obs,_
    def simple_step(self,action):
        """
        action = 0,1,2,3 top left right and down
        """
        self.agent_dir = 3
        self.agent_dir = self.agent_dir - action
        if self.grid.get(*self.front_pos) != None and self.grid.get(*self.front_pos).type == 'door':
            if self.grid.get(*self.front_pos).is_open == False:
                self.step(self.actions.toggle)

        return self.step(self.actions.forward)

    def reset_agent(self,pos):
        self.agent_pos = pos

    def search_optimal_path(self,pos_1,count=0):
        x_1,y_1 = pos_1
        self.agent_pos = (x_1,y_1)
        if count == 0:
            self.memory_count[x_1,y_1] = 0
        count += 1
        for act in range(4):
            self.agent_pos = (x_1, y_1)
            self.simple_step(act)
            if self.agent_pos == (x_1, y_1):
                if self.grid.get(*self.front_pos) != None and self.grid.get(*self.front_pos).type in ['key','ball']:
                    x_now,y_now = self.front_pos
                    if self.memory_count[x_now,y_now] > count:
                        self.memory_count[x_now,y_now] = count
                        self.memory_trajectory[x_now,y_now,0] = x_1
                        self.memory_trajectory[x_now,y_now,1] = y_1
                continue
            else:
                x_now,y_now = self.agent_pos
                if self.memory_count[x_now,y_now] > count:
                    self.memory_count[x_now,y_now] = count
                    self.memory_trajectory[x_now,y_now,0] = x_1
                    self.memory_trajectory[x_now,y_now,1] = y_1
                    self.search_optimal_path((x_now,y_now),count=count)
        return
    def get_trajectory(self,pos_1,pos_2):
        x_now, y_now = pos_2
        tmp_x = int(self.memory_trajectory[x_now,y_now,0])
        tmp_y = int(self.memory_trajectory[x_now,y_now,1])

        while True:
            if abs(tmp_x-x_now) + abs(tmp_y-y_now)>1:
                print((tmp_x,x_now),(tmp_y,y_now))
            record_x = int(self.memory_trajectory[tmp_x,tmp_y,0])
            record_y = int(self.memory_trajectory[tmp_x,tmp_y,1])
            self.memory_trajectory[tmp_x,tmp_y,0] = x_now
            self.memory_trajectory[tmp_x,tmp_y,1] = y_now
            if self.memory_count[tmp_x,tmp_y] == 0:
                break
            else:
                (x_now, y_now) = (tmp_x, tmp_y)
                (tmp_x, tmp_y) = (record_x, record_y)
        return self.memory_trajectory

    def close_all_doors(self,door_list):
        for door in door_list:
            self.grid.get(*door).is_open = False
    def step_with_trajectory(self,trajectory, init_pos, end_point, show_cs=False):
        self.reset_agent(init_pos)
        (point_x, point_y) = init_pos
        frames_partial = []
        frames_full = []
        frames_partial.append(self.get_frame(tile_size=4,agent_pov=True))
        frames_full.append(self.render())
        if show_cs:
            cs_list = []
        for s in range(int(self.memory_count[end_point[0],end_point[1]])):
            pre_point_x = point_x
            pre_point_y = point_y
            point_x = int(trajectory[pre_point_x,pre_point_y,0])
            point_y = int(trajectory[pre_point_x,pre_point_y,1])
            if point_x == pre_point_x:
                if point_y == pre_point_y+1:
                    obs, reward,terminated,truncated, _ = self.simple_step(2)
                else:
                    obs, reward,terminated,truncated, _ = self.simple_step(0)
            else:
                if point_x == pre_point_x+1:
                    obs, reward,terminated,truncated, _ = self.simple_step(3)
                else:
                    obs, reward,terminated,truncated, _ = self.simple_step(1)
            if show_cs:
                if self.grid.get(pre_point_x,pre_point_y) != None:
                    if self.grid.get(pre_point_x,pre_point_y).type in ["key","door","key","ball"]:
                        cs_list.append(1)
                    else:
                        cs_list.append(0)
                else:
                    cs_list.append(0)
            frames_partial.append(self.get_frame(tile_size=4,agent_pov=True))
            frames_full.append(self.render())
        if show_cs:
            return frames_partial,frames_full,cs_list
        else:
            return frames_partial,frames_full


    def pickup_key(self):
        for direct in range(4):
            self.agent_dir = direct
            self.step(self.actions.pickup)
        if self.carrying == None:
            return False
        else:
            return True
    def reset_memory(self):
        self.memory_count = np.zeros((self.height,self.width))+np.inf
        self.memory_trajectory = np.zeros((self.height,self.width,2))-1
def collect_positive_data(show_cs=False):
    key_corridor = SimpleKeyCorridor(render_mode='rgb_array')
    obs,info = key_corridor.reset()
    key_pos = key_corridor.get_obj("key")[0] # considering only 1 key
    door_list = key_corridor.get_obj("door")
    target_door = key_corridor.get_door_from_key(door_list,key_pos)[0] # considering only 1 door match the given key
    goal = key_corridor.get_obj("ball")[0]

    key_corridor.search_optimal_path(key_corridor.get_agent())
    trajectory = key_corridor.get_trajectory(key_corridor.start_pos,key_pos)
    key_corridor.close_all_doors(door_list)
    if show_cs:
        frames_partial, frames_full,cs_list = key_corridor.step_with_trajectory(trajectory,init_pos=key_corridor.start_pos,end_point=key_pos,show_cs=show_cs) # agent to key
    else:
        frames_partial, frames_full = key_corridor.step_with_trajectory(trajectory,init_pos=key_corridor.start_pos,end_point=key_pos,show_cs=show_cs) # agent to key
    status = key_corridor.pickup_key()
    key_corridor.close_all_doors(door_list)
    key_corridor.reset_memory()


    key_corridor.start_pos = key_corridor.get_agent()

    key_corridor.search_optimal_path(key_corridor.get_agent())
    trajectory = key_corridor.get_trajectory(key_corridor.start_pos,goal)
    key_corridor.close_all_doors(door_list)

    if show_cs:
        frames_partial_, frames_full_,cs_list_ = key_corridor.step_with_trajectory(trajectory,key_corridor.start_pos,end_point=goal,show_cs=show_cs)
    else:
        frames_partial_, frames_full_ = key_corridor.step_with_trajectory(trajectory,key_corridor.start_pos,end_point=goal,show_cs=show_cs)
    if status:
        if show_cs:
            return frames_partial +  frames_partial_, frames_full + frames_full_, cs_list+cs_list_
        else:
            return frames_partial +  frames_partial_, frames_full + frames_full_
    else:
        return [],[]




def collect_positive_new_policy_data():
    key_corridor = SimpleKeyCorridor(render_mode='rgb_array')
    obs,info = key_corridor.reset()
    key_pos = key_corridor.get_obj("key")[0] # considering only 1 key
    door_list = key_corridor.get_obj("door")
    target_door = key_corridor.get_door_from_key(door_list,key_pos)[0] # considering only 1 door match the given key
    goal = key_corridor.get_obj("ball")[0]

    key_corridor.search_optimal_path(key_corridor.get_agent())
    trajectory = key_corridor.get_trajectory(key_corridor.start_pos,key_pos)
    key_corridor.close_all_doors(door_list)

    frames_partial, frames_full = key_corridor.step_with_trajectory(trajectory,init_pos=key_corridor.start_pos,end_point=key_pos) # agent to key
    status = key_corridor.pickup_key()
    key_corridor.close_all_doors(door_list)
    key_corridor.reset_memory()


    key_corridor.start_pos = key_corridor.get_agent()


    random_goal = random.choice(door_list)

    key_corridor.search_optimal_path(key_corridor.get_agent())
    key_corridor.close_all_doors(door_list)
    trajectory = key_corridor.get_trajectory(key_corridor.start_pos,random_goal)
    frames_partial_, frames_full_ = key_corridor.step_with_trajectory(trajectory,init_pos=key_corridor.start_pos,end_point=random_goal) # agent to key

    key_corridor.reset_memory()


    key_corridor.start_pos = key_corridor.get_agent()

    key_corridor.search_optimal_path(key_corridor.get_agent())
    trajectory = key_corridor.get_trajectory(key_corridor.start_pos,goal)
    key_corridor.close_all_doors(door_list)

    frames_partial__, frames_full__ = key_corridor.step_with_trajectory(trajectory,key_corridor.start_pos,end_point=goal)
    if status:
        return frames_partial +  frames_partial_ + frames_partial__, frames_full + frames_full_ + frames_full__
    else:
        return [],[]



def collect_negative_data():
    key_corridor = SimpleKeyCorridor(render_mode='rgb_array')
    obs,info = key_corridor.reset()
    if random.random() >0.85:
        key_pos = key_corridor.get_obj("key")[0] # considering only 1 key
        door_list = key_corridor.get_obj("door")
        target_door = key_corridor.get_door_from_key(door_list,key_pos)[0] # considering only 1 door match the given key
        goal = key_corridor.get_obj("ball")[0]

        key_corridor.search_optimal_path(key_corridor.get_agent())
        trajectory = key_corridor.get_trajectory(key_corridor.start_pos,key_pos)
        key_corridor.close_all_doors(door_list)

        frames_partial, frames_full = key_corridor.step_with_trajectory(trajectory,init_pos=key_corridor.start_pos,end_point=key_pos) # agent to key
        status = key_corridor.pickup_key()
        key_corridor.close_all_doors(door_list)
        key_corridor.reset_memory()
        key_corridor.start_pos = key_corridor.get_agent()
        if random.random() > 0.5:
            key_corridor.search_optimal_path(key_corridor.get_agent())
            trajectory = key_corridor.get_trajectory(key_corridor.start_pos,target_door)
            key_corridor.close_all_doors(door_list)
            frames_partial_, frames_full_ = key_corridor.step_with_trajectory(trajectory,key_corridor.start_pos,end_point=goal)
            frames_partial = frames_partial +  frames_partial_
            frames_full = frames_full + frames_full_
            act = random.randint(0,3)
            for i in range(8):
                act = random.randint(0,3)
                key_corridor.simple_step(act)
                frames_partial.append(key_corridor.get_frame(tile_size=4,agent_pov=True))
                frames_full.append(key_corridor.render())
        else:
            for i in range(16):
                act = random.randint(0,3)
                key_corridor.simple_step(act)
                frames_partial.append(key_corridor.get_frame(tile_size=4,agent_pov=True))
                frames_full.append(key_corridor.render())
    else:
        frames_partial = []
        frames_full = []
        for i in range(24):
            act = random.randint(0,3)
            key_corridor.simple_step(act)
            frames_partial.append(key_corridor.get_frame(tile_size=4,agent_pov=True))
            frames_full.append(key_corridor.render())
    return frames_partial, frames_full



if __name__ == '__main__':
    from Env.GridWorld import display_frames_as_gif
    from tqdm import *
    import os
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Data Collection')
    parser.add_argument('--path', default='./toy_dataset/train/positive', type=str, help='the path to save the training trajectory')
    parser.add_argument('--num', default=1000, type=int, help='The number of the trajectory')
    parser.add_argument('--mode', default=0, type=int, help='[0/1/2] -> [positive/negative/new_policy/positive]')
    parser.add_argument('--total_workers', default=10, type=int, help='[0/1] -> [positive/negative]')
    parser.add_argument('--node_index', default=0, type=int, help='[0/1] -> [positive/negative]')

    args = parser.parse_args()
    index_list = range(args.num)
    list_chosen = np.array_split(index_list, args.total_workers)[args.node_index]

    if not os.path.isdir(args.path):
        os.makedirs(args.path)
    for i in tqdm(list_chosen):
        if args.mode == 0:
            frames_partial,frames_full = collect_positive_data()
        elif args.mode == 1:
            frames_partial,frames_full = collect_negative_data()
        elif args.mode == 2:
            frames_partial,frames_full = collect_positive_new_policy_data()
        if len(frames_partial) > 1:
            frames_partial = np.array(frames_partial)
            display_frames_as_gif(frames_full,os.path.join(args.path,str(i)+'.gif'))
            np.save(os.path.join(args.path,str(i)+'.npy'),frames_partial)
