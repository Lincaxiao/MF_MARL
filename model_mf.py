import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, mean_field, action, reward, next_obs, next_mean_field, done, action_mask):
        experience = (
            obs, mean_field, action, reward, next_obs,
            next_mean_field, done, action_mask
        )
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        obs_list, mf_list, act_list, r_list, next_obs_list, \
            next_mf_list, done_list, mask_list = zip(*batch)

        mf_batch = np.array(mf_list, dtype=np.float32)
        act_batch = np.array(act_list, dtype=np.int64)
        r_batch = np.array(r_list, dtype=np.float32).reshape(-1, 1)
        next_mf_batch = np.array(next_mf_list, dtype=np.float32)
        done_batch = np.array(done_list, dtype=np.bool_).reshape(-1, 1)

        return obs_list, mf_batch, act_batch, r_batch, \
            next_obs_list, next_mf_batch, done_batch, mask_list

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, mean_field_dim: int, action_dim: int, hidden_size=128):
        super(Actor, self).__init__()
        # 输入维度是局部观测和平均场的拼接
        self.input_dim = obs_dim + mean_field_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, obs, mean_field):
        # 拼接输入
        x = torch.cat([obs, mean_field], dim=-1)
        return self.net(x)

    def get_action(self, obs, mean_field, action_mask):
        logits = self.forward(obs, mean_field)
        # 掩码，非法动作会给出极小概率
        logits[action_mask == 0] = -1e10

        # 从概率分布中采样
        dist = Categorical(logits=logits)
        action = dist.sample()

        # 计算该动作的log-prob，用于后续的梯度更新
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, mean_field_dim: int, action_dim: int, hidden_size=128):
        super(Critic, self).__init__()
        self.input_dim = obs_dim + mean_field_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)  # 输出每个动作的Q值
        )

    def forward(self, obs, mean_field):
        x = torch.cat([obs, mean_field], dim=-1)
        return self.net(x)
