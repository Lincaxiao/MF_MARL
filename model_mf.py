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


class HierarchicalActor(nn.Module):
    """
    第1步决定是否分配
    第2步选择具体服务器
    最终输出 n_servers+1 个动作
    """
    def __init__(self, obs_dim: int, mean_field_dim: int, n_servers: int, hidden_size: int = 128):
        super().__init__()
        self.n_servers = n_servers
        self.action_dim = n_servers + 1  # 0 表示不分配
        # 共享干层
        self.input_dim = obs_dim + mean_field_dim
        self.base = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # 两个策略头
        self.head_alloc = nn.Linear(hidden_size, 2)         # 是否分配 (0/1)
        self.head_server = nn.Linear(hidden_size, n_servers) # 选哪台服务器

    def _build_joint_probs(self, alloc_logits: torch.Tensor, server_logits: torch.Tensor):
        """构造长度 (n_servers+1) 的联合概率向量。"""
        alloc_probs = F.softmax(alloc_logits, dim=-1)                    # (B, 2)
        server_probs = F.softmax(server_logits, dim=-1)                  # (B, n_servers)
        p_no_alloc = alloc_probs[:, 0:1]                                 # (B,1)
        p_alloc = alloc_probs[:, 1:2]                                    # (B,1)
        joint_server = p_alloc * server_probs                            # (B,n_servers)
        probs = torch.cat([p_no_alloc, joint_server], dim=-1)            # (B, n_servers+1)
        # 避免 log(0)
        probs = torch.clamp(probs, min=1e-12)
        logits = torch.log(probs)                                        # 对应联合 logits
        return logits, probs

    def forward(self, obs: torch.Tensor, mean_field: torch.Tensor):
        """返回联合动作 logits，形状 (B, n_servers+1)"""
        x = torch.cat([obs, mean_field], dim=-1)
        h = self.base(x)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)
        joint_logits, _ = self._build_joint_probs(alloc_logits, server_logits)
        return joint_logits  # 供外部做 softmax/log_softmax

    def get_action(self, obs: torch.Tensor, mean_field: torch.Tensor, action_mask: torch.Tensor = None):
        """采样动作并返回 (环境动作索引, log_prob)"""
        # 计算层次化 logits
        x = torch.cat([obs, mean_field], dim=-1)
        h = self.base(x)
        alloc_logits = self.head_alloc(h)                    # (1,2)
        server_logits = self.head_server(h)                  # (1,n_servers)

        # 第 1 步：是否分配
        alloc_dist = Categorical(logits=alloc_logits)
        alloc_action = alloc_dist.sample()                   # 0: 不分配, 1: 分配
        log_prob_alloc = alloc_dist.log_prob(alloc_action)

        if alloc_action.item() == 0:
            final_action = torch.tensor([0], device=obs.device)
            log_prob = log_prob_alloc  # Final log-prob 就是 alloc 部分
        else:
            # 第 2 步：选择服务器，可选掩码
            if action_mask is not None:
                # mask 第 0 位对应不分配，后面的 mask 对应各服务器
                server_mask = action_mask[:, 1:].clone()
                server_logits = server_logits.masked_fill(~server_mask, -1e10)
                # 如果没有任何服务器可选，直接退回为不分配
                if (~server_mask).all():
                    final_action = torch.tensor([0], device=obs.device)
                    log_prob = log_prob_alloc  # 退化为只依赖 alloc
                    return final_action.item(), log_prob
            server_dist = Categorical(logits=server_logits)
            server_action = server_dist.sample()             # 0..n_servers-1
            log_prob_server = server_dist.log_prob(server_action)

            final_action = server_action + 1                 # 环境动作编码：1..n_servers
            log_prob = log_prob_alloc + log_prob_server      # 联合对数概率

        return final_action.item(), log_prob

    def get_policy(self, obs: torch.Tensor, mean_field: torch.Tensor):
        """返回 (probs, log_probs) 长度 n_servers+1，用于批量计算损失"""
        x = torch.cat([obs, mean_field], dim=-1)
        h = self.base(x)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)
        joint_logits, probs = self._build_joint_probs(alloc_logits, server_logits)
        log_probs = torch.log(probs)
        return probs, log_probs
