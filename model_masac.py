import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import collections
import random


class ReplayBuffer:
    """用于 MASAC 的经验回放缓冲区。
    存储局部观测列表、全局状态、动作列表、奖励、下一时刻对应内容以及 done。
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(
        self,
        obs: list,
        global_state: np.ndarray,
        action: list,
        reward: float,
        next_obs: list,
        next_global_state: np.ndarray,
        done: bool,
        action_mask: list = None,
    ):
        experience = (
            obs,
            global_state,
            action,
            reward,
            next_obs,
            next_global_state,
            done,
            action_mask,
        )
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        (
            obs_list,
            gs_list,
            act_list,
            r_list,
            next_obs_list,
            next_gs_list,
            done_list,
            mask_list,
        ) = zip(*batch)

        gs_batch = np.array(gs_list, dtype=np.float32)
        act_batch = np.array(act_list, dtype=np.int64)
        r_batch = np.array(r_list, dtype=np.float32).reshape(-1, 1)
        next_gs_batch = np.array(next_gs_list, dtype=np.float32)
        done_batch = np.array(done_list, dtype=np.bool_).reshape(-1, 1)

        return (
            obs_list,
            gs_batch,
            act_batch,
            r_batch,
            next_obs_list,
            next_gs_batch,
            done_batch,
            mask_list,
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """简单 Actor，仅接收局部观测。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs)

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """返回 (action, log_prob)。obs 形状 (B, obs_dim) 或 (obs_dim,)"""
        logits = self.forward(obs)
        if action_mask is not None:
            logits[action_mask == 0] = -1e10
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class Critic(nn.Module):
    """Soft Q-Network，输入全局状态，输出每个动作的 Q 值。"""

    def __init__(self, global_state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, global_state: torch.Tensor):
        return self.net(global_state)


class HierarchicalActor(nn.Module):
    """两层策略：先决定是否分配，再选择服务器。只使用局部观测。"""

    def __init__(self, obs_dim: int, n_servers: int, hidden_size: int = 128):
        super().__init__()
        self.n_servers = n_servers
        self.action_dim = n_servers + 1  # 0 表示不分配
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.head_alloc = nn.Linear(hidden_size, 2)  # 是否分配
        self.head_server = nn.Linear(hidden_size, n_servers)  # 服务器选择

    @staticmethod
    def _build_joint_probs(alloc_logits: torch.Tensor, server_logits: torch.Tensor):
        alloc_probs = F.softmax(alloc_logits, dim=-1)  # (B,2)
        server_probs = F.softmax(server_logits, dim=-1)  # (B,n_servers)
        p_no_alloc = alloc_probs[:, 0:1]
        p_alloc = alloc_probs[:, 1:2]
        joint_server = p_alloc * server_probs
        probs = torch.cat([p_no_alloc, joint_server], dim=-1)  # (B, n_servers+1)
        probs = torch.clamp(probs, min=1e-12)
        logits = torch.log(probs)
        return logits, probs

    def forward(self, obs: torch.Tensor):
        h = self.base(obs)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)
        joint_logits, _ = self._build_joint_probs(alloc_logits, server_logits)
        return joint_logits

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        h = self.base(obs)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)

        alloc_dist = Categorical(logits=alloc_logits)
        alloc_action = alloc_dist.sample()
        log_prob_alloc = alloc_dist.log_prob(alloc_action)

        if alloc_action.item() == 0:
            final_action = torch.tensor([0], device=obs.device)
            log_prob = log_prob_alloc
        else:
            if action_mask is not None:
                server_mask = action_mask[:, 1:].clone()
                server_logits = server_logits.masked_fill(~server_mask, -1e10)
                if (~server_mask).all():
                    final_action = torch.tensor([0], device=obs.device)
                    log_prob = log_prob_alloc
                    return final_action.item(), log_prob
            server_dist = Categorical(logits=server_logits)
            server_action = server_dist.sample()
            log_prob_server = server_dist.log_prob(server_action)

            final_action = server_action + 1
            log_prob = log_prob_alloc + log_prob_server

        return final_action.item(), log_prob

    def get_policy(self, obs: torch.Tensor):
        h = self.base(obs)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)
        joint_logits, probs = self._build_joint_probs(alloc_logits, server_logits)
        log_probs = torch.log(probs)
        return probs, log_probs 