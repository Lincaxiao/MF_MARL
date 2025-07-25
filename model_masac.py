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
        next_action_mask: list = None,
    ):
        """存储一次交互。
        参数说明同 train_masac.Trainer 中调用；新增 next_action_mask 用于 target 期望的合法动作过滤。"""
        experience = (
            obs,
            global_state,
            action,
            reward,
            next_obs,
            next_global_state,
            done,
            action_mask,
            next_action_mask,
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
            next_mask_list,
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
            next_mask_list,
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
    """
    具备顺序不变性的层次化 Actor。
    使用共享的编码器处理服务器状态，并通过聚合来消除位置偏见。
    """
    def __init__(self, user_obs_dim: int, server_obs_dim: int, n_servers: int, hidden_size: int = 128):
        super().__init__()
        self.n_servers = n_servers
        self.action_dim = n_servers + 1

        # 1. 服务器状态编码器 (权重共享)
        self.server_encoder = nn.Sequential(
            nn.Linear(server_obs_dim, hidden_size),
            nn.ReLU(),
        )

        # 2. 用聚合后的服务器信息 + 用户自身信息 决定 "是否分配"
        self.alloc_net = nn.Sequential(
            nn.Linear(user_obs_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

        # 3. 对每台服务器单独打分 (共享权重)
        self.score_net = nn.Sequential(
            nn.Linear(user_obs_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    @staticmethod
    def _build_joint_probs(alloc_logits: torch.Tensor, server_logits: torch.Tensor):
        alloc_probs = F.softmax(alloc_logits, dim=-1)
        server_probs = F.softmax(server_logits, dim=-1)
        p_no_alloc = alloc_probs[:, 0:1]
        p_alloc = alloc_probs[:, 1:2]
        joint_server = p_alloc * server_probs
        probs = torch.cat([p_no_alloc, joint_server], dim=-1)
        probs = torch.clamp(probs, min=1e-12)
        logits = torch.log(probs)
        return logits, probs

    def forward(self, user_obs: torch.Tensor, servers_obs: torch.Tensor):
        """
        Args:
            user_obs (torch.Tensor): 用户自身观测, shape (B, user_obs_dim)
            servers_obs (torch.Tensor): 所有服务器的观测, shape (B, n_servers, server_obs_dim)
        """
        # 1. 编码所有服务器状态
        # (B, n_servers, server_obs_dim) -> (B * n_servers, server_obs_dim)
        batch_size = servers_obs.size(0)
        servers_obs_flat = servers_obs.view(-1, servers_obs.shape[-1])
        encoded_servers = self.server_encoder(servers_obs_flat)
        # (B * n_servers, hidden_size) -> (B, n_servers, hidden_size)
        encoded_servers = encoded_servers.view(batch_size, self.n_servers, -1)

        # 2. 聚合服务器信息 (均值) 用于是否分配逻辑
        aggregated_servers = torch.mean(encoded_servers, dim=1)  # (B, hidden)

        # 3. 计算 alloc_logits
        alloc_input = torch.cat([user_obs, aggregated_servers], dim=-1)
        alloc_logits = self.alloc_net(alloc_input)               # (B,2)

        # 4. 为每台服务器计算打分
        #    concat: (B, n, user_dim+hidden) -> (B*n, ...)
        user_expand = user_obs.unsqueeze(1).expand(-1, self.n_servers, -1)
        per_server_input = torch.cat([user_expand, encoded_servers], dim=-1)
        per_server_input = per_server_input.reshape(-1, per_server_input.shape[-1])
        scores = self.score_net(per_server_input).view(batch_size, self.n_servers)  # (B, n_servers)
        server_logits = scores
        
        joint_logits, _ = self._build_joint_probs(alloc_logits, server_logits)
        return joint_logits

    def get_action(self, user_obs: torch.Tensor, servers_obs: torch.Tensor, action_mask: torch.Tensor = None):
        """从联合概率分布中一次性采样动作。"""
        logits = self.forward(user_obs, servers_obs)

        if action_mask is not None:
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e10)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 如果只有一个实例 (B=1)，返回 item()
        if action.numel() == 1:
            return action.item(), log_prob.item()
        return action, log_prob

    def get_policy(self, user_obs: torch.Tensor, servers_obs: torch.Tensor):
        """获取策略的概率和 log 概率。"""
        # 与 forward 逻辑类似，但需要返回 probs
        batch_size = servers_obs.size(0)
        servers_obs_flat = servers_obs.view(-1, servers_obs.shape[-1])
        encoded_servers = self.server_encoder(servers_obs_flat)
        encoded_servers = encoded_servers.view(batch_size, self.n_servers, -1)
        aggregated_servers = torch.mean(encoded_servers, dim=1)
        combined_obs = torch.cat([user_obs, aggregated_servers], dim=-1)
        h = self.base(combined_obs)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)
        
        _, probs = self._build_joint_probs(alloc_logits, server_logits)
        log_probs = torch.log(probs)
        return probs, log_probs