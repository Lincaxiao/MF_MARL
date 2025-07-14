import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorLocal(nn.Module):
    """服务器或普通离散动作智能体的局部策略网络（无平均场输入）。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    # ... existing code ...
    def forward(self, obs: torch.Tensor):
        return self.net(obs)

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """采样动作并返回 (动作索引, log_prob)。支持可选的动作掩码。"""
        logits = self.forward(obs)
        if action_mask is not None:
            logits[action_mask == 0] = -1e10
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """在更新阶段计算给定动作的 log_prob 与熵。actions shape = (B, 1) 或 (B,)"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy()
        return log_prob, entropy


class HierarchicalActorLocal(nn.Module):
    """用户侧层次化策略：先决定是否分配，再选择服务器。与 mfac 版本不同，输入仅为局部观测。"""

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
        self.head_alloc = nn.Linear(hidden_size, 2)         # 是否分配
        self.head_server = nn.Linear(hidden_size, n_servers) # 选服务器

    def _build_joint(self, alloc_logits: torch.Tensor, server_logits: torch.Tensor):
        alloc_probs = F.softmax(alloc_logits, dim=-1)  # (B,2)
        server_probs = F.softmax(server_logits, dim=-1)  # (B,n_servers)
        p_no_alloc = alloc_probs[:, 0:1]
        p_alloc = alloc_probs[:, 1:2]
        probs_server = p_alloc * server_probs  # (B,n_servers)
        probs = torch.cat([p_no_alloc, probs_server], dim=-1)
        probs = torch.clamp(probs, min=1e-12)
        logits = torch.log(probs)
        return logits, probs

    def forward(self, obs: torch.Tensor):
        h = self.base(obs)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)
        joint_logits, _ = self._build_joint(alloc_logits, server_logits)
        return joint_logits

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        h = self.base(obs)
        alloc_logits = self.head_alloc(h)
        server_logits = self.head_server(h)

        alloc_dist = Categorical(logits=alloc_logits)
        alloc_action = alloc_dist.sample()
        log_prob_alloc = alloc_dist.log_prob(alloc_action)

        if alloc_action.item() == 0:
            return 0, log_prob_alloc

        if action_mask is not None:
            server_mask = action_mask[:, 1:].clone()
            server_logits = server_logits.masked_fill(~server_mask, -1e10)
            if (~server_mask).all():
                return 0, log_prob_alloc
        server_dist = Categorical(logits=server_logits)
        server_action = server_dist.sample()
        log_prob_server = server_dist.log_prob(server_action)
        final_action = server_action + 1
        return final_action.item(), log_prob_alloc + log_prob_server

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """计算批量动作的 log_prob 与熵。actions shape (B,) or (B,1)"""
        logits = self.forward(obs)  # (B, n_servers+1)
        log_probs_all = F.log_softmax(logits, dim=-1)
        probs_all = F.softmax(logits, dim=-1)
        actions = actions.long().unsqueeze(-1)
        selected_log_prob = log_probs_all.gather(1, actions).squeeze(-1)
        entropy = -(probs_all * log_probs_all).sum(dim=-1)
        return selected_log_prob, entropy


class CentralCritic(nn.Module):
    """中央 Critic：接收全局观测，输出状态价值。"""

    def __init__(self, global_state_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, global_state: torch.Tensor):
        return self.net(global_state) 