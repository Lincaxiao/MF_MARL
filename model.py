import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal
from torch.distributions import Categorical
import torch.nn.functional as F
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# hyperparameters
lr = config['hyperparameters']['learning_rate']

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
print("Using device:", DEVICE)

# --------------------------------------------------
# 权重初始化函数（Orthogonal + ReLU 增益）
# --------------------------------------------------

def init_weights(m):
    """对 Linear 层做 Orthogonal 初始化并将 bias 置 0."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, act_dim)
        # 权重初始化
        self.apply(init_weights)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x):  # [B, obs_dim]
        return self.out(self.fc(x))


class Critic(nn.Module):
    def __init__(self, global_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(global_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # 权重初始化
        self.apply(init_weights)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_low, action_high, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        # self.log_std = nn.Parameter(torch.zeros(1))  # 原始初始化方式
        self.log_std = nn.Parameter(torch.ones(1) * -0.5)  # 更小的初始 std
        self.action_low = action_low
        self.action_high = action_high
        # 权重初始化
        self.apply(init_weights)
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mu_head.bias, 0.0)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std

    # def sample(self, x):
    #     mu, std = self.forward(x)
    #     dist = torch.distributions.Normal(mu, std)
    #     action = dist.rsample()
    #     action_clipped = torch.clamp(action, self.action_low, self.action_high)
    #     log_prob = dist.log_prob(action)
    #     return action_clipped.squeeze(-1), log_prob.squeeze(-1), mu.squeeze(-1), std.squeeze(-1)
    def sample(self, x):
        # 原先的 mu/std 计算不变
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)

        # re-param 采样
        z = dist.rsample()                     # raw action
        log_prob_z = dist.log_prob(z)          # 对应的 log prob

        # squash 到 (-1,1)
        y = torch.tanh(z)

        # Jacobian 校正：logp = logπ(z) - ∑ log(1 - tanh(z)^2)
        log_prob = log_prob_z - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 线性映射到 [low, high]
        action = self.action_low + (y + 1) * 0.5 * (self.action_high - self.action_low)

        # 保持原来 squeeze 行为
        return (
            action.squeeze(-1),
            log_prob.squeeze(-1),
            mu.squeeze(-1),
            std.squeeze(-1),
        )


class SACGaussianActor(nn.Module):
    """SAC Actor with squashed Gaussian policy."""
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        # 权重初始化
        self.apply(init_weights)
        nn.init.uniform_(self.mu_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mu_layer.bias, 0.0)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.log_std_layer.bias, -0.5)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = normal.Normal(mu, std)

        if deterministic:
            # For evaluation, return deterministic action (mean)
            action = mu
        else:
            action = dist.rsample()

        if with_logprob:
            # Compute log_prob from the squashed distribution
            log_prob = dist.log_prob(action)
            # Apply correction for the tanh squashing function
            log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6)
            log_prob = log_prob.sum(axis=-1)
        else:
            log_prob = None

        action = torch.tanh(action)
        return action, log_prob

class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, global_dim, lr=lr, continuous=False, action_low=1.0, action_high=10.0):
        self.continuous = continuous
        if continuous:
            self.actor = GaussianActor(obs_dim, action_low, action_high).to(DEVICE)
        else:
            self.actor = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic = Critic(global_dim).to(DEVICE)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def select(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            if self.continuous:
                action, logp, mu, std = self.actor.sample(obs)
                return float(action.cpu().item()), float(logp.cpu().item())
            else:
                logits = self.actor(obs)
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
                return int(a.cpu().detach().item()), logp.cpu().detach().item()

    def get_value(self, global_state):
        global_state = torch.tensor(global_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            v = self.critic(global_state)
        return v.cpu().item()

    def train(self, traj, clipr=0.2, epochs=4, ent_coef=0, batch_size=64):
        if not traj:
            return
        # [obs, act, logp, reward, val, gs, adv, return, done]
        obs = torch.tensor(np.array([t[0] for t in traj]), dtype=torch.float32, device=DEVICE)
        acts = torch.tensor([t[1] for t in traj], dtype=torch.float32 if self.continuous else torch.long, device=DEVICE)
        logps_old = torch.tensor([t[2] for t in traj], dtype=torch.float32, device=DEVICE)
        adv = torch.tensor([t[6] for t in traj], dtype=torch.float32, device=DEVICE)
        returns = torch.tensor([t[7] for t in traj], dtype=torch.float32, device=DEVICE)
        glo_obs = torch.tensor(np.array([t[5] for t in traj]), dtype=torch.float32, device=DEVICE)

        n = len(traj)
        idlist = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idlist)
            for b in range(0, n, batch_size):
                bb = idlist[b:b + batch_size]
                if self.continuous:
                    mu, std = self.actor.forward(obs[bb])
                    dist = torch.distributions.Normal(mu, std)
                    logps = dist.log_prob(acts[bb]).squeeze(-1) # negative
                    entropy = dist.entropy().mean()
                else:
                    logits = self.actor(obs[bb])
                    dist = Categorical(logits=logits)
                    logps = dist.log_prob(acts[bb])
                    entropy = dist.entropy().mean()
                ratio = torch.exp(logps - logps_old[bb])
                surr1 = ratio * adv[bb]
                surr2 = torch.clamp(ratio, 1 - clipr, 1 + clipr) * adv[bb]
                actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy # entropy_coef=0
                # critic loss
                values = self.critic(glo_obs[bb]).squeeze(-1)
                critic_loss = ((values - returns[bb]) ** 2).mean()

                loss = actor_loss + 0.2 * critic_loss # 0.1, 0.2
                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                loss.backward() # loss / grad_acc_steps * lr if step % grad_acc_steps == 0
                self.optim_actor.step()
                self.optim_critic.step()

# [obs, act, logp, reward, val, gs, adv, return, done]
def compute_gae(traj, gamma=0.99, lam=0.95):
    v_tp1 = 0
    traj_ = []
    rew = [t[3] for t in traj]
    v = [t[4] for t in traj] + [0.0] # TODO：terminal value
    advs = []
    lastgaelam = 0
    for t in reversed(range(len(traj))):
        non_terminal = 1.0 - float(traj[t][8])  # done
        delta = rew[t] + gamma * v[t + 1] * non_terminal - v[t]
        lastgaelam = delta + gamma * lam * non_terminal * lastgaelam
        advs.insert(0, lastgaelam)
    rets = [a + v[i] for i, a in enumerate(advs)]
    for i in range(len(traj)):
        traj[i] = traj[i][:6] + (advs[i], rets[i]) + (traj[i][8],)
    return traj

############################################
# MASAC
############################################

class ReplayBuffer:
    """CTDE ReplayBuffer: 既保存各智能体局部观测，也保存全局观测，方便中心化 Critic 训练"""
    def __init__(self, obs_dim, act_dim, global_dim, capacity=50000):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        # 全局观测
        self.global_obs_buf = np.zeros((capacity, global_dim), dtype=np.float32)
        self.next_global_obs_buf = np.zeros((capacity, global_dim), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def push(self, obs, act, rew, next_obs, done, global_obs, next_global_obs):
        """写入一条经历（包含全局信息）"""
        self.obs_buf[self.ptr] = obs
        # 确保动作、奖励、done 具备正确形状
        self.acts_buf[self.ptr] = np.asarray(act, dtype=np.float32)
        self.rews_buf[self.ptr] = np.asarray(rew, dtype=np.float32)
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = np.asarray(done, dtype=np.float32)
        self.global_obs_buf[self.ptr] = global_obs
        self.next_global_obs_buf[self.ptr] = next_global_obs
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     acts=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     done=self.done_buf[idxs],
                     global_obs=self.global_obs_buf[idxs],
                     next_global_obs=self.next_global_obs_buf[idxs])
        # 转成 torch 张量
        for k in batch:
            batch[k] = torch.tensor(batch[k], dtype=torch.float32, device=DEVICE)
        return batch

    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    """SAC 的 Q 网络 (Critic)"""
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 权重初始化
        self.apply(init_weights)
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class MASACAgent:
    """多智能体 Soft Actor-Critic"""
    def __init__(self, obs_dim, act_dim, global_dim, action_low=-1.0, action_high=1.0,
                 lr=lr, gamma=0.99, tau=0.005, alpha=0.2,
                 automatic_entropy_tuning=True,
                 buffer_size=5_000_0, hidden_dim=128):
        self.obs_dim = obs_dim
        self.act_dim = act_dim if isinstance(act_dim, int) else int(act_dim)
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Actor
        self.actor = SACGaussianActor(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        # Critic
        self.q1 = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2 = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        # Target Critic
        self.q1_target = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2_target = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Entropy 温度参数自动调整
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod((act_dim,)).item()  # 推荐的 target entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        # Replay Buffer（CTDE，需要 global_dim）
        assert 'global_dim' in locals() or 'global_dim' in globals(), "需提供 global_dim"
        self.replay_buffer = ReplayBuffer(obs_dim, self.act_dim, global_dim, capacity=buffer_size)

    @torch.no_grad()
    def select(self, obs, evaluate=False):
        """Select action for interaction."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action_tanh, log_prob = self.actor(obs_tensor, deterministic=evaluate, with_logprob=True)
        
        # Scale action to environment range
        scaled_action = self.action_low + (action_tanh + 1) * 0.5 * (self.action_high - self.action_low)
        
        return float(scaled_action.cpu().item()), float(log_prob.cpu().item())

    def store(self, obs, act, reward, next_obs, done, global_obs, next_global_obs):
        """写入含全局信息的数据"""
        self.replay_buffer.push(obs, act, reward, next_obs, done, global_obs, next_global_obs)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, batch_size=256, updates=1):
        if len(self.replay_buffer) < batch_size:
            return  # 数据不足时跳过
        for _ in range(updates):
            batch = self.replay_buffer.sample(batch_size)
            obs, actions, rewards, next_obs, dones = batch['obs'], batch['acts'], batch['rews'], batch['next_obs'], batch['done']
            glob_obs, glob_next_obs = batch['global_obs'], batch['next_global_obs']

            # 转换 actions 形状 [B, act_dim]
            actions = actions.view(-1, self.act_dim)

            # ----------------------- 更新 Q 网络 -----------------------
            with torch.no_grad():
                next_action_tanh, next_log_prob = self.actor(next_obs, with_logprob=True)
                next_action_scaled = self.action_low + (next_action_tanh + 1) * 0.5 * (self.action_high - self.action_low)
                
                target_q1 = self.q1_target(glob_next_obs, next_action_scaled)
                target_q2 = self.q2_target(glob_next_obs, next_action_scaled)
                target_q_min = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target_q = rewards + (1 - dones) * self.gamma * target_q_min.unsqueeze(-1)
                target_q = target_q.squeeze(-1)

            current_q1 = self.q1(glob_obs, actions)
            current_q2 = self.q2(glob_obs, actions)
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            # ----------------------- 更新 Actor 网络 -----------------------
            new_actions_tanh, log_prob = self.actor(obs, with_logprob=True)
            new_actions_scaled = self.action_low + (new_actions_tanh + 1) * 0.5 * (self.action_high - self.action_low)

            q1_new = self.q1(glob_obs, new_actions_scaled)
            q2_new = self.q2(glob_obs, new_actions_scaled)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ------------------------ 调整 alpha ------------------------
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp().item()

            # ------------------------ 软更新目标网络 ------------------------
            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)

    def get_value(self, obs):
        """SAC 中不需要单独的 value，但为了接口兼容返回 Q1 的值"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self.actor.sample(obs_tensor)
            value = self.q1(obs_tensor, action.view(1, -1))
        return value.cpu().item()


################################################
# TD3
################################################

class DeterministicActor(nn.Module):
    """TD3 中用于输出确定性动作的 Actor 网络"""
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        # 权重初始化
        self.apply(init_weights)
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)  # Linear 层在倒数第2位
        nn.init.constant_(self.net[-2].bias, 0.0)

    def forward(self, obs):
        # 输出范围 [-1,1]
        return self.net(obs)


class MATD3Agent:
    """TD3"""
    def __init__(self, obs_dim, act_dim, global_dim, action_low=-1.0, action_high=1.0,
                 lr=lr, gamma=0.99, tau=0.005,
                 expl_noise=0.1, policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 buffer_size=1_000_000, hidden_dim=128):
        self.obs_dim = obs_dim
        self.act_dim = act_dim if isinstance(act_dim, int) else int(act_dim)
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0


        # Actor & Critics
        self.actor = DeterministicActor(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.actor_target = DeterministicActor(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2 = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q1_target = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2_target = QNetwork(global_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Replay Buffer（CTDE，需要 global_dim）
        assert 'global_dim' in locals() or 'global_dim' in globals(), "需提供 global_dim"
        self.replay_buffer = ReplayBuffer(obs_dim, self.act_dim, global_dim, capacity=buffer_size)

        self.total_it = 0  # 用于延迟策略更新

    @torch.no_grad()
    def select(self, obs, evaluate=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mu_tanh = self.actor(obs_tensor) # actor输出的是tanh后的值
        
        # 将tanh后的动作缩放到实际环境范围
        mu_scaled = mu_tanh * self.action_scale + self.action_bias

        if not evaluate:
            # 在真实的动作空间添加噪声
            noise = torch.normal(0, self.action_scale * self.expl_noise, size=mu_scaled.shape, device=DEVICE)
            mu_scaled += noise
        
        # 裁剪到有效范围
        action_clipped = torch.clamp(mu_scaled, self.action_low, self.action_high)
        
        # 确定性策略没有 log_prob，返回 0.0 占位
        return float(action_clipped.cpu().item()), 0.0


    def store(self, obs, act, reward, next_obs, done, global_obs, next_global_obs):
        """写入含全局信息的数据"""
        self.replay_buffer.push(obs, act, reward, next_obs, done, global_obs, next_global_obs)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, batch_size=256, updates=1):
        if len(self.replay_buffer) < batch_size:
            return
        for _ in range(updates):
            self.total_it += 1
            batch = self.replay_buffer.sample(batch_size)
            obs, actions, rewards, next_obs, dones = batch['obs'], batch['acts'], batch['rews'], batch['next_obs'], batch['done']
            glob_obs, glob_next_obs = batch['global_obs'], batch['next_global_obs']
            actions = actions.view(-1, self.act_dim)
            rewards = rewards.squeeze(-1) # 保证形状为 [B]
            dones = dones.squeeze(-1) # 保证形状为 [B]

            # --- 目标 Q 值计算 ---
            with torch.no_grad():
                # ***** 已修正: 目标策略平滑 (与 CleanRL 对齐的最优实现) *****
                # 1. 从目标 actor 获取 [-1, 1] 范围的动作，并立刻映射到真实动作空间
                next_action_scaled = self.actor_target(next_obs) * self.action_scale + self.action_bias
                
                # 2. 生成噪声，其尺度与真实动作空间匹配，并进行裁剪
                # 注意：policy_noise 和 noise_clip 是为归一化动作空间设计的超参
                noise = (
                    torch.randn_like(next_action_scaled) * self.policy_noise * self.action_scale
                ).clamp(-self.noise_clip * self.action_scale, self.noise_clip * self.action_scale)
                
                # 3. 在真实动作空间中添加噪声，并进行最终裁剪
                next_action_clipped = torch.clamp(next_action_scaled + noise, self.action_low, self.action_high)

                # 4. 计算目标 Q 值
                target_q1 = self.q1_target(glob_next_obs, next_action_clipped)
                target_q2 = self.q2_target(glob_next_obs, next_action_clipped)
                min_target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - dones) * self.gamma * min_target_q
                
            # --- Critic 更新 ---
            current_q1 = self.q1(glob_obs, actions)
            current_q2 = self.q2(glob_obs, actions)
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)
            q_loss = q1_loss + q2_loss

            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            q_loss.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.step()

            # --- 延迟策略更新 ---
            if self.total_it % self.policy_delay == 0:
                # Actor 损失计算
                actor_actions_tanh = self.actor(obs)
                actor_actions_scaled = actor_actions_tanh * self.action_scale + self.action_bias
                actor_loss = -self.q1(glob_obs, actor_actions_scaled).mean()

                # Actor 更新
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 目标网络软更新
                self.soft_update(self.actor, self.actor_target)
                self.soft_update(self.q1, self.q1_target)
                self.soft_update(self.q2, self.q2_target)

    def get_value(self, obs):
        """返回 Q1 的估计值（与其它 agent 接口统一）"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            act_tanh = self.actor(obs_tensor)
            act_scaled = act_tanh * self.action_scale + self.action_bias
            # 注意：这里的 global_obs 在单智能体场景下就是 obs
            # 在多智能体下，这里可能需要传入 global_obs
            value = self.q1(obs_tensor, act_scaled)
        return value.cpu().item()


