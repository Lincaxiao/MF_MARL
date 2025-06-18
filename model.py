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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):  # [B, obs_dim]
        return self.out(self.fc(x))


class Critic(nn.Module):
    def __init__(self, global_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(global_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_low, action_high, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        action_clipped = torch.clamp(action, self.action_low, self.action_high)
        log_prob = dist.log_prob(action)
        return action_clipped.squeeze(-1), log_prob.squeeze(-1), mu.squeeze(-1), std.squeeze(-1)


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
# 以下为 MASAC（多智能体 SAC）实现
############################################

class ReplayBuffer:
    """简单的循环缓冲区，用于存储 S,A,R,S',done 五元组"""
    def __init__(self, obs_dim, act_dim, capacity=100000):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def push(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     acts=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     done=self.done_buf[idxs])
        # 转成 torch 张量
        for k in batch:
            batch[k] = torch.tensor(batch[k], dtype=torch.float32, device=DEVICE)
        return batch

    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    """SAC 的 Q 网络 (Critic)"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class MASACAgent:
    """多智能体 Soft Actor-Critic（这里实现为单智能体接口，可在多智能体框架中实例化多个）"""
    def __init__(self, obs_dim, act_dim, action_low=-1.0, action_high=1.0,
                 lr=lr, gamma=0.99, tau=0.005, alpha=0.2,
                 automatic_entropy_tuning=True,
                 buffer_size=1_000_000, hidden_dim=256):
        self.obs_dim = obs_dim
        self.act_dim = act_dim if isinstance(act_dim, int) else int(act_dim)
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Actor
        self.actor = GaussianActor(obs_dim, action_low, action_high, hidden_dim).to(DEVICE)
        # Critic
        self.q1 = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2 = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        # Target Critic
        self.q1_target = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2_target = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
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

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(obs_dim, self.act_dim, capacity=buffer_size)

    @torch.no_grad()
    def select(self, obs, evaluate=False):
        """与 PPO 接口保持一致，返回动作和 log_prob（评估时可忽略）"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if evaluate:
            mu, _ = self.actor.forward(obs_tensor)
            action = mu
            log_prob = torch.zeros(1, device=DEVICE)
        else:
            action, log_prob, _, _ = self.actor.sample(obs_tensor)
        action = action.clamp(self.action_low, self.action_high)
        return float(action.cpu().item()), float(log_prob.cpu().item())

    def store(self, obs, act, reward, next_obs, done):
        self.replay_buffer.push(obs, [act], [reward], next_obs, [done])

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, batch_size=256, updates=1):
        if len(self.replay_buffer) < batch_size:
            return  # 数据不足时跳过
        for _ in range(updates):
            batch = self.replay_buffer.sample(batch_size)
            obs, actions, rewards, next_obs, dones = batch['obs'], batch['acts'], batch['rews'], batch['next_obs'], batch['done']

            # 转换 actions 形状 [B, act_dim]
            actions = actions.view(-1, self.act_dim)

            # ----------------------- 更新 Q 网络 -----------------------
            with torch.no_grad():
                next_action, next_log_prob, _, _ = self.actor.sample(next_obs)
                next_action = next_action.view(-1, self.act_dim)
                target_q1 = self.q1_target(next_obs, next_action)
                target_q2 = self.q2_target(next_obs, next_action)
                target_q_min = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target_q = rewards + (1 - dones) * self.gamma * target_q_min.unsqueeze(-1)
                target_q = target_q.squeeze(-1)

            current_q1 = self.q1(obs, actions)
            current_q2 = self.q2(obs, actions)
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            # ----------------------- 更新 Actor 网络 -----------------------
            new_actions, log_prob, _, _ = self.actor.sample(obs)
            new_actions = new_actions.view(-1, self.act_dim)
            q1_new = self.q1(obs, new_actions)
            q2_new = self.q2(obs, new_actions)
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
# 以下为 MATD3（多智能体 TD3）实现
################################################

class DeterministicActor(nn.Module):
    """TD3 中用于输出确定性动作的 Actor 网络"""
    def __init__(self, obs_dim, act_dim, action_low=-1.0, action_high=1.0, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))

    def forward(self, obs):
        # 输出范围 [-1,1]，然后映射到指定区间
        raw_action = self.net(obs)
        scaled_action = (raw_action + 1.0) / 2.0 * (self.action_high - self.action_low) + self.action_low
        return scaled_action


class MATD3Agent:
    """多智能体 TD3，实现接口与 MASAC/MAPPO 一致"""
    def __init__(self, obs_dim, act_dim, action_low=-1.0, action_high=1.0,
                 lr=lr, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 buffer_size=1_000_000, hidden_dim=256):
        self.obs_dim = obs_dim
        self.act_dim = act_dim if isinstance(act_dim, int) else int(act_dim)
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        # Actor & Critics
        self.actor = DeterministicActor(obs_dim, self.act_dim, action_low, action_high, hidden_dim).to(DEVICE)
        self.actor_target = DeterministicActor(obs_dim, self.act_dim, action_low, action_high, hidden_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2 = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q1_target = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q2_target = QNetwork(obs_dim, self.act_dim, hidden_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(obs_dim, self.act_dim, capacity=buffer_size)

        self.total_it = 0  # 用于延迟策略更新

    @torch.no_grad()
    def select(self, obs, noise_scale=0.0):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = self.actor(obs_tensor).squeeze(0)
        if noise_scale > 0.0:
            noise = torch.normal(mean=0, std=noise_scale, size=action.shape, device=DEVICE)
            action = action + noise
        action = action.clamp(self.action_low, self.action_high)
        # 为兼容性，返回 log_prob=0
        return float(action.cpu().item()), 0.0

    def store(self, obs, act, reward, next_obs, done):
        self.replay_buffer.push(obs, [act], [reward], next_obs, [done])

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
            actions = actions.view(-1, self.act_dim)

            # 生成带噪声的下一个动作（目标策略平滑）
            with torch.no_grad():
                noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_obs) + noise).clamp(self.action_low, self.action_high)
                target_q1 = self.q1_target(next_obs, next_action)
                target_q2 = self.q2_target(next_obs, next_action)
                target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2).unsqueeze(-1)
                target_q = target_q.squeeze(-1)

            # 更新 Critic
            current_q1 = self.q1(obs, actions)
            current_q2 = self.q2(obs, actions)
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            # 延迟策略更新
            if self.total_it % self.policy_delay == 0:
                actor_actions = self.actor(obs)
                actor_loss = -self.q1(obs, actor_actions).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 更新目标网络
                self.soft_update(self.actor, self.actor_target)
                self.soft_update(self.q1, self.q1_target)
                self.soft_update(self.q2, self.q2_target)

    def get_value(self, obs):
        """返回 Q1 的估计值（与其它 agent 接口统一）"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            act = self.actor(obs_tensor)
            value = self.q1(obs_tensor, act)
        return value.cpu().item()


