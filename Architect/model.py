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


