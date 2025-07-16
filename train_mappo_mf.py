"""
本文件实现了 MAPPO 的两种平均场 (Mean-Field) 变体，通过 `use_global` 参数控制。

1. `use_global=True`:
   - Critic 输入: 全局状态 (Global State) + 平均场 (Mean-Field)
   - Actor 输入:  局部观测 (Local Observation) + 平均场 (Mean-Field)

2. `use_global=False`:
   - Critic 输入: 局部观测 (Local Observation) + 平均场 (Mean-Field)
   - Actor 输入:  局部观测 (Local Observation) + 平均场 (Mean-Field)
    (类似 MFAC)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import logging
from datetime import datetime
import csv

from EdgeBatchEnv import EdgeBatchEnv
from model_mf import Actor, HierarchicalActor  # 具备均场输入的 Actor


# ---------------- 日志 ----------------

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
    )
    logging.info('Logger initialized.')


# ---------------- 网络 ----------------

class ValueNet(nn.Module):
    """简单的两层 MLP，用于输出状态价值 V(s)。"""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Trainer ----------------

class MAPPO_MF_Trainer:
    def __init__(self, env: EdgeBatchEnv, config: dict):
        self.env = env
        self.config = config
        self.use_global = config.get('use_global', True)
        self.device = 'cpu'

        # 维度
        self.n_users = env.n_users
        self.n_servers = env.n_servers
        self.n_agents = env.n_agents
        self.user_obs_dim = len(env._get_user_obs(0))
        self.server_obs_dim = len(env._get_server_obs(0))
        self.mf_dim = len(env.get_mean_field_state())
        self.global_state_dim = len(env.get_global_state())

        # Actor
        self.user_actors = [HierarchicalActor(self.user_obs_dim, self.mf_dim, self.n_servers).to(self.device)
                            for _ in range(self.n_users)]
        self.server_actors = [Actor(self.server_obs_dim, self.mf_dim, env.server_action_dim).to(self.device)
                              for _ in range(self.n_servers)]

        # Critic
        if self.use_global:
            self.critic = ValueNet(self.global_state_dim + self.mf_dim).to(self.device)
            self.local_critics = None
        else:
            self.critic = None
            self.local_critics = [
                ValueNet(self.user_obs_dim + self.mf_dim).to(self.device) for _ in range(self.n_users)
            ] + [
                ValueNet(self.server_obs_dim + self.mf_dim).to(self.device) for _ in range(self.n_servers)
            ]  # len = n_agents

        # 优化器
        actor_params = sum([list(a.parameters()) for a in self.user_actors + self.server_actors], [])
        self.actor_opt = optim.Adam(actor_params, lr=config['actor_lr'])
        if self.use_global:
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        else:
            local_params = sum([list(c.parameters()) for c in self.local_critics], [])
            self.critic_opt = optim.Adam(local_params, lr=config['critic_lr'])
        self.all_actor_params = actor_params

        logging.info("MAPPO-MF Trainer Initialized.")
        logging.info(f"Device: {self.device}")
        logging.info(f"User Obs Dim: {self.user_obs_dim}, Server Obs Dim: {self.server_obs_dim}")
        if self.use_global:
            logging.info(f"Global State Dim: {self.global_state_dim}")
        logging.info(f"Mean-field Dim: {self.mf_dim}")
        logging.info(f"Use Global Critic: {self.use_global}")

    # ------------ 主训练 ------------
    def train(self):
        for ep in range(self.config['num_episodes']):
            obs = self.env.reset()
            ep_rew, ep_aoi = 0, 0

            # 轨迹缓存
            obs_buf = [[] for _ in range(self.n_agents)]
            mf_buf = []
            act_buf = [[] for _ in range(self.n_agents)]
            logp_buf = [[] for _ in range(self.n_agents)]
            gstate_buf = []
            rew_buf, done_buf = [], []

            for t in range(self.config['episode_length']):
                mf = self.env.get_mean_field_state()
                mf_buf.append(mf)
                gstate = self.env.get_global_state()
                gstate_buf.append(gstate)

                masks = self.env.get_agent_action_mask()
                actions = []

                with torch.no_grad():
                    mf_t = torch.tensor(mf, dtype=torch.float32).unsqueeze(0)
                    # 用户
                    for i in range(self.n_users):
                        obs_t = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.tensor(masks[i], dtype=torch.bool).unsqueeze(0)
                        a, lp = self.user_actors[i].get_action(obs_t, mf_t, mask_t)
                        actions.append(a)
                        obs_buf[i].append(obs[i])
                        act_buf[i].append(a)
                        logp_buf[i].append(lp.item())
                    # 服务器
                    for i in range(self.n_servers):
                        idx = self.n_users + i
                        obs_t = torch.tensor(obs[idx], dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.tensor(masks[idx], dtype=torch.bool).unsqueeze(0)
                        a, lp = self.server_actors[i].get_action(obs_t, mf_t, mask_t)
                        actions.append(a)
                        obs_buf[idx].append(obs[idx])
                        act_buf[idx].append(a)
                        logp_buf[idx].append(lp.item())

                next_obs, reward, done, info = self.env.step(actions)
                ep_rew += reward
                ep_aoi += info.get('average_aoi', -1)

                rew_buf.append(reward)
                done_buf.append(done)

                obs = next_obs
                if done:
                    break

            # ---------- 计算优势 ----------
            rewards = np.array(rew_buf, dtype=np.float32)
            dones = np.array(done_buf, dtype=np.bool_)
            mf_arr = torch.tensor(np.asarray(mf_buf, dtype=np.float32))

            if self.use_global:
                gstate_arr = torch.tensor(np.asarray(gstate_buf, dtype=np.float32))
                critic_input = torch.cat([gstate_arr, mf_arr], dim=-1)
                with torch.no_grad():
                    values = self.critic(critic_input).squeeze(-1).cpu().numpy()

                adv, ret = self._gae(rewards, dones, values)
                adv_t = torch.tensor(adv, dtype=torch.float32)
                ret_t = torch.tensor(ret, dtype=torch.float32)
            else:
                # 每个 agent 单独
                values_agents = []
                for aid in range(self.n_agents):
                    obs_arr = torch.tensor(np.asarray(obs_buf[aid], dtype=np.float32))
                    critic_in = torch.cat([obs_arr, mf_arr], dim=-1)
                    with torch.no_grad():
                        v = self.local_critics[aid](critic_in).squeeze(-1).cpu().numpy()
                    values_agents.append(v)
                adv_agents, ret_agents = [], []
                for aid in range(self.n_agents):
                    adv, ret = self._gae(rewards, dones, values_agents[aid])
                    adv_agents.append(torch.tensor(adv, dtype=torch.float32))
                    ret_agents.append(torch.tensor(ret, dtype=torch.float32))

            # ---------- PPO 更新 ----------
            for _ in range(self.config['update_epochs']):
                idxs = np.arange(len(rewards))
                np.random.shuffle(idxs)
                for start in range(0, len(rewards), self.config['mini_batch_size']):
                    end = start + self.config['mini_batch_size']
                    mb_idx = idxs[start:end]

                    # Critic 更新
                    self.critic_opt.zero_grad()
                    if self.use_global:
                        v_pred = self.critic(critic_input[mb_idx]).squeeze(-1)
                        critic_loss = F.mse_loss(v_pred, ret_t[mb_idx])
                        critic_loss.backward()
                    else:
                        critic_loss = 0.0
                        for aid in range(self.n_agents):
                            obs_arr = torch.tensor(np.asarray(obs_buf[aid], dtype=np.float32))[mb_idx]
                            ci = torch.cat([obs_arr, mf_arr[mb_idx]], dim=-1)
                            v = self.local_critics[aid](ci).squeeze(-1)
                            loss = F.mse_loss(v, ret_agents[aid][mb_idx])
                            loss.backward()
                            critic_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.critic_opt.param_groups[0]['params'], self.config['max_grad_norm'])
                    self.critic_opt.step()

                    # Actor
                    self.actor_opt.zero_grad()
                    total_actor_loss = 0.0
                    for aid in range(self.n_agents):
                        mb_obs = torch.tensor([obs_buf[aid][i] for i in mb_idx], dtype=torch.float32)
                        mb_act = torch.tensor([act_buf[aid][i] for i in mb_idx], dtype=torch.int64)
                        old_lp = torch.tensor([logp_buf[aid][i] for i in mb_idx], dtype=torch.float32)
                        mb_mf = mf_arr[mb_idx]
                        actor = self.user_actors[aid] if aid < self.n_users else self.server_actors[aid - self.n_users]
                        # 重新计算 log_prob 与熵：需要均场输入
                        logits = actor(mb_obs, mb_mf)
                        log_probs_all = F.log_softmax(logits, dim=-1)
                        probs_all = F.softmax(logits, dim=-1)
                        new_lp = log_probs_all.gather(1, mb_act.unsqueeze(1)).squeeze(1)
                        ent = -(probs_all * log_probs_all).sum(dim=-1)

                        adv = adv_t[mb_idx] if self.use_global else adv_agents[aid][mb_idx]
                        # 局部模式下，对每个 agent 的优势再做标准化，提升稳定性
                        if not self.use_global:
                            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                        ratio = torch.exp(new_lp - old_lp)
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * adv
                        loss = -torch.min(surr1, surr2).mean() - self.config['entropy_coeff'] * ent.mean()
                        loss.backward()
                        total_actor_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.all_actor_params, self.config['max_grad_norm'])
                    self.actor_opt.step()

            # --- 日志记录 ---
            self.save_completed_tasks_log(ep)
            avg_aoi = ep_aoi / len(rew_buf)
            completed_tasks_this_episode = len(self.env.completed_tasks_log)
            logging.info(f"Episode: {ep + 1}/{self.config['num_episodes']}, "
                         f"Reward: {ep_rew:.2f}, Avg AoI: {avg_aoi:.2f}, Completed Tasks: {completed_tasks_this_episode}")

    # ---------- GAE ----------
    def _gae(self, rewards, dones, values):
        adv = np.zeros_like(rewards)
        ret = np.zeros_like(rewards)
        gae = 0.0
        next_v = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config['gamma'] * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[t]) * gae
            adv[t] = gae
            ret[t] = adv[t] + values[t]
            next_v = values[t]
        return adv, ret

    def save_completed_tasks_log(self, epoch: int):
        """将 env.completed_tasks_log 写入 CSV，保存在 log_dir/completed_tasks 内。"""
        if not self.env.completed_tasks_log:
            logging.info("No tasks completed in this episode; skip writing completed_tasks log.")
            return

        base_path = os.path.join(self.config['log_dir'], 'completed_tasks')
        os.makedirs(base_path, exist_ok=True)
        filepath = os.path.join(base_path, f"completed_tasks_log_{epoch}.csv")

        headers = ['task_id', 'user_id', 'server_id', 'generation_time', 'completion_time', 'latency']
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for task in self.env.completed_tasks_log:
                writer.writerow([
                    task['task_id'],
                    task['user_id'],
                    task['server_id'],
                    task['generation_time'],
                    task['completion_time'],
                    task['latency']
                ])


if __name__ == '__main__':
    # Scale 表示 user 和 server 放大的倍数
    # Scale = 3
    config = {
        'n_users': 7 * 3,
        'n_servers': 3 * 3,
        'batch_proc_time': {'base': 3, 'per_task': 2},
        'max_batch_size': 3,
        'num_episodes': 1000,
        'episode_length': 400,
        'actor_lr': 1e-5,
        'critic_lr': 1e-5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'update_epochs': 4,
        'mini_batch_size': 128,
        'entropy_coeff': 0.01,
        'max_grad_norm': 1.0,
        # 'use_global': False,
        'use_global': True,  # 是否使用全局状态作为 Critic 输入
        'log_dir': '',  # 占位，稍后根据参数自动生成
    }

    # 根据配置动态生成日志目录，格式: logs/mappo_mf_⟨Nuser⟩_⟨Nserver⟩_⟨Global|Local⟩_⟨Lr⟩
    suffix = f"{config['n_users']}_{config['n_servers']}_{'Global' if config['use_global'] else 'Local'}_{config['actor_lr']}"
    config['log_dir'] = f"logs/mappo_mf_{suffix}"

    setup_logger(config['log_dir'])
    # 记录所有配置参数
    logging.info("--- Training Configuration ---")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info("--------------------------")

    env = EdgeBatchEnv(
        n_users=config['n_users'],
        n_servers=config['n_servers'],
        batch_proc_time=config['batch_proc_time'],
        max_batch_size=config['max_batch_size']
    )
    trainer = MAPPO_MF_Trainer(env, config)
    trainer.train()
    
    logging.info("Training finished.")