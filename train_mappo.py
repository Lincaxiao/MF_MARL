import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import logging
from datetime import datetime
import csv

from EdgeBatchEnv import EdgeBatchEnv
from model_mappo import ActorLocal, HierarchicalActorLocal, CentralCritic


# --- 日志配置，与 train_mf.py 保持一致 ---
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logger initialized.")


class MAPPO_Trainer:
    """中央训练 / 分散执行的 MAPPO 实现。"""

    def __init__(self, env: EdgeBatchEnv, config: dict):
        self.env = env
        self.config = config
        self.device = "cpu"

        # --- 环境维度信息 ---
        self.n_users = env.n_users
        self.n_servers = env.n_servers
        self.n_agents = env.n_agents

        # 局部观测维度
        self.user_obs_dim = len(env._get_user_obs(0))
        self.server_obs_dim = len(env._get_server_obs(0))

        # 动作维度
        self.user_action_dim = env.user_action_dim  # n_servers+1
        self.server_action_dim = env.server_action_dim  # 2

        # 全局观测维度
        self.global_state_dim = len(env.get_global_state())

        # --- 模型初始化 ---
        self.user_actors = [HierarchicalActorLocal(self.user_obs_dim, self.n_servers).to(self.device)
                            for _ in range(self.n_users)]
        self.server_actors = [ActorLocal(self.server_obs_dim, self.server_action_dim).to(self.device)
                              for _ in range(self.n_servers)]
        self.critic = CentralCritic(self.global_state_dim).to(self.device)

        # --- 优化器 ---
        actor_params = sum([list(a.parameters()) for a in self.user_actors + self.server_actors], [])
        self.actor_optimizer = optim.Adam(actor_params, lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        self.all_actor_params = actor_params

        logging.info("MAPPO Trainer Initialized.")
        logging.info(f"Device: {self.device}")
        logging.info(f"User Obs Dim: {self.user_obs_dim}, Server Obs Dim: {self.server_obs_dim}")
        logging.info(f"Global State Dim: {self.global_state_dim}")

    # ------------------------------------------------------------------
    # 主要训练循环
    # ------------------------------------------------------------------
    def train(self):
        for episode in range(self.config['num_episodes']):
            obs = self.env.reset()
            episode_reward = 0
            episode_aoi = 0

            # 轨迹缓存（按 agent 存储）
            obs_buf = [[] for _ in range(self.n_agents)]
            act_buf = [[] for _ in range(self.n_agents)]
            logp_buf = [[] for _ in range(self.n_agents)]
            global_state_buf = []
            rew_buf = []
            done_buf = []

            # ----------------------------------------------------------
            for t in range(self.config['episode_length']):
                global_state = self.env.get_global_state()
                global_state_buf.append(global_state)

                action_masks = self.env.get_agent_action_mask()
                actions = []

                with torch.no_grad():
                    # 用户动作采样
                    for i in range(self.n_users):
                        obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                        mask_tensor = torch.tensor(action_masks[i], dtype=torch.bool).unsqueeze(0)
                        a, logp = self.user_actors[i].get_action(obs_tensor, mask_tensor)
                        actions.append(a)
                        obs_buf[i].append(obs[i])
                        act_buf[i].append(a)
                        logp_buf[i].append(logp.item())

                    # 服务器动作采样
                    for i in range(self.n_servers):
                        idx = self.n_users + i
                        obs_tensor = torch.tensor(obs[idx], dtype=torch.float32).unsqueeze(0)
                        mask_tensor = torch.tensor(action_masks[idx], dtype=torch.bool).unsqueeze(0)
                        a, logp = self.server_actors[i].get_action(obs_tensor, mask_tensor)
                        actions.append(a)
                        obs_buf[idx].append(obs[idx])
                        act_buf[idx].append(a)
                        logp_buf[idx].append(logp.item())

                # 与环境交互
                next_obs, reward, done, info = self.env.step(actions)
                episode_reward += reward
                episode_aoi += info.get('average_aoi', -1)

                rew_buf.append(reward)
                done_buf.append(done)

                obs = next_obs
                if done:
                    break

            # ----------------------------------------------------------
            # 计算 GAE Advantage
            # ----------------------------------------------------------
            # 先将列表转换为 numpy.ndarray，再转为 Tensor，可显著加速
            global_states = torch.from_numpy(np.asarray(global_state_buf, dtype=np.float32))
            with torch.no_grad():
                values = self.critic(global_states).squeeze(-1).cpu().numpy()

            rewards = np.array(rew_buf, dtype=np.float32)
            dones = np.array(done_buf, dtype=np.bool_)

            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            gae = 0.0
            next_value = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.config['gamma'] * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
                next_value = values[t]

            advantages_t = torch.tensor(advantages, dtype=torch.float32)
            returns_t = torch.tensor(returns, dtype=torch.float32)

            # 标准化 advantage
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # ----------------------------------------------------------
            # PPO 更新
            # ----------------------------------------------------------
            for _ in range(self.config['update_epochs']):
                idxs = np.arange(len(rewards))
                np.random.shuffle(idxs)
                for start in range(0, len(rewards), self.config['mini_batch_size']):
                    end = start + self.config['mini_batch_size']
                    mb_idx = idxs[start:end]

                    # ---- Critic 更新 ----
                    mb_states = global_states[mb_idx]
                    mb_returns = returns_t[mb_idx]
                    pred_values = self.critic(mb_states).squeeze(-1)
                    critic_loss = F.mse_loss(pred_values, mb_returns)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm'])
                    self.critic_optimizer.step()

                    # ---- Actor 更新 ----
                    total_actor_loss = 0.0
                    for agent_id in range(self.n_agents):
                        # 批次数据整理
                        mb_obs = torch.tensor([obs_buf[agent_id][i] for i in mb_idx], dtype=torch.float32)
                        mb_act = torch.tensor([act_buf[agent_id][i] for i in mb_idx], dtype=torch.int64)
                        old_logp = torch.tensor([logp_buf[agent_id][i] for i in mb_idx], dtype=torch.float32)
                        mb_adv = advantages_t[mb_idx]

                        if agent_id < self.n_users:
                            actor = self.user_actors[agent_id]
                        else:
                            actor = self.server_actors[agent_id - self.n_users]

                        new_logp, entropy = actor.evaluate_actions(mb_obs, mb_act)
                        ratio = torch.exp(new_logp - old_logp)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * mb_adv
                        actor_loss = -torch.min(surr1, surr2).mean() - self.config['entropy_coeff'] * entropy.mean()
                        total_actor_loss += actor_loss

                    self.actor_optimizer.zero_grad()
                    total_actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.all_actor_params, self.config['max_grad_norm'])
                    self.actor_optimizer.step()

            # ----------------------------------------------------------
            # 保存当回合完成任务日志
            # ----------------------------------------------------------
            self.save_completed_tasks_log(episode)

            # ----------------------------------------------------------
            # 日志记录
            # ----------------------------------------------------------
            avg_aoi = episode_aoi / len(rew_buf)
            logging.info(f"Episode: {episode + 1}/{self.config['num_episodes']}, "
                         f"Reward: {episode_reward:.2f}, Avg AoI: {avg_aoi:.2f}, Steps: {len(rew_buf)}")

    # ------------------------------------------------------------------
    # 任务完成日志保存
    # ------------------------------------------------------------------
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
        logging.info(f"Saved completed_tasks log to {filepath}")


if __name__ == '__main__':
    # --- 配置 ---
    config = {
        'n_users': 28,
        'n_servers': 12,
        'batch_proc_time': {'base': 3, 'per_task': 2},
        'max_batch_size': 12,
        'num_episodes': 200,
        'episode_length': 1000,
        'actor_lr': 5e-5,
        'critic_lr': 5e-5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'update_epochs': 4,
        'mini_batch_size': 256,
        'entropy_coeff': 0.01,
        'max_grad_norm': 1.0,
        'log_dir': f'logs/mappo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    setup_logger(config['log_dir'])

    env = EdgeBatchEnv(
        n_users=config['n_users'],
        n_servers=config['n_servers'],
        batch_proc_time=config['batch_proc_time'],
        max_batch_size=config['max_batch_size']
    )

    trainer = MAPPO_Trainer(env, config)
    trainer.train()

    logging.info("Training finished.") 