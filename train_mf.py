import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import logging
from datetime import datetime
import csv
from dataclasses import asdict

from EdgeBatchEnv import EdgeBatchEnv
from model_mf import Actor, Critic, ReplayBuffer, HierarchicalActor


# --- 日志配置 ---
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


# --- 训练主类 ---
class MFAC_Trainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # --- 获取环境维度信息 ---
        self.n_users = env.n_users
        self.n_servers = env.n_servers
        self.n_agents = env.n_agents

        # 动态获取obs维度
        self.user_obs_dim = len(env._get_user_obs(0))
        self.server_obs_dim = len(env._get_server_obs(0))

        # 动作维度
        self.user_action_dim = env.user_action_dim
        self.server_action_dim = env.server_action_dim

        # 平均场维度
        self.mean_field_dim = len(env.get_mean_field_state())

        # --- 初始化 Actor 和 Critic 网络 ---
        # 用户使用层次化策略网络，解除“是否分配”与“选服务器”耦合
        self.user_actors = [HierarchicalActor(self.user_obs_dim, self.mean_field_dim, self.n_servers).to(self.device)
                            for _ in range(self.n_users)]
        self.user_critics = [Critic(self.user_obs_dim, self.mean_field_dim, self.user_action_dim).to(self.device) for _
                             in range(self.n_users)]
        self.server_actors = [Actor(self.server_obs_dim, self.mean_field_dim, self.server_action_dim).to(self.device)
                              for _ in range(self.n_servers)]
        self.server_critics = [Critic(self.server_obs_dim, self.mean_field_dim, self.server_action_dim).to(self.device)
                               for _ in range(self.n_servers)]

        all_actor_params = sum([list(actor.parameters()) for actor in self.user_actors + self.server_actors], [])
        all_critic_params = sum([list(critic.parameters()) for critic in self.user_critics + self.server_critics], [])

        self.actor_optimizer = optim.Adam(all_actor_params, lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(all_critic_params, lr=config['critic_lr'])

        self.buffer = ReplayBuffer(config['buffer_capacity'])

        self.smooth_mean_field = self.env.get_mean_field_state()

        self.all_completed_tasks = []

        logging.info("MFAC Trainer Initialized.")
        logging.info(f"Device: {self.device}")
        logging.info(f"User Obs Dim: {self.user_obs_dim}, Server Obs Dim: {self.server_obs_dim}")
        logging.info(f"Mean-field Dim: {self.mean_field_dim}")

    def train(self):
        for episode in range(self.config['num_episodes']):
            total_steps = 0
            obs = self.env.reset()
            episode_reward = 0
            episode_aoi = 0

            # 初始平均场
            self.smooth_mean_field = self.env.get_mean_field_state()

            for t in range(self.config['episode_length']):
                # 1. 更新平滑平均场 (EMA)
                true_mean_field = self.env.get_mean_field_state()
                self.smooth_mean_field = (1 - self.config['alpha']) * self.smooth_mean_field + self.config[
                    'alpha'] * true_mean_field

                # 2. 动作选择
                actions = []
                action_masks = self.env.get_agent_action_mask()

                with torch.no_grad():
                    mf_tensor = torch.tensor(self.smooth_mean_field, dtype=torch.float32).unsqueeze(0).to(self.device)

                    # 用户动作
                    for i in range(self.n_users):
                        obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                        mask_tensor = torch.tensor(action_masks[i], dtype=torch.bool).unsqueeze(0).to(self.device)
                        action, _ = self.user_actors[i].get_action(obs_tensor, mf_tensor, mask_tensor)
                        actions.append(action)

                    # 服务器动作
                    for i in range(self.n_servers):
                        agent_idx = self.n_users + i
                        obs_tensor = torch.tensor(obs[agent_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
                        mask_tensor = torch.tensor(action_masks[agent_idx], dtype=torch.bool).unsqueeze(0).to(
                            self.device)
                        action, _ = self.server_actors[i].get_action(obs_tensor, mf_tensor, mask_tensor)
                        actions.append(action)

                # 3. 与环境交互
                next_obs, reward, done, info = self.env.step(actions)
                episode_reward += reward
                episode_aoi += info.get('average_aoi', -1)
                total_steps += 1

                # 4. 存储到回放缓冲区
                next_mf = (1 - self.config['alpha']) * self.smooth_mean_field + self.config['alpha'] * true_mean_field
                self.buffer.push(obs, self.smooth_mean_field, actions, reward, next_obs, next_mf, done, action_masks)

                obs = next_obs

                # 5. 更新网络
                if len(self.buffer) > self.config['batch_size']:
                    self.update_models()

                if done:  # 环境内置结束条件
                    break

            # --- 回合结束日志 ---
            self.save_completed_tasks_log(episode)
            avg_aoi = episode_aoi / total_steps
            completed_tasks_this_episode = len(self.env.completed_tasks_log)
            logging.info(f"Episode: {episode + 1}/{self.config['num_episodes']}, "
                         f"Reward: {episode_reward:.2f}, Avg AoI: {avg_aoi:.2f}, Completed Tasks: {completed_tasks_this_episode}")

    def save_completed_tasks_log(self, epoch):
        log_path = self.config['log_dir']
        if not self.env.completed_tasks_log:
            logging.info("No tasks were completed during the training. Log file not created.")
            return

        os.makedirs(log_path, exist_ok=True)
        # 定义CSV文件名和表头
        filepath = os.path.join(log_path, "completed_tasks_log" + "_" + str(epoch) + ".csv")
        headers = ['task_id', 'user_id', 'server_id', 'generation_time', 'completion_time', 'latency']

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for task in self.env.completed_tasks_log:
                row = [
                    task['task_id'],
                    task['user_id'],
                    task['server_id'],
                    task['generation_time'],
                    task['completion_time'],
                    task['latency']
                ]
                writer.writerow(row)

    def update_models(self):
        """缓冲区采样并更新 Actor/Critic"""
        # 从buffer中采样，obs 和 mask 是列表
        obs_b_list, mf_b, act_b, r_b, next_obs_b_list, next_mf_b, done_b, _ = self.buffer.sample(
            self.config['batch_size'])

        mf_b = torch.tensor(mf_b, dtype=torch.float32).to(self.device)
        act_b = torch.tensor(act_b, dtype=torch.int64).to(self.device)
        r_b = torch.tensor(r_b, dtype=torch.float32).to(self.device)
        next_mf_b = torch.tensor(next_mf_b, dtype=torch.float32).to(self.device)
        done_b = torch.tensor(done_b, dtype=torch.float32).to(self.device)

        total_critic_loss = 0
        total_actor_loss = 0

        # 每个agent计算损失
        for i in range(self.n_agents):
            is_user = i < self.n_users
            actor = self.user_actors[i] if is_user else self.server_actors[i - self.n_users]
            critic = self.user_critics[i] if is_user else self.server_critics[i - self.n_users]

            agent_obs = torch.tensor([obs[i] for obs in obs_b_list], dtype=torch.float32).to(self.device)
            agent_next_obs = torch.tensor([next_obs[i] for next_obs in next_obs_b_list], dtype=torch.float32).to(
                self.device)
            agent_act = act_b[:, i].unsqueeze(1)

            # --- 1. Critic Loss 计算 ---
            q_values = critic(agent_obs, mf_b)
            q_value = q_values.gather(1, agent_act)

            with torch.no_grad():
                next_q_values = critic(agent_next_obs, next_mf_b)
                next_logits = actor(agent_next_obs, next_mf_b)
                next_action = next_logits.argmax(dim=1, keepdim=True)
                target_q_value = next_q_values.gather(1, next_action)
                target_q = r_b + self.config['gamma'] * (1 - done_b) * target_q_value

            critic_loss = F.mse_loss(q_value, target_q)
            total_critic_loss += critic_loss

            # --- 2. Actor Loss 计算 ---
            logits = actor(agent_obs, mf_b)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            q_values_detached = critic(agent_obs, mf_b).detach()

            # 计算状态价值
            v_s = (probs * q_values_detached).sum(dim=-1, keepdim=True)

            # log_prob
            action_log_prob = log_probs.gather(1, agent_act)

            # 计算优势函数 Advantage A(s, a) = Q(s, a) - V(s)
            advantage = q_values_detached.gather(1, agent_act) - v_s

            # Actor Loss
            actor_loss = -(action_log_prob * advantage).mean()

            # 熵正则项
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            actor_loss -= self.config['entropy_coeff'] * entropy

            total_actor_loss += actor_loss

        # --- 3. 优化网络 ---
        # 优化 Critic
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(sum([list(c.parameters()) for c in self.user_critics + self.server_critics], []),
                                       1.0)
        self.critic_optimizer.step()

        # 优化 Actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(sum([list(a.parameters()) for a in self.user_actors + self.server_actors], []),
                                       1.0)
        self.actor_optimizer.step()


if __name__ == '__main__':
    # --- 训练配置参数 ---
    config = {
        'n_users': 28, # 7,
        'n_servers': 12, #3,
        'batch_proc_time': {'base': 3, 'per_task': 2}, # 批处理时间 = base + per_task * batch_size
        'max_batch_size': 12, # 3,
        'num_episodes': 35, # 20,  # 训练的总回合数
        'episode_length': 1000,  # 每个回合的最大步数
        'buffer_capacity': 50000,  # 经验回放缓冲区的容量
        'batch_size': 128,  # 每次更新时采样的批次大小
        'actor_lr': 1e-4,  # Actor学习率
        'critic_lr': 1e-4,  # Critic学习率
        'gamma': 0.99,  # 折扣因子
        'alpha': 0.1,  # 平滑平均场更新率 (EMA)
        'entropy_coeff': 0.01,  # 熵正则化系数
        'log_dir': f'logs/mfac_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }

    setup_logger(config['log_dir'])

    env = EdgeBatchEnv(
        n_users=config['n_users'],
        n_servers=config['n_servers'],
        batch_proc_time=config['batch_proc_time'],
        max_batch_size=config['max_batch_size']
    )

    trainer = MFAC_Trainer(env, config)
    trainer.train()

    logging.info("Training finished.")

