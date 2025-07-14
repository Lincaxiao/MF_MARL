import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
from datetime import datetime
import csv

from EdgeBatchEnv import EdgeBatchEnv
from model_masac import Actor, Critic, ReplayBuffer, HierarchicalActor


# --- 日志配置 ---

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logger initialized.")


class MASAC_Trainer:
    """多智能体 SAC，采用 CTDE：
    - Actor 仅使用局部观测
    - Critic 使用全局状态
    """

    def __init__(self, env: EdgeBatchEnv, config: dict):
        self.env = env
        self.config = config
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        # 维度信息
        self.n_users = env.n_users
        self.n_servers = env.n_servers
        self.n_agents = env.n_agents

        # 局部观测维度
        self.user_obs_dim = len(env._get_user_obs(0))
        self.server_obs_dim = len(env._get_server_obs(0))

        # 全局状态维度
        self.global_state_dim = len(env.get_global_state())

        # 动作维度
        self.user_action_dim = env.user_action_dim
        self.server_action_dim = env.server_action_dim

        # --- 初始化网络 ---
        # 用户使用层次化 Actor
        self.user_actors = [
            HierarchicalActor(self.user_obs_dim, self.n_servers).to(self.device)
            for _ in range(self.n_users)
        ]
        self.server_actors = [
            Actor(self.server_obs_dim, self.server_action_dim).to(self.device)
            for _ in range(self.n_servers)
        ]

        # 每个智能体各有 2 个 Q 网络 及其 target
        self.user_qf1 = [Critic(self.global_state_dim, self.user_action_dim).to(self.device) for _ in range(self.n_users)]
        self.user_qf2 = [Critic(self.global_state_dim, self.user_action_dim).to(self.device) for _ in range(self.n_users)]
        self.user_qf1_target = [Critic(self.global_state_dim, self.user_action_dim).to(self.device) for _ in range(self.n_users)]
        self.user_qf2_target = [Critic(self.global_state_dim, self.user_action_dim).to(self.device) for _ in range(self.n_users)]
        # 拷贝参数
        for i in range(self.n_users):
            self.user_qf1_target[i].load_state_dict(self.user_qf1[i].state_dict())
            self.user_qf2_target[i].load_state_dict(self.user_qf2[i].state_dict())

        self.server_qf1 = [Critic(self.global_state_dim, self.server_action_dim).to(self.device) for _ in range(self.n_servers)]
        self.server_qf2 = [Critic(self.global_state_dim, self.server_action_dim).to(self.device) for _ in range(self.n_servers)]
        self.server_qf1_target = [Critic(self.global_state_dim, self.server_action_dim).to(self.device) for _ in range(self.n_servers)]
        self.server_qf2_target = [Critic(self.global_state_dim, self.server_action_dim).to(self.device) for _ in range(self.n_servers)]
        for i in range(self.n_servers):
            self.server_qf1_target[i].load_state_dict(self.server_qf1[i].state_dict())
            self.server_qf2_target[i].load_state_dict(self.server_qf2[i].state_dict())

        # --- 优化器 ---
        all_actor_params = sum([list(a.parameters()) for a in self.user_actors + self.server_actors], [])
        all_critic_params = sum(
            [
                list(c.parameters())
                for c in self.user_qf1
                + self.user_qf2
                + self.server_qf1
                + self.server_qf2
            ],
            [],
        )

        self.actor_optimizer = optim.Adam(all_actor_params, lr=config["actor_lr"])
        self.critic_optimizer = optim.Adam(all_critic_params, lr=config["critic_lr"])

        # --- 经验回放 ---
        self.buffer = ReplayBuffer(config["buffer_capacity"])

        # --- 其他 ---
        self.total_env_steps = 0

        logging.info("MASAC Trainer Initialized.")
        logging.info(f"Device: {self.device}")
        logging.info(
            f"obs_dim(user, server) = ({self.user_obs_dim}, {self.server_obs_dim}), global_state_dim = {self.global_state_dim}"
        )

    # -------------------- 训练主循环 --------------------
    def train(self):
        for episode in range(self.config["num_episodes"]):
            obs = self.env.reset()
            global_state = self.env.get_global_state()

            episode_reward = 0
            episode_aoi = 0
            total_steps = 0

            for t in range(self.config["episode_length"]):
                action_masks = self.env.get_agent_action_mask()
                actions = []
                with torch.no_grad():
                    # 采样动作
                    for i in range(self.n_users):
                        obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                        mask_tensor = torch.tensor(action_masks[i], dtype=torch.bool).unsqueeze(0).to(self.device)
                        action, _ = self.user_actors[i].get_action(obs_tensor, mask_tensor)
                        actions.append(action)
                    for i in range(self.n_servers):
                        agent_idx = self.n_users + i
                        obs_tensor = torch.tensor(obs[agent_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
                        mask_tensor = torch.tensor(action_masks[agent_idx], dtype=torch.bool).unsqueeze(0).to(self.device)
                        action, _ = self.server_actors[i].get_action(obs_tensor, mask_tensor)
                        actions.append(action)

                # 环境交互
                next_obs, reward, done, info = self.env.step(actions)
                next_global_state = self.env.get_global_state()
                episode_reward += reward
                episode_aoi += info.get("average_aoi", -1)
                total_steps += 1
                self.total_env_steps += 1

                # 存入回放缓冲区
                self.buffer.push(
                    obs,
                    global_state,
                    actions,
                    reward,
                    next_obs,
                    next_global_state,
                    done,
                    action_masks,
                )

                obs = next_obs
                global_state = next_global_state

                # 更新网络
                if len(self.buffer) > self.config["batch_size"]:
                    self.update_models()

                if done:
                    break

            # 记录日志
            avg_aoi = episode_aoi / total_steps
            logging.info(
                f"Episode {episode + 1}/{self.config['num_episodes']}, Reward: {episode_reward:.2f}, Avg AoI: {avg_aoi:.2f}"
            )

        logging.info("Training finished.")

    # -------------------- 更新网络 --------------------
    def update_models(self):
        (
            obs_b_list,
            gs_b,
            act_b,
            r_b,
            next_obs_b_list,
            next_gs_b,
            done_b,
            _,
        ) = self.buffer.sample(self.config["batch_size"])

        gs_b = torch.tensor(gs_b, dtype=torch.float32).to(self.device)
        next_gs_b = torch.tensor(next_gs_b, dtype=torch.float32).to(self.device)
        r_b = torch.tensor(r_b, dtype=torch.float32).to(self.device)
        done_b = torch.tensor(done_b, dtype=torch.float32).to(self.device)

        total_q_loss = 0
        total_actor_loss = 0

        # -------- 用户代理 --------
        for i in range(self.n_users):
            actor = self.user_actors[i]
            qf1, qf2 = self.user_qf1[i], self.user_qf2[i]
            qf1_t, qf2_t = self.user_qf1_target[i], self.user_qf2_target[i]
            action_dim = self.user_action_dim

            agent_obs = torch.tensor([obs[i] for obs in obs_b_list], dtype=torch.float32).to(self.device)
            agent_next_obs = torch.tensor([next_obs[i] for next_obs in next_obs_b_list], dtype=torch.float32).to(self.device)
            agent_act = torch.tensor(act_b[:, i]).long().unsqueeze(1).to(self.device)

            # ---- Critic 更新 ----
            with torch.no_grad():
                next_logits = actor(agent_next_obs)
                next_log_pi = F.log_softmax(next_logits, dim=-1)
                next_action_probs = F.softmax(next_logits, dim=-1)
                q1_next = qf1_t(next_gs_b)
                q2_next = qf2_t(next_gs_b)
                min_q_next = next_action_probs * (
                    torch.min(q1_next, q2_next) - self.config["alpha"] * next_log_pi
                )
                min_q_next = min_q_next.sum(dim=1, keepdim=True)
                target_q = r_b + self.config["gamma"] * (1 - done_b) * min_q_next

            q1_values = qf1(gs_b).gather(1, agent_act)
            q2_values = qf2(gs_b).gather(1, agent_act)
            qf1_loss = F.mse_loss(q1_values, target_q)
            qf2_loss = F.mse_loss(q2_values, target_q)
            total_q_loss += qf1_loss + qf2_loss

            # ---- Actor 更新 ----
            logits = actor(agent_obs)
            log_pi = F.log_softmax(logits, dim=-1)
            action_probs = F.softmax(logits, dim=-1)
            with torch.no_grad():
                q1_curr = qf1(gs_b)
                q2_curr = qf2(gs_b)
                min_q_curr = torch.min(q1_curr, q2_curr)
            actor_loss = (action_probs * (self.config["alpha"] * log_pi - min_q_curr)).mean()
            total_actor_loss += actor_loss
            # 软更新将在优化之后统一进行

        # -------- 服务器代理 --------
        for i in range(self.n_servers):
            idx = self.n_users + i
            actor = self.server_actors[i]
            qf1, qf2 = self.server_qf1[i], self.server_qf2[i]
            qf1_t, qf2_t = self.server_qf1_target[i], self.server_qf2_target[i]
            action_dim = self.server_action_dim

            agent_obs = torch.tensor([obs[idx] for obs in obs_b_list], dtype=torch.float32).to(self.device)
            agent_next_obs = torch.tensor([next_obs[idx] for next_obs in next_obs_b_list], dtype=torch.float32).to(
                self.device
            )
            agent_act = torch.tensor(act_b[:, idx]).long().unsqueeze(1).to(self.device)

            with torch.no_grad():
                next_logits = actor(agent_next_obs)
                next_log_pi = F.log_softmax(next_logits, dim=-1)
                next_action_probs = F.softmax(next_logits, dim=-1)
                q1_next = qf1_t(next_gs_b)
                q2_next = qf2_t(next_gs_b)
                min_q_next = next_action_probs * (
                    torch.min(q1_next, q2_next) - self.config["alpha"] * next_log_pi
                )
                min_q_next = min_q_next.sum(dim=1, keepdim=True)
                target_q = r_b + self.config["gamma"] * (1 - done_b) * min_q_next

            q1_values = qf1(gs_b).gather(1, agent_act)
            q2_values = qf2(gs_b).gather(1, agent_act)
            qf1_loss = F.mse_loss(q1_values, target_q)
            qf2_loss = F.mse_loss(q2_values, target_q)
            total_q_loss += qf1_loss + qf2_loss

            logits = actor(agent_obs)
            log_pi = F.log_softmax(logits, dim=-1)
            action_probs = F.softmax(logits, dim=-1)
            with torch.no_grad():
                q1_curr = qf1(gs_b)
                q2_curr = qf2(gs_b)
                min_q_curr = torch.min(q1_curr, q2_curr)
            actor_loss = (action_probs * (self.config["alpha"] * log_pi - min_q_curr)).mean()
            total_actor_loss += actor_loss
            # 软更新将在优化之后统一进行

        # ---- 优化参数 ----
        self.critic_optimizer.zero_grad()
        total_q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.critic_optimizer.param_groups[0]["params"] if p.requires_grad], 1.0
        )
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.actor_optimizer.param_groups[0]["params"] if p.requires_grad], 1.0
        )
        self.actor_optimizer.step()

        # ---- 软更新所有 target 网络 ----
        for i in range(self.n_users):
            self.soft_update(self.user_qf1[i], self.user_qf1_target[i])
            self.soft_update(self.user_qf2[i], self.user_qf2_target[i])

        for i in range(self.n_servers):
            self.soft_update(self.server_qf1[i], self.server_qf1_target[i])
            self.soft_update(self.server_qf2[i], self.server_qf2_target[i])

    # -------------------- 工具函数 --------------------
    def soft_update(self, net: torch.nn.Module, target_net: torch.nn.Module):
        tau = self.config["tau"]
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# -------------------- 主入口 --------------------
if __name__ == "__main__":
    config = {
        "n_users": 28,
        "n_servers": 12,
        "batch_proc_time": {"base": 3, "per_task": 2},
        "max_batch_size": 12,
        "num_episodes": 35,
        "episode_length": 1000,
        "buffer_capacity": 50000,
        "batch_size": 128,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "gamma": 0.99,
        "alpha": 0.2,  # 固定熵系数
        "tau": 0.005,  # target network update rate
        "log_dir": f"logs/masac_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    setup_logger(config["log_dir"])

    env = EdgeBatchEnv(
        n_users=config["n_users"],
        n_servers=config["n_servers"],
        batch_proc_time=config["batch_proc_time"],
        max_batch_size=config["max_batch_size"],
    )

    trainer = MASAC_Trainer(env, config)
    trainer.train() 