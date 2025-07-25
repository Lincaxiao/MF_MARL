import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
from datetime import datetime
import csv

from RealEdgeBatchEnv import EdgeBatchEnv
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

        # 全局状态维度
        self.global_state_dim = len(env.get_global_state())

        # 动作维度
        self.user_action_dim = env.user_action_dim
        self.server_action_dim = env.server_action_dim

        # 分离的观测维度
        self.user_local_obs_dim = len(env._get_user_obs(0))
        self.server_obs_dim = len(env._get_server_obs(0))

        # --- 初始化共享网络 ---
        # 1) Actor（共享）
        self.user_actor = HierarchicalActor(
            user_obs_dim=self.user_local_obs_dim,
            server_obs_dim=self.server_obs_dim,
            n_servers=self.n_servers
        ).to(self.device)
        self.server_actor = Actor(self.server_obs_dim, self.server_action_dim).to(self.device)

        # 2) Critic（共有两套 Q 网络）
        self.user_qf1 = Critic(self.global_state_dim, self.user_action_dim).to(self.device)
        self.user_qf2 = Critic(self.global_state_dim, self.user_action_dim).to(self.device)
        self.user_qf1_target = Critic(self.global_state_dim, self.user_action_dim).to(self.device)
        self.user_qf2_target = Critic(self.global_state_dim, self.user_action_dim).to(self.device)
        self.user_qf1_target.load_state_dict(self.user_qf1.state_dict())
        self.user_qf2_target.load_state_dict(self.user_qf2.state_dict())

        self.server_qf1 = Critic(self.global_state_dim, self.server_action_dim).to(self.device)
        self.server_qf2 = Critic(self.global_state_dim, self.server_action_dim).to(self.device)
        self.server_qf1_target = Critic(self.global_state_dim, self.server_action_dim).to(self.device)
        self.server_qf2_target = Critic(self.global_state_dim, self.server_action_dim).to(self.device)
        self.server_qf1_target.load_state_dict(self.server_qf1.state_dict())
        self.server_qf2_target.load_state_dict(self.server_qf2.state_dict())
        
        # --- 使用 torch.compile() 加速 ---
        # 仅编译在线网络，目标网络无需编译
        # self.user_actor = torch.compile(self.user_actor)
        # self.server_actor = torch.compile(self.server_actor)
        # self.user_qf1 = torch.compile(self.user_qf1)
        # self.user_qf2 = torch.compile(self.user_qf2)
        # self.server_qf1 = torch.compile(self.server_qf1)
        # self.server_qf2 = torch.compile(self.server_qf2)

        # 3) Target Actor（用于计算下一状态策略熵和概率）
        import copy
        self.user_actor_target = copy.deepcopy(self.user_actor)
        self.server_actor_target = copy.deepcopy(self.server_actor)

        # --- 自适应熵系数 ---
        self.log_alpha = torch.tensor(np.log(config.get("alpha", 0.2)), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.get("alpha_lr", 1e-4))
        # 目标熵：通常设为 -|A|
        self.target_entropy_user = -float(self.user_action_dim)
        self.target_entropy_server = -float(self.server_action_dim)

        # --- 优化器 ---
        all_actor_params = list(self.user_actor.parameters()) + list(self.server_actor.parameters())
        all_critic_params = list(self.user_qf1.parameters()) + list(self.user_qf2.parameters()) + \
                           list(self.server_qf1.parameters()) + list(self.server_qf2.parameters())

        self.actor_optimizer = optim.Adam(all_actor_params, lr=config["actor_lr"])
        self.critic_optimizer = optim.Adam(all_critic_params, lr=config["critic_lr"])

        # --- 经验回放 ---
        self.buffer = ReplayBuffer(config["buffer_capacity"])

        # --- 其他 ---
        self.total_env_steps = 0

        logging.info("MASAC Trainer Initialized.")
        logging.info(f"Device: {self.device}")
        logging.info(
            f"obs_dim(user_local, server) = ({self.user_local_obs_dim}, {self.server_obs_dim}), global_state_dim = {self.global_state_dim}"
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
                    # 分离观测
                    user_obs_list = obs[:self.n_users]
                    servers_obs_list = obs[self.n_users:]
                    servers_obs_tensor = torch.tensor(servers_obs_list, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, n_s, s_dim)

                    # 采样用户动作
                    for i in range(self.n_users):
                        user_obs_tensor = torch.tensor(user_obs_list[i], dtype=torch.float32).unsqueeze(0).to(self.device) # (1, u_dim)
                        mask_tensor = torch.tensor(action_masks[i], dtype=torch.bool).unsqueeze(0).to(self.device)
                        action, _ = self.user_actor.get_action(user_obs_tensor, servers_obs_tensor, mask_tensor)
                        actions.append(action)
                    
                    # 采样服务器动作
                    for i in range(self.n_servers):
                        server_obs_tensor = torch.tensor(servers_obs_list[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                        mask_tensor = torch.tensor(action_masks[self.n_users + i], dtype=torch.bool).unsqueeze(0).to(self.device)
                        action, _ = self.server_actor.get_action(server_obs_tensor, mask_tensor)
                        actions.append(action)

                # 环境交互
                next_obs, reward, done, info = self.env.step(actions)
                next_action_masks = self.env.get_agent_action_mask()
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
                    next_action_masks,
                )

                obs = next_obs
                global_state = next_global_state

                # 更新网络
                if self.total_env_steps % self.config.get("train_frequency", 1) == 0 and len(self.buffer) > self.config["batch_size"]:
                    self.update_models()

                if done:
                    break

            # 记录日志
            # 保存完成任务日志，与其他算法保持一致
            self.save_completed_tasks_log(episode)

            avg_aoi = episode_aoi / total_steps
            completed_tasks_this_episode = len(self.env.completed_tasks_log)
            logging.info(
                f"Episode: {episode + 1}/{self.config['num_episodes']}, Reward: {episode_reward:.2f}, Avg AoI: {avg_aoi:.2f}, Completed Tasks: {completed_tasks_this_episode}"
            )

        logging.info("Training finished.")

    # -------------------- 更新网络 --------------------
    def update_models(self):
        """批量化向量化更新：一次性处理所有智能体"""
        (
            obs_b_list,
            gs_b,
            act_b,
            r_b,
            next_obs_b_list,
            next_gs_b,
            done_b,
            mask_b,
            next_mask_b,
        ) = self.buffer.sample(self.config["batch_size"])

        batch_size = len(obs_b_list)
        
        # 转换为张量
        gs_b = torch.from_numpy(np.array(gs_b)).float().to(self.device)
        next_gs_b = torch.from_numpy(np.array(next_gs_b)).float().to(self.device)
        r_b = torch.from_numpy(np.array(r_b)).float().to(self.device)
        done_b = torch.from_numpy(np.array(done_b)).float().to(self.device)
        act_b = torch.from_numpy(np.array(act_b)).long().to(self.device)

        # ========== 批量化数据准备 (NumPy-based) ==========
        # 1. 用户
        user_local_obs_arr = np.array([np.array(obs[:self.n_users]) for obs in obs_b_list], dtype=np.float32)
        user_next_local_obs_arr = np.array([np.array(obs[:self.n_users]) for obs in next_obs_b_list], dtype=np.float32)
        user_mask_arr = np.array([mask[:self.n_users] for mask in mask_b], dtype=np.bool_)
        user_next_mask_arr = np.array([mask[:self.n_users] for mask in next_mask_b], dtype=np.bool_)

        user_local_obs_tensor = torch.from_numpy(user_local_obs_arr).to(self.device).reshape(batch_size * self.n_users, -1)
        user_next_local_obs_tensor = torch.from_numpy(user_next_local_obs_arr).to(self.device).reshape(batch_size * self.n_users, -1)
        user_mask_tensor = torch.from_numpy(user_mask_arr).to(self.device).reshape(batch_size * self.n_users, -1)
        user_next_mask_tensor = torch.from_numpy(user_next_mask_arr).to(self.device).reshape(batch_size * self.n_users, -1)

        # 2. 服务器
        server_obs_arr = np.array([np.array(obs[self.n_users:]) for obs in obs_b_list], dtype=np.float32)
        server_next_obs_arr = np.array([obs[self.n_users:] for obs in next_obs_b_list], dtype=np.float32)
        server_mask_arr = np.array([mask[self.n_users:] for mask in mask_b], dtype=np.bool_)
        server_next_mask_arr = np.array([mask[self.n_users:] for mask in next_mask_b], dtype=np.bool_)

        # 为每个用户准备一份服务器观测
        # server_obs_arr shape: (batch_size, n_servers, server_obs_dim)
        servers_obs_tensor = torch.from_numpy(server_obs_arr).to(self.device)
        servers_obs_tensor = servers_obs_tensor.unsqueeze(1).expand(-1, self.n_users, -1, -1)
        servers_obs_tensor = servers_obs_tensor.reshape(batch_size * self.n_users, self.n_servers, -1)

        servers_next_obs_tensor = torch.from_numpy(server_next_obs_arr).to(self.device)
        servers_next_obs_tensor = servers_next_obs_tensor.unsqueeze(1).expand(-1, self.n_users, -1, -1)
        servers_next_obs_tensor = servers_next_obs_tensor.reshape(batch_size * self.n_users, self.n_servers, -1)

        # 服务器自身更新用的张量
        server_obs_tensor = torch.from_numpy(server_obs_arr).to(self.device).reshape(batch_size * self.n_servers, -1)
        server_next_obs_tensor = torch.from_numpy(server_next_obs_arr).to(self.device).reshape(batch_size * self.n_servers, -1)
        server_mask_tensor = torch.from_numpy(server_mask_arr).to(self.device).reshape(batch_size * self.n_servers, -1)
        server_next_mask_tensor = torch.from_numpy(server_next_mask_arr).to(self.device).reshape(batch_size * self.n_servers, -1)

        # 3. 动作张量
        user_act_tensor = act_b[:, :self.n_users].reshape(-1, 1)  # (batch*n_users, 1)
        server_act_tensor = act_b[:, self.n_users:].reshape(-1, 1)  # (batch*n_servers, 1)

        # 4. 扩展全局状态以匹配 Critic 输入
        user_gs_expanded = gs_b.unsqueeze(1).expand(-1, self.n_users, -1).reshape(batch_size * self.n_users, -1)
        user_next_gs_expanded = next_gs_b.unsqueeze(1).expand(-1, self.n_users, -1).reshape(batch_size * self.n_users, -1)
        
        server_gs_expanded = gs_b.unsqueeze(1).expand(-1, self.n_servers, -1).reshape(batch_size * self.n_servers, -1)
        server_next_gs_expanded = next_gs_b.unsqueeze(1).expand(-1, self.n_servers, -1).reshape(batch_size * self.n_servers, -1)

        # 5. 扩展奖励和done信号
        user_r_expanded = r_b.unsqueeze(1).expand(-1, self.n_users, -1).reshape(batch_size * self.n_users, -1)
        user_done_expanded = done_b.unsqueeze(1).expand(-1, self.n_users, -1).reshape(batch_size * self.n_users, -1)
        
        server_r_expanded = r_b.unsqueeze(1).expand(-1, self.n_servers, -1).reshape(batch_size * self.n_servers, -1)
        server_done_expanded = done_b.unsqueeze(1).expand(-1, self.n_servers, -1).reshape(batch_size * self.n_servers, -1)

        # ========== 用户智能体批量更新 ==========
        # Critic 更新
        with torch.no_grad():
            user_next_logits = self.user_actor_target(user_next_local_obs_tensor, servers_next_obs_tensor).masked_fill(~user_next_mask_tensor, -1e10)
            user_next_log_pi = F.log_softmax(user_next_logits, dim=-1)
            user_next_action_probs = F.softmax(user_next_logits, dim=-1)
            
            user_q1_next = self.user_qf1_target(user_next_gs_expanded)
            user_q2_next = self.user_qf2_target(user_next_gs_expanded)
            user_min_q_next = user_next_action_probs * (
                torch.min(user_q1_next, user_q2_next) - self.log_alpha.exp() * user_next_log_pi
            )
            user_min_q_next = user_min_q_next.sum(dim=1, keepdim=True)
            user_target_q = user_r_expanded + self.config["gamma"] * (1 - user_done_expanded) * user_min_q_next

        user_q1_values = self.user_qf1(user_gs_expanded).gather(1, user_act_tensor)
        user_q2_values = self.user_qf2(user_gs_expanded).gather(1, user_act_tensor)
        user_qf1_loss = F.mse_loss(user_q1_values, user_target_q)
        user_qf2_loss = F.mse_loss(user_q2_values, user_target_q)
        
        # Actor 更新
        user_logits = self.user_actor(user_local_obs_tensor, servers_obs_tensor).masked_fill(~user_mask_tensor, -1e10)
        user_log_pi = F.log_softmax(user_logits, dim=-1)
        user_action_probs = F.softmax(user_logits, dim=-1)
        
        with torch.no_grad():
            user_q1_curr = self.user_qf1(user_gs_expanded).masked_fill(~user_mask_tensor, -1e10)
            user_q2_curr = self.user_qf2(user_gs_expanded).masked_fill(~user_mask_tensor, -1e10)
            user_min_q_curr = torch.min(user_q1_curr, user_q2_curr)
        
        user_actor_loss = (user_action_probs * (self.log_alpha.exp() * user_log_pi - user_min_q_curr)).sum(dim=-1).mean()
        user_entropy = -(user_action_probs * user_log_pi).sum(dim=-1)

        # ========== 服务器智能体批量更新 ==========
        with torch.no_grad():
            server_next_logits = self.server_actor_target(server_next_obs_tensor).masked_fill(~server_next_mask_tensor, -1e10)
            server_next_log_pi = F.log_softmax(server_next_logits, dim=-1)
            server_next_action_probs = F.softmax(server_next_logits, dim=-1)
            
            server_q1_next = self.server_qf1_target(server_next_gs_expanded)
            server_q2_next = self.server_qf2_target(server_next_gs_expanded)
            server_min_q_next = server_next_action_probs * (
                torch.min(server_q1_next, server_q2_next) - self.log_alpha.exp() * server_next_log_pi
            )
            server_min_q_next = server_min_q_next.sum(dim=1, keepdim=True)
            server_target_q = server_r_expanded + self.config["gamma"] * (1 - server_done_expanded) * server_min_q_next

        server_q1_values = self.server_qf1(server_gs_expanded).gather(1, server_act_tensor)
        server_q2_values = self.server_qf2(server_gs_expanded).gather(1, server_act_tensor)
        server_qf1_loss = F.mse_loss(server_q1_values, server_target_q)
        server_qf2_loss = F.mse_loss(server_q2_values, server_target_q)
        
        server_logits = self.server_actor(server_obs_tensor).masked_fill(~server_mask_tensor, -1e10)
        server_log_pi = F.log_softmax(server_logits, dim=-1)
        server_action_probs = F.softmax(server_logits, dim=-1)
        
        with torch.no_grad():
            server_q1_curr = self.server_qf1(server_gs_expanded).masked_fill(~server_mask_tensor, -1e10)
            server_q2_curr = self.server_qf2(server_gs_expanded).masked_fill(~server_mask_tensor, -1e10)
            server_min_q_curr = torch.min(server_q1_curr, server_q2_curr)
        
        server_actor_loss = (server_action_probs * (self.log_alpha.exp() * server_log_pi - server_min_q_curr)).sum(dim=-1).mean()
        server_entropy = -(server_action_probs * server_log_pi).sum(dim=-1)

        # ========== 合并损失 ==========
        total_q_loss = (user_qf1_loss + user_qf2_loss + server_qf1_loss + server_qf2_loss) / 4
        total_actor_loss = (user_actor_loss + server_actor_loss) / 2
        
        # ========== 优化器更新 ==========
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

        # ========== 自适应熵 α 更新 ==========
        alpha_loss_user = -(self.log_alpha * (user_entropy + self.target_entropy_user).detach()).mean()
        alpha_loss_server = -(self.log_alpha * (server_entropy + self.target_entropy_server).detach()).mean()
        alpha_loss = (alpha_loss_user + alpha_loss_server) / 2.0

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ========== 软更新所有 target 网络 ==========
        self.soft_update(self.user_qf1, self.user_qf1_target)
        self.soft_update(self.user_qf2, self.user_qf2_target)
        self.soft_update(self.server_qf1, self.server_qf1_target)
        self.soft_update(self.server_qf2, self.server_qf2_target)
        
        self.soft_update(self.user_actor, self.user_actor_target)
        self.soft_update(self.server_actor, self.server_actor_target)

    # -------------------- 工具函数 --------------------
    def soft_update(self, net: torch.nn.Module, target_net: torch.nn.Module):
        tau = self.config["tau"]
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # -------------------- 保存已完成任务日志 --------------------
    def save_completed_tasks_log(self, epoch: int):
        """将 self.env.completed_tasks_log 写入 CSV 文件，路径: log_dir/completed_tasks"""
        if not self.env.completed_tasks_log:
            logging.info("No tasks completed in this episode; skip writing completed_tasks log.")
            return

        base_path = os.path.join(self.config["log_dir"], "completed_tasks")
        os.makedirs(base_path, exist_ok=True)
        filepath = os.path.join(base_path, f"completed_tasks_log_{epoch}.csv")

        headers = ["task_id", "user_id", "server_id", "generation_time", "completion_time", "latency"]
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for task in self.env.completed_tasks_log:
                writer.writerow([
                    task["task_id"],
                    task["user_id"],
                    task["server_id"],
                    task["generation_time"],
                    task["completion_time"],
                    task["latency"],
                ])
        # logging.info(f"Saved completed_tasks log to {filepath}")

# -------------------- 主入口 --------------------
if __name__ == "__main__":
    config = {
        'n_users': 35,
        'n_servers': 5,
        'batch_proc_time': {1: 5, 2: 9, 4: 18, 8: 35},
        'max_batch_size': 8,
        'num_episodes': 500,
        'episode_length': 1000,
        'buffer_capacity': 5000,
        'batch_size': 128,
        'train_frequency': 10,  # 每 10 个环境步骤更新一次网络
        'actor_lr': 5e-7,
        'critic_lr': 5e-7,
        'alpha_lr': 1e-6,  # 自适应熵的学习率
        'gamma': 0.99,
        'alpha': 0.2,      # 自适应熵的初始值
        'tau': 0.005,      # 目标网络软更新系数
        'log_dir': '',     # 将根据下面的配置自动生成
    }

    # 根据配置动态生成日志目录
    # 格式: logs/Real_masac_{N_users}_{N_servers}_Global_NoMF_{Lr}
    suffix = f"{config['n_users']}_{config['n_servers']}_Global_NoMF_{config['actor_lr']}"
    config['log_dir'] = f"logs/Real_masac_{suffix}"

    setup_logger(config["log_dir"])
    
    # 记录所有配置参数
    logging.info("--- Training Configuration ---")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info("--------------------------")

    env = EdgeBatchEnv(
        n_users=config["n_users"],
        n_servers=config["n_servers"],
        batch_proc_time=config["batch_proc_time"],
        max_batch_size=config["max_batch_size"],
    )

    trainer = MASAC_Trainer(env, config)
    trainer.train() 