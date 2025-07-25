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

from RealEdgeBatchEnv import EdgeBatchEnv
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
        self.use_mean_field = config.get('use_mean_field', True)
        self.device = 'cpu'

        # 维度
        self.n_users = env.n_users
        self.n_servers = env.n_servers
        self.n_agents = env.n_agents
        
        # 重新计算观测维度
        # user观测：自身状态 + 所有服务器状态 + 平均场
        self.user_local_obs_dim = len(env._get_user_obs(0))  # 用户自身观测维度
        self.server_local_obs_dim = len(env._get_server_obs(0))  # 单个服务器观测维度
        # user的完整观测包括所有服务器的状态
        self.user_obs_dim = self.user_local_obs_dim + self.n_servers * self.server_local_obs_dim
        # server只观测自身状态
        self.server_obs_dim = self.server_local_obs_dim
        
        self.mf_dim = len(env.get_mean_field_state()) if self.use_mean_field else 0
        self.global_state_dim = len(env.get_global_state())

        # 共享的Actor - 所有user共享一个，所有server共享一个
        self.user_actor = HierarchicalActor(self.user_obs_dim, self.mf_dim, self.n_servers).to(self.device)
        self.server_actor = Actor(self.server_obs_dim, self.mf_dim, env.server_action_dim).to(self.device)
        # self.user_actor = torch.compile(HierarchicalActor(self.user_obs_dim, self.mf_dim, self.n_servers).to(self.device))
        # self.server_actor = torch.compile(Actor(self.server_obs_dim, self.mf_dim, env.server_action_dim).to(self.device))

        # Critic - 根据use_global选择架构
        if self.use_global:
            # 全局Critic：接收全局状态 + 平均场
            self.global_critic = ValueNet(self.global_state_dim + self.mf_dim).to(self.device)
            # self.global_critic = torch.compile(ValueNet(self.global_state_dim + self.mf_dim).to(self.device))
            self.user_critic = None
            self.server_critic = None
        else:
            # 局部Critic：user和server分别有自己的Critic
            self.user_critic = ValueNet(self.user_obs_dim + self.mf_dim).to(self.device)
            self.server_critic = ValueNet(self.server_obs_dim + self.mf_dim).to(self.device)
            # self.user_critic = torch.compile(ValueNet(self.user_obs_dim + self.mf_dim).to(self.device))
            # self.server_critic = torch.compile(ValueNet(self.server_obs_dim + self.mf_dim).to(self.device))
            self.global_critic = None

        # 优化器
        actor_params = list(self.user_actor.parameters()) + list(self.server_actor.parameters())
        self.actor_opt = optim.Adam(actor_params, lr=config['actor_lr'])
        
        if self.use_global:
            critic_params = list(self.global_critic.parameters())
        else:
            critic_params = list(self.user_critic.parameters()) + list(self.server_critic.parameters())
        self.critic_opt = optim.Adam(critic_params, lr=config['critic_lr'])
        
        self.all_actor_params = actor_params

        logging.info("MAPPO-MF Trainer Initialized with Shared Networks.")
        logging.info(f"Device: {self.device}")
        logging.info(f"User Obs Dim (including all servers): {self.user_obs_dim}, Server Obs Dim: {self.server_obs_dim}")
        if self.use_global:
            logging.info(f"Global State Dim: {self.global_state_dim}")
        logging.info(f"Mean-field Dim: {self.mf_dim}")
        logging.info(f"Use Global Critic: {self.use_global}")
        logging.info(f"Use Mean Field: {self.use_mean_field}")

    def _get_enhanced_obs(self):
        """获取增强的观测，user包含所有服务器状态，server只包含自身状态"""
        # 获取基础观测
        user_local_obs = [self.env._get_user_obs(i) for i in range(self.n_users)]
        server_obs = [self.env._get_server_obs(i) for i in range(self.n_servers)]
        
        # 构建user的增强观测（包含所有服务器状态）
        all_server_states = []
        for server_obs_single in server_obs:
            all_server_states.extend(server_obs_single)
        
        user_enhanced_obs = []
        for i in range(self.n_users):
            # user观测 = 自身状态 + 所有服务器状态
            enhanced = user_local_obs[i] + all_server_states
            user_enhanced_obs.append(enhanced)
        
        # server观测保持原样（只有自身状态）
        return user_enhanced_obs + server_obs

    # ------------ 主训练 ------------
    def _warmup_critic(self):
        """在特定模式下，预热 Critic 网络，以提供更稳定的价值基线。"""
        warmup_epochs = self.config.get('critic_warmup_epochs', 0)
            
        print(f"--- Starting Critic Warm-up for {warmup_epochs} epochs ---")
        
        for epoch in range(warmup_epochs):
            obs = self.env.reset()
            enhanced_obs = self._get_enhanced_obs()
            
            # 预热阶段也需要收集完整的轨迹
            obs_buf = [[] for _ in range(self.n_agents)]
            mf_buf = []
            gstate_buf = []  # 全局模式需要
            rew_buf, done_buf = [], []

            for t in range(self.config['episode_length']):
                if self.use_mean_field:
                    mf = self.env.get_mean_field_state()
                else:
                    mf = np.zeros(self.mf_dim, dtype=np.float32)
                mf_buf.append(mf)
                if self.use_global:
                    gstate_buf.append(self.env.get_global_state())
                
                masks = self.env.get_agent_action_mask()
                actions = []
                with torch.no_grad():
                    mf_t = torch.tensor(mf, dtype=torch.float32).unsqueeze(0)
                    for i in range(self.n_users):
                        obs_t = torch.tensor(enhanced_obs[i], dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.tensor(masks[i], dtype=torch.bool).unsqueeze(0)
                        a, _ = self.user_actor.get_action(obs_t, mf_t, mask_t)
                        actions.append(a)
                        obs_buf[i].append(enhanced_obs[i])
                    for i in range(self.n_servers):
                        idx = self.n_users + i
                        obs_t = torch.tensor(enhanced_obs[idx], dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.tensor(masks[idx], dtype=torch.bool).unsqueeze(0)
                        a, _ = self.server_actor.get_action(obs_t, mf_t, mask_t)
                        actions.append(a)
                        obs_buf[idx].append(enhanced_obs[idx])

                next_obs, reward, done, _ = self.env.step(actions)
                enhanced_obs = self._get_enhanced_obs()
                rew_buf.append(reward)
                done_buf.append(done)
                obs = next_obs
                if done:
                    break
            
            # --- GAE 计算 ---
            rewards = np.array(rew_buf, dtype=np.float32)
            dones = np.array(done_buf, dtype=np.bool_)
            mf_arr = torch.tensor(np.asarray(mf_buf, dtype=np.float32))
            
            if self.use_global:
                # 全局模式：使用全局状态
                gstate_arr = torch.tensor(np.asarray(gstate_buf, dtype=np.float32))
                critic_input = torch.cat([gstate_arr, mf_arr], dim=-1)
                with torch.no_grad():
                    values = self.global_critic(critic_input).squeeze(-1).cpu().numpy()
                _, ret_t = self._gae(rewards, dones, values)
                ret_t = torch.tensor(ret_t, dtype=torch.float32)
            else:
                # 局部模式：分别计算用户和服务器的价值函数
                user_values_list = []
                server_values_list = []
                
                for aid in range(self.n_agents):
                    obs_arr = torch.tensor(np.asarray(obs_buf[aid], dtype=np.float32))
                    critic_in = torch.cat([obs_arr, mf_arr], dim=-1)
                    with torch.no_grad():
                        if aid < self.n_users:
                            v = self.user_critic(critic_in).squeeze(-1).cpu().numpy()
                            user_values_list.append(v)
                        else:
                            v = self.server_critic(critic_in).squeeze(-1).cpu().numpy()
                            server_values_list.append(v)

                # 计算优势和回报
                user_ret_list = []
                server_ret_list = []
                for v in user_values_list:
                    _, ret = self._gae(rewards, dones, v)
                    user_ret_list.append(torch.tensor(ret, dtype=torch.float32))
                for v in server_values_list:
                    _, ret = self._gae(rewards, dones, v)
                    server_ret_list.append(torch.tensor(ret, dtype=torch.float32))

            # --- PPO 更新 (只更新 Critic) ---
            total_critic_loss_epoch = 0.0
            for _ in range(self.config['update_epochs']):
                idxs = np.arange(len(rewards))
                np.random.shuffle(idxs)
                for start in range(0, len(rewards), self.config['mini_batch_size']):
                    end = start + self.config['mini_batch_size']
                    mb_idx = idxs[start:end]
                    
                    self.critic_opt.zero_grad()
                    if self.use_global:
                        # 全局模式更新
                        v_pred = self.global_critic(critic_input[mb_idx]).squeeze(-1)
                        loss = F.mse_loss(v_pred, ret_t[mb_idx])
                        loss.backward()
                        total_critic_loss_epoch += loss.item()
                    else:
                        # 局部模式更新
                        batch_loss = 0.0
                        
                        # 更新用户critic
                        for uid in range(self.n_users):
                            obs_arr = torch.tensor(np.asarray(obs_buf[uid], dtype=np.float32))[mb_idx]
                            ci = torch.cat([obs_arr, mf_arr[mb_idx]], dim=-1)
                            v = self.user_critic(ci).squeeze(-1)
                            loss = F.mse_loss(v, user_ret_list[uid][mb_idx])
                            loss.backward()
                            batch_loss += loss.item()
                        
                        # 更新服务器critic
                        for sid in range(self.n_servers):
                            aid = self.n_users + sid
                            obs_arr = torch.tensor(np.asarray(obs_buf[aid], dtype=np.float32))[mb_idx]
                            ci = torch.cat([obs_arr, mf_arr[mb_idx]], dim=-1)
                            v = self.server_critic(ci).squeeze(-1)
                            loss = F.mse_loss(v, server_ret_list[sid][mb_idx])
                            loss.backward()
                            batch_loss += loss.item()
                        
                        total_critic_loss_epoch += batch_loss

                    torch.nn.utils.clip_grad_norm_(self.critic_opt.param_groups[0]['params'], self.config['max_grad_norm'])
                    self.critic_opt.step()
            
            print(f"Critic Warm-up Epoch [{epoch + 1}/{warmup_epochs}], Avg Critic Loss: {total_critic_loss_epoch / self.config['update_epochs']:.4f}")
        
        # 预热结束后，如果收集了足够的数据，训练GMM
        total_samples = sum(len(data) for data in self.env.history_buffer.values())
        if total_samples >= self.env.gmm_config['min_samples'] * 3:
            print("预热阶段收集了足够的数据，开始训练GMM...")
            self.env.train_gmm_models()
        
        print("--- Critic Warm-up Finished ---")


    def train(self):
        # 在正式训练前先收集 GMM 数据
        self._collect_initial_gmm_data()

        self._warmup_critic()

        for ep in range(self.config['num_episodes']):
            obs = self.env.reset()
            enhanced_obs = self._get_enhanced_obs()
            ep_rew, ep_aoi = 0, 0

            # 轨迹缓存
            obs_buf = [[] for _ in range(self.n_agents)]
            mf_buf = []
            act_buf = [[] for _ in range(self.n_agents)]
            logp_buf = [[] for _ in range(self.n_agents)]
            gstate_buf = []  # 全局模式需要
            rew_buf, done_buf = [], []

            for t in range(self.config['episode_length']):
                # 根据是否使用 Mean-Field 决定均场输入
                if self.use_mean_field:
                    mf = self.env.get_mean_field_state()
                else:
                    mf = np.zeros(self.mf_dim, dtype=np.float32)
                mf_buf.append(mf)
                if self.use_global:
                    gstate_buf.append(self.env.get_global_state())

                masks = self.env.get_agent_action_mask()
                actions = []

                with torch.no_grad():
                    mf_t = torch.tensor(mf, dtype=torch.float32).unsqueeze(0)
                    # 用户
                    for i in range(self.n_users):
                        obs_t = torch.tensor(enhanced_obs[i], dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.tensor(masks[i], dtype=torch.bool).unsqueeze(0)
                        a, lp = self.user_actor.get_action(obs_t, mf_t, mask_t)
                        actions.append(a)
                        obs_buf[i].append(enhanced_obs[i])
                        act_buf[i].append(a)
                        logp_buf[i].append(lp.item())
                    # 服务器
                    for i in range(self.n_servers):
                        idx = self.n_users + i
                        obs_t = torch.tensor(enhanced_obs[idx], dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.tensor(masks[idx], dtype=torch.bool).unsqueeze(0)
                        a, lp = self.server_actor.get_action(obs_t, mf_t, mask_t)
                        actions.append(a)
                        obs_buf[idx].append(enhanced_obs[idx])
                        act_buf[idx].append(a)
                        logp_buf[idx].append(lp.item())

                next_obs, reward, done, info = self.env.step(actions)
                enhanced_obs = self._get_enhanced_obs()
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
                # 全局模式：使用全局状态
                gstate_arr = torch.tensor(np.asarray(gstate_buf, dtype=np.float32))
                critic_input = torch.cat([gstate_arr, mf_arr], dim=-1)
                with torch.no_grad():
                    values = self.global_critic(critic_input).squeeze(-1).cpu().numpy()

                adv, ret = self._gae(rewards, dones, values)
                adv_t = torch.tensor(adv, dtype=torch.float32)
                ret_t = torch.tensor(ret, dtype=torch.float32)
            else:
                # 局部模式：分别计算用户和服务器的价值函数和优势
                user_values_list = []
                server_values_list = []
                user_adv_list = []
                user_ret_list = []
                server_adv_list = []
                server_ret_list = []
                
                for aid in range(self.n_agents):
                    obs_arr = torch.tensor(np.asarray(obs_buf[aid], dtype=np.float32))
                    critic_in = torch.cat([obs_arr, mf_arr], dim=-1)
                    with torch.no_grad():
                        if aid < self.n_users:
                            v = self.user_critic(critic_in).squeeze(-1).cpu().numpy()
                            user_values_list.append(v)
                            adv, ret = self._gae(rewards, dones, v)
                            user_adv_list.append(torch.tensor(adv, dtype=torch.float32))
                            user_ret_list.append(torch.tensor(ret, dtype=torch.float32))
                        else:
                            v = self.server_critic(critic_in).squeeze(-1).cpu().numpy()
                            server_values_list.append(v)
                            adv, ret = self._gae(rewards, dones, v)
                            server_adv_list.append(torch.tensor(adv, dtype=torch.float32))
                            server_ret_list.append(torch.tensor(ret, dtype=torch.float32))

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
                        # 全局模式：使用全局critic
                        v_pred = self.global_critic(critic_input[mb_idx]).squeeze(-1)
                        critic_loss = F.mse_loss(v_pred, ret_t[mb_idx])
                        critic_loss.backward()
                    else:
                        # 局部模式：分别更新user和server critic
                        critic_loss = 0.0
                        
                        # 更新用户critic
                        for uid in range(self.n_users):
                            obs_arr = torch.tensor(np.asarray(obs_buf[uid], dtype=np.float32))[mb_idx]
                            ci = torch.cat([obs_arr, mf_arr[mb_idx]], dim=-1)
                            v = self.user_critic(ci).squeeze(-1)
                            loss = F.mse_loss(v, user_ret_list[uid][mb_idx])
                            loss.backward()
                            critic_loss += loss.item()
                        
                        # 更新服务器critic
                        for sid in range(self.n_servers):
                            aid = self.n_users + sid
                            obs_arr = torch.tensor(np.asarray(obs_buf[aid], dtype=np.float32))[mb_idx]
                            ci = torch.cat([obs_arr, mf_arr[mb_idx]], dim=-1)
                            v = self.server_critic(ci).squeeze(-1)
                            loss = F.mse_loss(v, server_ret_list[sid][mb_idx])
                            loss.backward()
                            critic_loss += loss.item()
                    
                    torch.nn.utils.clip_grad_norm_(self.critic_opt.param_groups[0]['params'], self.config['max_grad_norm'])
                    self.critic_opt.step()

                    # Actor 更新
                    self.actor_opt.zero_grad()
                    total_actor_loss = 0.0
                    
                    # 更新用户actor
                    for uid in range(self.n_users):
                        mb_obs = torch.tensor([obs_buf[uid][i] for i in mb_idx], dtype=torch.float32)
                        mb_act = torch.tensor([act_buf[uid][i] for i in mb_idx], dtype=torch.int64)
                        old_lp = torch.tensor([logp_buf[uid][i] for i in mb_idx], dtype=torch.float32)
                        mb_mf = mf_arr[mb_idx]
                        
                        # 重新计算 log_prob 与熵：需要均场输入
                        logits = self.user_actor(mb_obs, mb_mf)
                        log_probs_all = F.log_softmax(logits, dim=-1)
                        probs_all = F.softmax(logits, dim=-1)
                        new_lp = log_probs_all.gather(1, mb_act.unsqueeze(1)).squeeze(1)
                        ent = -(probs_all * log_probs_all).sum(dim=-1)

                        if self.use_global:
                            adv = adv_t[mb_idx]
                        else:
                            adv = user_adv_list[uid][mb_idx]
                            # 局部模式下，对每个 agent 的优势再做标准化，提升稳定性
                            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                        
                        ratio = torch.exp(new_lp - old_lp)
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * adv
                        loss = -torch.min(surr1, surr2).mean() - self.config['entropy_coeff'] * ent.mean()
                        loss.backward()
                        total_actor_loss += loss.item()
                    
                    # 更新服务器actor
                    for sid in range(self.n_servers):
                        aid = self.n_users + sid
                        mb_obs = torch.tensor([obs_buf[aid][i] for i in mb_idx], dtype=torch.float32)
                        mb_act = torch.tensor([act_buf[aid][i] for i in mb_idx], dtype=torch.int64)
                        old_lp = torch.tensor([logp_buf[aid][i] for i in mb_idx], dtype=torch.float32)
                        mb_mf = mf_arr[mb_idx]
                        
                        # 重新计算 log_prob 与熵：需要均场输入
                        logits = self.server_actor(mb_obs, mb_mf)
                        log_probs_all = F.log_softmax(logits, dim=-1)
                        probs_all = F.softmax(logits, dim=-1)
                        new_lp = log_probs_all.gather(1, mb_act.unsqueeze(1)).squeeze(1)
                        ent = -(probs_all * log_probs_all).sum(dim=-1)

                        if self.use_global:
                            adv = adv_t[mb_idx]
                        else:
                            adv = server_adv_list[sid][mb_idx]
                            # 局部模式下，对每个 agent 的优势再做标准化，提升稳定性
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
            
            # GMM训练逻辑：在每个回合结束时调用
            self.env.episode_end_callback()
            
            logging.info(f"Episode: {ep + 1}/{self.config['num_episodes']}, "
                         f"Reward: {ep_rew:.2f}, Avg AoI: {avg_aoi:.2f}, Completed Tasks: {completed_tasks_this_episode}")
            
            # 如果是前几个回合且GMM还未训练，提前训练GMM
            if (ep < 10 and not self.env.gmm_trained and 
                sum(len(data) for data in self.env.history_buffer.values()) >= self.env.gmm_config['min_samples'] * 3):
                # logging.info(f"Episode {ep + 1}: 提前训练GMM模型...")
                self.env.train_gmm_models()

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

    # ---------- 初始 GMM 数据收集 ----------
    def _collect_initial_gmm_data(self):
        """使用随机策略收集足够的数据以训练首次 GMM"""
        if not (self.use_mean_field and self.env.gmm_config['use_gmm']):
            return

        required = self.env.gmm_config['min_samples']
        # 只需保证其中一个变量样本达到 min_samples 即可，因为每个 step 都会收集所有变量
        while len(self.env.history_buffer['user_aoi']) < required:
            obs = self.env.reset()
            for _ in range(self.config['episode_length']):
                masks = self.env.get_agent_action_mask()
                actions = []
                # 随机选择一个合法动作
                for aid, mask in enumerate(masks):
                    valid = [idx for idx, m in enumerate(mask) if m]
                    actions.append(np.random.choice(valid))
                obs, _, done, _ = self.env.step(actions)
                if done:
                    break
        # 收集完毕后立即训练一次 GMM
        print("[Init] 收集完毕，开始首次训练 GMM...")
        self.env.train_gmm_models()

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
        'n_users': 35,
        'n_servers': 5,
        'batch_proc_time': {1: 5, 2: 9, 4: 18, 8: 35},
        'max_batch_size': 8,
        'num_episodes': 500,
        'episode_length': 1000,
        'actor_lr': 4e-5,
        'critic_lr': 4e-5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'update_epochs': 4,
        'mini_batch_size': 128,
        'entropy_coeff': 0.01,
        'max_grad_norm': 1.0,
        'use_global': False,  # 是否使用全局状态作为 Critic 输入
        # 'use_global': True, # 两行便于切换
        'use_mean_field': False,  # 是否将 Mean-Field 作为额外输入
        # 'use_mean_field': True, # 两行便于切换
        'critic_warmup_epochs': 0, # 在正式训练前预热 Critic 的回合数
        # GMM配置参数
        'gmm_config': {
            'n_components': 2,    # GMM组件数量
            'min_samples': 100,     # 最小样本数量（增加以提高GMM质量）
            'use_gmm': True,      # 是否使用GMM
            'regularization': 1e-6,  # 正则化参数
        },
        'log_dir': '',  # 占位，根据参数自动生成
    }

    # 根据配置动态生成日志目录，格式: logs/mappo_mf_shared_⟨Nuser⟩_⟨Nserver⟩_⟨Global|Local⟩_⟨MF|NoMF⟩_⟨Lr⟩
    mf_tag = 'MF' if config.get('use_mean_field', True) else 'NoMF'
    global_tag = 'Global' if config.get('use_global', True) else 'Local'
    gmm_tag = f"GMM{config['gmm_config']['n_components']}" if config['gmm_config']['use_gmm'] else 'NoGMM'
    suffix = f"{config['n_users']}_{config['n_servers']}_{global_tag}_{mf_tag}_{gmm_tag}_{config['actor_lr']}"
    config['log_dir'] = f"logs/Real_mappo_mf_{suffix}"

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
        gmm_config=config['gmm_config'],
        max_batch_size=config['max_batch_size']
    )
    trainer = MAPPO_MF_Trainer(env, config)
    trainer.train()
    
    logging.info("Training finished.")
