import numpy as np
import collections
from dataclasses import dataclass


@dataclass
class Task:
    id: int
    user_id: int
    generation_time: int
    assigned_server_id: int


class EdgeBatchEnv:
    def __init__(self, n_users: int, n_servers: int, batch_proc_time: dict, mf_bins: dict = None,
                 max_batch_size: int = 2):
        print("Initializing Edge Batch Processing Environment...")
        self.n_users = n_users
        self.n_servers = n_servers
        self.batch_proc_time_config = batch_proc_time
        self.n_agents = self.n_users + self.n_servers
        self.max_batch_size = max_batch_size

        # 定义平均场直方图的分箱
        if mf_bins is None:
            self.mf_bins = {
                'aoi': np.array([0, 5, 10, 15, 20, 25]),  # 用户瞬时AoI分布
                'h': np.array([-1, 0, 2, 4, 6]),  # 用户任务计时h分布,任务处理完后h就是新的AoI
                'q_len': np.array([0, 1, 3, 5, 7]),  # 服务器等待队列长度分布
                'b_size': np.array([0, 1, 2, 3, 4]),  # 服务器批处理大小分布
                't': np.array([-1, 0, 2, 4, 6]),  # 服务器剩余处理时间分布
            }
        else:
            self.mf_bins = mf_bins

        self.user_action_dim = self.n_servers + 1
        self.server_action_dim = 2

        self.reset()
        print(f"Environment created with {self.n_users} users and {self.n_servers} servers.")

    def reset(self):
        self.time_step = 0
        self.task_id_counter = 0
        self.users_state = [{'aoi': 0.0, 'h': -1} for _ in range(self.n_users)]
        self.servers_state = [{'q': collections.deque(), 'b': [], 't': -1} for _ in range(self.n_servers)]

        self.completed_tasks_log = []

        return self._get_all_obs()

    def _calculate_batch_time(self, batch_size: int) -> int:
        if batch_size == 0: return 0
        base = self.batch_proc_time_config.get('base', 2)
        per_task = self.batch_proc_time_config.get('per_task', 1)
        return base + per_task * batch_size

    def step(self, actions: list):
        user_actions = actions[:self.n_users]
        server_actions = actions[self.n_users:]

        for i in range(self.n_users):
            self._execute_user_action(i, user_actions[i])

        for i in range(self.n_servers):
            self._execute_server_action(i, server_actions[i])

        self._update_env_state()

        avg_aoi = np.mean([user['aoi'] for user in self.users_state])
        reward = -avg_aoi

        all_obs = self._get_all_obs()
        done = False
        info = {'average_aoi': avg_aoi}

        return all_obs, reward, done, info

    def _execute_user_action(self, user_id: int, action: int):
        user = self.users_state[user_id]
        if user['h'] == -1 and action > 0:
            server_idx = action - 1
            if 0 <= server_idx < self.n_servers:
                new_task = Task(
                    id=self.task_id_counter,
                    user_id=user_id,
                    generation_time=self.time_step,
                    assigned_server_id=server_idx  # 记录分配的服务器
                )
                self.task_id_counter += 1
                self.servers_state[server_idx]['q'].append(new_task)
                user['h'] = 0
                
    def _execute_server_action(self, server_id: int, action: int):
        server = self.servers_state[server_id]
        if server['t'] == -1 and len(server['q']) > 0 and action == 1:
            batch_size = min(len(server['q']), self.max_batch_size)
            batch_tasks = [server['q'].popleft() for _ in range(batch_size)]
            server['b'] = batch_tasks
            server['t'] = self._calculate_batch_time(len(server['b']))

    def _update_env_state(self):
        for user in self.users_state: #用户AoI每个时间步加一，如果任务正在处理，h也每个时间步加一
            user['aoi'] += 1
            if user['h'] != -1:
                user['h'] += 1
        for server_id, server in enumerate(self.servers_state):
            if server['t'] > 0:#如果server正在批处理，所需时间t每个时间步减一
                server['t'] -= 1
            if server['t'] == 0:#表示此时批处理完成
                completion_time = self.time_step
                for task in server['b']:
                    user_id = task.user_id
                    # 记录完成的任务信息
                    self.completed_tasks_log.append({
                        'user_id': user_id,
                        'task_id': task.id,
                        'server_id': task.assigned_server_id,
                        'generation_time': task.generation_time,
                        'completion_time': completion_time,
                        'latency': completion_time - task.generation_time
                    })
                    #重置AoI和h
                    if self.users_state[user_id]['h'] != -1:
                        self.users_state[user_id]['aoi'] = self.users_state[user_id]['h']
                        self.users_state[user_id]['h'] = -1
                server['b'] = []
                server['t'] = -1

        self.time_step += 1

    def _get_user_obs(self, user_id: int):
        user = self.users_state[user_id]
        return [user['h'], user['aoi']]

    def _get_server_obs(self, server_id: int):
        server = self.servers_state[server_id]
        return [len(server['q']), len(server['b']), server['t']]

    def _get_all_obs(self):
        return [self._get_user_obs(i) for i in range(self.n_users)] + \
            [self._get_server_obs(i) for i in range(self.n_servers)]

    def get_agent_action_mask(self):
        # 在user产生任务之后和server启动批处理之后，不应该动作，相应动作mask设置为False
        masks = []
        for i in range(self.n_users):
            mask = [self.users_state[i]['h'] == -1] * self.user_action_dim
            mask[0] = True
            masks.append(mask)
        for i in range(self.n_servers):
            server = self.servers_state[i]
            mask = [server['t'] == -1 and len(server['q']) > 0] * self.server_action_dim
            mask[0] = True
            masks.append(mask)
        return masks

    def get_global_state(self):
        user_obs_flat = [obs[0] for obs in self._get_all_obs()[:self.n_users]]
        server_obs_flat = [item for obs in self._get_all_obs()[self.n_users:] for item in obs]
        return np.array(user_obs_flat + server_obs_flat)

    def get_mean_field_state(self):
        """获取平均场"""
        # 提取所有智能体的状态
        user_aois = [u['aoi'] for u in self.users_state]
        user_hs = [u['h'] for u in self.users_state]
        server_q_lens = [len(s['q']) for s in self.servers_state]
        server_b_sizes = [len(s['b']) for s in self.servers_state]
        server_ts = [s['t'] for s in self.servers_state]

        # 计算直方图并归一化
        aoi_hist = np.histogram(user_aois, bins=self.mf_bins['aoi'])[0] / self.n_users
        h_hist = np.histogram(user_hs, bins=self.mf_bins['h'])[0] / self.n_users

        if self.n_servers > 0:
            q_len_hist = np.histogram(server_q_lens, bins=self.mf_bins['q_len'])[0] / self.n_servers
            b_size_hist = np.histogram(server_b_sizes, bins=self.mf_bins['b_size'])[0] / self.n_servers
            t_hist = np.histogram(server_ts, bins=self.mf_bins['t'])[0] / self.n_servers
        else:
            q_len_hist, b_size_hist, t_hist = [], [], []

        # 拼接成一个长向量
        mf_vector = np.concatenate([aoi_hist, h_hist, q_len_hist, b_size_hist, t_hist])
        return mf_vector

