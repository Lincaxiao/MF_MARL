import numpy as np
import collections
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class Task:
    id: int
    user_id: int
    generation_time: int
    assigned_server_id: int


class EdgeBatchEnv:
    def __init__(self, n_users: int, n_servers: int, batch_proc_time: dict, gmm_config: dict = None,
                 max_batch_size: int = 2):
        # print("Initializing Edge Batch Processing Environment...")
        self.n_users = n_users
        self.n_servers = n_servers
        self.batch_proc_time_config = batch_proc_time
        self.n_agents = self.n_users + self.n_servers
        self.max_batch_size = max_batch_size

        # ---- GMM 默认配置 ----
        default_gmm = {
            'n_components': 2,        # GMM组件数量
            'min_samples': 100,       # 最小样本数量（用于训练GMM）
            'use_gmm': True,          # 是否使用GMM
            'regularization': 1e-6,   # 正则化参数
            'update_interval': 50,    # GMM更新间隔（训练回合）
            'buffer_size': 5000,      # 滑动窗口大小（每个变量保留最近 buffer_size 个样本）
        }

        # 合并用户自定义配置
        if gmm_config is None:
            self.gmm_config = default_gmm
        else:
            # 允许用户部分覆盖，缺失字段使用默认值
            self.gmm_config = {**default_gmm, **gmm_config}

        # GMM模型存储
        self.gmm_models = {
            'user_aoi': None,
            'user_h': None,
            'server_q_len': None,
            'server_b_size': None,
            'server_t': None,
        }
        
        # 历史数据缓存（滑动窗口），用于训练GMM
        buf_len = self.gmm_config['buffer_size']
        self.history_buffer = {
            'user_aoi': collections.deque(maxlen=buf_len),
            'user_h': collections.deque(maxlen=buf_len),
            'server_q_len': collections.deque(maxlen=buf_len),
            'server_b_size': collections.deque(maxlen=buf_len),
            'server_t': collections.deque(maxlen=buf_len),
        }
        
        # GMM训练状态
        self.gmm_trained = False
        self.episodes_since_gmm_update = 0

        self.user_action_dim = self.n_servers + 1
        self.server_action_dim = 2

        self.reset()
        print(f"Environment created with {self.n_users} users and {self.n_servers} servers.")
        print(
            f"GMM: {self.gmm_config['n_components']} comps, window={self.gmm_config['buffer_size']}, "
            f"update every {self.gmm_config['update_interval']} episodes."
        )

    def collect_data_for_gmm(self):
        """收集当前时间步的数据用于GMM训练"""
        # 收集用户状态
        user_aois = [u['aoi'] for u in self.users_state]
        user_hs = [u['h'] for u in self.users_state]
        
        # 收集服务器状态
        server_q_lens = [len(s['q']) for s in self.servers_state]
        server_b_sizes = [len(s['b']) for s in self.servers_state]
        server_ts = [s['t'] for s in self.servers_state]
        
        # 添加到历史缓存
        self.history_buffer['user_aoi'].extend(user_aois)
        self.history_buffer['user_h'].extend(user_hs)
        self.history_buffer['server_q_len'].extend(server_q_lens)
        self.history_buffer['server_b_size'].extend(server_b_sizes)
        self.history_buffer['server_t'].extend(server_ts)

    def train_gmm_models(self):
        """从历史数据训练GMM模型"""
        if not self.gmm_config['use_gmm']:
            print("GMM训练被禁用，跳过...")
            return False
            
        print("开始训练GMM模型...")
        success_count = 0
        
        for var_name, data in self.history_buffer.items():
            data_list = list(data)
            if len(data_list) < self.gmm_config['min_samples']:
                print(f"变量 {var_name} 的样本数不足 ({len(data)} < {self.gmm_config['min_samples']})，跳过训练")
                continue
                
            try:
                # 准备数据
                data_array = np.array(data_list, dtype=np.float32).reshape(-1, 1)
                
                # 训练GMM
                gmm = GaussianMixture(
                    n_components=self.gmm_config['n_components'],
                    covariance_type='spherical',
                    reg_covar=self.gmm_config['regularization'],
                    max_iter=100,
                    random_state=42
                )
                gmm.fit(data_array)
                
                # 存储训练好的模型
                self.gmm_models[var_name] = gmm
                success_count += 1
                print(f"成功训练 {var_name} 的GMM模型 (样本数: {len(data)})")
                
            except Exception as e:
                print(f"训练 {var_name} 的GMM模型失败: {e}")
                self.gmm_models[var_name] = None
        
        if success_count > 0:
            self.gmm_trained = True
            print(f"GMM训练完成，成功训练 {success_count} 个模型")
            return True
        else:
            print("GMM训练失败，没有成功训练任何模型")
            return False

    def clear_history_buffer(self):
        """清空历史数据缓存"""
        for key in self.history_buffer:
            self.history_buffer[key] = []

    def should_update_gmm(self):
        """判断是否应该更新GMM"""
        return (self.episodes_since_gmm_update >= self.gmm_config['update_interval'] and
                self.gmm_config['use_gmm'])

    def episode_end_callback(self):
        """回合结束时的回调函数，处理GMM更新逻辑"""
        self.episodes_since_gmm_update += 1
        
        if self.should_update_gmm():
            success = self.train_gmm_models()
            if success:
                self.episodes_since_gmm_update = 0
                # 训练完成后可以选择清空缓存来节省内存
                # self.clear_history_buffer()

    def _predict_with_gmm(self, data: list, variable_name: str):
        """使用预训练的GMM进行推断"""
        if not data:
            # 没有数据时返回零向量
            return np.zeros(self.gmm_config['n_components'])
            
        # 检查是否有预训练的GMM模型
        gmm = self.gmm_models.get(variable_name)
        if gmm is None:
            # 没有训练好的模型，使用简单统计量
            return self._compute_simple_features(data)
        
        try:
            # 使用GMM计算后验概率
            data_array = np.array(data, dtype=np.float32).reshape(-1, 1)
            # 计算每个数据点属于各个组件的后验概率
            posteriors = gmm.predict_proba(data_array)  # Shape: (n_samples, n_components)
            # 返回平均后验概率作为特征
            mean_posteriors = np.mean(posteriors, axis=0)
            return mean_posteriors
            
        except Exception as e:
            # print(f"GMM推断失败 ({variable_name}): {e}")
            return self._compute_simple_features(data)
    
    def _compute_simple_features(self, data: list):
        """计算简单特征作为GMM的回退"""
        n_comp = self.gmm_config['n_components']
        features = np.zeros(n_comp)
        
        if data:
            # 使用数据的分位数作为特征
            data_array = np.array(data, dtype=np.float32)
            if n_comp >= 2:
                features[0] = np.mean(data_array < np.median(data_array))  # 低于中位数的比例
                features[1] = np.mean(data_array >= np.median(data_array))  # 高于等于中位数的比例
            if n_comp >= 3:
                q75 = np.percentile(data_array, 75)
                features[2] = np.mean(data_array >= q75)  # 高于75%分位数的比例
        
        # 确保概率和为1
        if np.sum(features) > 0:
            features = features / np.sum(features)
        else:
            features[0] = 1.0  # 默认情况
            
        return features

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

        # 收集当前状态数据用于GMM训练
        self.collect_data_for_gmm()

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
        """获取基于GMM推断的平均场状态表示"""
        # 提取当前时间步的状态数据
        user_aois = [u['aoi'] for u in self.users_state]
        user_hs = [u['h'] for u in self.users_state]
        server_q_lens = [len(s['q']) for s in self.servers_state]
        server_b_sizes = [len(s['b']) for s in self.servers_state]
        server_ts = [s['t'] for s in self.servers_state]

        # 使用预训练的GMM进行快速推断
        mf_features = []
        
        # 用户AoI分布的后验概率
        aoi_features = self._predict_with_gmm(user_aois, "user_aoi")
        mf_features.append(aoi_features)
        
        # 用户任务计时h分布的后验概率
        h_features = self._predict_with_gmm(user_hs, "user_h")
        mf_features.append(h_features)
        
        if self.n_servers > 0:
            # 服务器队列长度分布的后验概率
            q_len_features = self._predict_with_gmm(server_q_lens, "server_q_len")
            mf_features.append(q_len_features)
            
            # 服务器批处理大小分布的后验概率
            b_size_features = self._predict_with_gmm(server_b_sizes, "server_b_size")
            mf_features.append(b_size_features)
            
            # 服务器剩余处理时间分布的后验概率
            t_features = self._predict_with_gmm(server_ts, "server_t")
            mf_features.append(t_features)

        # 拼接所有特征向量
        mf_vector = np.concatenate(mf_features) if mf_features else np.array([])
        return mf_vector

