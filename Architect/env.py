import yaml
from utils import linear_zeta, log_zeta, get_batch_reward_coef
import math
import numpy as np

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
MAX_WAITING_TIME = config['env']['Max_waiting_time']
WAIT_OPTIONS = [i for i in range(1, MAX_WAITING_TIME + 1)]


class EdgeBatchEnv:
    def __init__(self, num_users=3, num_servers=2, max_queue_len=8, episode_limit=500, lamda=1):
        self.num_users = num_users
        self.num_servers = num_servers
        self.max_queue_len = max_queue_len
        self.episode_limit = episode_limit
        self.lamda = lamda
        self.reset()

    def reset(self):
        self.t = 0.0
        self.steps = 0
        self.batch_idx = 0
        self.user_states = []
        self.user_next_action_time = []  # [(time, 'wait'/'assign')]
        for _ in range(self.num_users):
            # 随机初始化user的首次决策时刻
            init_time = np.random.uniform(0, 5)
            self.user_next_action_time.append((init_time, 'wait'))

            st = {'t_km1': 0.0, 'q_km1': init_time, 'w': 0.0, 't_k': 0.0, 'q_k': 0.0, 'phase': 'wait',
                  'pending_wait': None}
            self.user_states.append(st)

        self.server_queues = [[] for _ in range(self.num_servers)]
        self.server_next_action_time = [(float('inf'), 'start') for _ in
                                        range(self.num_servers)]  # (time, 'start'/'end')
        self.completed_tasks = [[] for _ in range(self.num_users)]
        return self.get_obs_all()

    def get_user_obs(self, idx):
        st = self.user_states[idx]
        obs = [st['t_km1'], st['q_km1'], st['w']]  # 删 w
        obs += [len(q) for q in self.server_queues]
        # server空闲状态：server_next_action_time[i][1] == 'start' 判定
        obs += [1.0 if self.server_next_action_time[i][1] == 'start' else 0.0 for i in range(self.num_servers)]
        return np.array(obs, dtype=np.float32)

    def get_server_obs(self, idx):
        queue = self.server_queues[idx]
        obs = [len(queue)]
        # 到达时间序列，长度补齐到max_queue_len
        arrival_times = [self.t - task['t_new'] for task in queue]
        while len(arrival_times) < self.max_queue_len:
            arrival_times.append(0.0)
        obs += arrival_times[:self.max_queue_len]
        return np.array(obs, dtype=np.float32)

    def get_obs_all(self):
        # 全局状态：所有user的[t_km1, q_km1, w] + 所有server队列长度+所有server队列到达时间序列
        obs = []
        # 用户部分
        for i in range(self.num_users):
            st = self.user_states[i]
            obs += [st['t_km1'], st['q_km1'], st['w']]
        # 服务器部分
        for j in range(self.num_servers):
            queue = self.server_queues[j]
            obs.append(len(queue))
            arrival_times = [self.t - task['t_new'] for task in queue]
            while len(arrival_times) < self.max_queue_len:
                arrival_times.append(0.0)
            obs += arrival_times[:self.max_queue_len]
        # 新增：每个server是否空闲（1.0=空闲，0.0=忙）
        obs += [1.0 if self.server_next_action_time[j][1] == 'start' else 0.0 for j in range(self.num_servers)]
        return np.array(obs, dtype=np.float32)

    def user_wait_action_space(self):
        return len(WAIT_OPTIONS)

    def user_assign_action_space(self):
        return self.num_servers

    def server_action_space(self):
        # 0表示不处理，1表示批处理max(len(queue), max_batch_size)
        return 2

    def decode_user_wait_action(self, act):
        return act  # wait_idx

    def decode_user_assign_action(self, act):
        return act  # serv_idx

    def decode_server_action(self, act, queue_len):
        # act=0表示不处理，1~max_queue_len表示批处理数量
        batch_size = act
        if batch_size > queue_len:
            batch_size = queue_len
        return batch_size

    def step_user_wait(self, idx, w):
        st = self.user_states[idx]
        # 连续动作，裁剪到[1, MAX_WAITING_TIME]
        # w = float(np.clip(w, 1, MAX_WAITING_TIME))
        t_k = st['q_km1'] + w
        st['w'] = w
        st['t_k'] = t_k
        st['phase'] = 'assign'
        st['pending_wait'] = w
        self.user_next_action_time[idx] = (t_k, 'assign')  # 下一步到t_k时分配服务器
        return

    def step_user_assign(self, idx, serv_idx):
        st = self.user_states[idx]
        t_k = st['t_k']
        w = st['pending_wait']
        # 入队（只要队未满，真正投入时间为t_k）
        if len(self.server_queues[serv_idx]) <= self.max_queue_len:
            task = {'user_id': idx, 't_new': t_k}
            self.server_queues[serv_idx].append(task)
            # 判断server是否空闲
            if self.server_next_action_time[serv_idx][1] == 'start':
                # 空闲，立即触发server_start
                self.server_next_action_time[serv_idx] = (t_k, 'start')
            # 否则什么都不做，等server_end时再判断
        st['phase'] = 'wait'
        st['pending_wait'] = None
        # 标记该用户暂时无事件，直到server处理后激活
        self.user_next_action_time[idx] = (float('inf'), 'wait')
        return

    def step_server_start(self, serv_idx, act):
        # 是否启动批处理，只安排批处理结束时间，不做任务出队
        q = self.server_queues[serv_idx]
        max_batch_size = config['env']['Max_batch_size']
        # 强制处理
        if len(q) >= max_batch_size and act == 0:
            act = 1

        if act == 0:
            # 不处理任务，server空闲
            self.server_next_action_time[serv_idx] = (float('inf'), 'start')
            return

        batch_size = min(len(q), max_batch_size)
        if batch_size == 1:
            proc_time = 6
        else:
            proc_time = 2 * batch_size + 4
        finish_time = self.t + proc_time
        self.server_next_action_time[serv_idx] = (finish_time, 'end')
        return

    def step_server_end(self, serv_idx):
        # 批处理完成，真正出队、奖励分配、唤起user
        q = self.server_queues[serv_idx]
        batch_size = min(len(q), config['env']['Max_batch_size'])
        server_reward = 0
        user_reward_table = {}
        batch = [q.pop(0) for _ in range(batch_size)]
        t_arrival = [task['t_new'] for task in batch]
        coef_arr = get_batch_reward_coef(t_arrival)
        for task, coef in zip(batch, coef_arr):
            user_id = task['user_id']
            t_k = task['t_new']
            st = self.user_states[user_id]
            t_km1, q_km1, w = st['t_km1'], st['q_km1'], st['w']
            q_k = self.t
            # base_ur = (0.5 * (q_k - t_km1) ** 2 - 0.5 * (q_k - t_k) ** 2) / (q_km1 - t_km1 + w)
            base_ur = (0.5 * (q_k - t_km1) ** 2 - 0.5 * (q_k - t_k) ** 2) - self.lamda * (q_km1 - t_km1 + w)
            base_ur = -base_ur
            coef = 1.0  # 忽略折扣系数
            ur = base_ur * coef
            st['t_km1'] = t_k
            st['q_km1'] = q_k
            self.completed_tasks[user_id].append([t_km1, q_km1, t_k, q_k, w, ur, batch_size, self.batch_idx])
            user_reward_table[user_id] = ur
            self.user_next_action_time[user_id] = (q_k, 'wait')
            server_reward += ur
        self.batch_idx += 1
        # 检查队列是否有新任务，有则再次触发server_start，否则server空闲
        if len(self.server_queues[serv_idx]) > 0:
            self.server_next_action_time[serv_idx] = (self.t, 'start')
        else:
            self.server_next_action_time[serv_idx] = (float('inf'), 'start')
        return server_reward, user_reward_table

    def get_next_event(self):
        # 下一个user动作及server动作（事件最早者先发生）
        user_events = [(self.user_next_action_time[i][0], ('user', i, self.user_next_action_time[i][1])) for i in
                       range(self.num_users)]
        server_events = []
        for j in range(self.num_servers):
            t, phase = self.server_next_action_time[j]
            if t < float('inf'):
                server_events.append((t, ('server', j, phase)))
        min_user = min(user_events) if user_events else (float('inf'), None)
        min_server = min(server_events) if server_events else (float('inf'), None)
        # 谁的事件最早
        if min_user[0] < min_server[0]:
            self.t = min_user[0]
            return min_user[1]
        elif min_server[1] is not None:
            self.t = min_server[0]
            return min_server[1]
        else:
            return None

    def is_done(self):
        self.steps += 1
        return self.steps >= self.episode_limit