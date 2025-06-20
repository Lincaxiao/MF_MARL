from env import EdgeBatchEnv
from model import MASACAgent
from utils import plot_train_test_curves, save_model, append_csv_row, save_completed_tasks_to_csv, evaluate_aoi
import numpy as np
import yaml
import os
import csv
from pathlib import Path


def push_episode_to_buffer(traj_list, agent):
    """将一次 episode 的轨迹列表写入 agent 的 replay buffer
    traj 元素格式: (obs, act, logp, reward, val, gs, adv, return, done)
    只取 obs, act, reward, gs 字段
    """
    if not traj_list:
        return
    for i, trans in enumerate(traj_list):
        obs, act, _, reward, _, gs, _, _, done_flag = trans
        # 下一个观测: 若不是最后一步则取下一条的 obs / gs
        if i < len(traj_list) - 1:
            next_obs = traj_list[i + 1][0]
            next_gs = traj_list[i + 1][5]
            done = False
        else:
            next_obs = np.zeros_like(obs, dtype=np.float32)
            next_gs = np.zeros_like(gs, dtype=np.float32)
            done = True
        agent.store(obs, act, reward, next_obs, done, gs, next_gs)


def train_sac_with_log(env,
                        agent_u_wait,
                        agent_u_assign,
                        agent_s,
                        lamda_init=1.0,
                        reward_tol=1e-2,
                        reward_window=10,
                        lamda_tol=1e-3,
                        max_outer_iter=20,
                        epochs_per_lamda=1000,
                        log_interval=10,
                        outdir="./trainlog_sac/sub"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    train_logfile = os.path.join(outdir, "train_history.csv")
    model_path = os.path.join(outdir, "masac_checkpoint")
    header = ['epoch', 'train_user_reward', 'train_server_reward', 'test_user_reward', 'test_server_reward', 'test_mean_AoI', 'lamda']

    # 读取 config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyperparams = config.get('hyperparameters', {})

    # 保存参数信息到 CSV
    param_info = {
        'NUM_USERS': env.num_users,
        'NUM_SERVERS': env.num_servers,
        'MAX_QUEUE': env.max_queue_len,
        'EPISODE_LIMIT': env.episode_limit,
        'MAX_WAITING_TIME': globals().get('MAX_WAITING_TIME', None),
        'lamda_init': lamda_init,
        'reward_tol': reward_tol,
        'reward_window': reward_window,
        'lamda_tol': lamda_tol,
        'max_outer_iter': max_outer_iter,
        'epochs_per_lamda': epochs_per_lamda
    }
    Path(os.path.dirname(train_logfile)).mkdir(parents=True, exist_ok=True)
    with open(train_logfile, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in param_info.items():
            writer.writerow([f"# {k}", v])
        for k, v in hyperparams.items():
            writer.writerow([f"# hyper_{k}", v])
        writer.writerow(header)

    # 训练相关超参
    train_batch_size = hyperparams.get('batch_size', 256)
    gamma = hyperparams.get('gamma', 0.99)
    lam = hyperparams.get('lam', 0.95)  # SAC 不用，但保留
    updates_per_step = hyperparams.get('updates_per_step', 1)

    env.lamda = lamda_init
    alpha = 0.2  # lamda 更新系数

    outer_iter = 0
    while True:
        outer_iter += 1
        print(f"=== Lamda Iter {outer_iter}, lamda={env.lamda:.6f} ===")
        reward_hist = []
        ep = 0
        converged = False
        while not converged and ep < epochs_per_lamda:
            ep += 1
            _ = env.reset()
            user_trajs_wait = [[] for _ in range(env.num_users)]
            user_trajs_assign = [[] for _ in range(env.num_users)]
            server_trajs = [[] for _ in range(env.num_servers)]

            while not env.is_done():
                event = env.get_next_event()
                if event is None:
                    break
                if isinstance(event, tuple) and len(event) == 3 and event[0] == 'user':
                    _, idx, phase = event
                    obs_u = env.get_user_obs(idx)
                    gs = env.get_obs_all()
                    if phase == 'wait':
                        w, _ = agent_u_wait.select(obs_u)
                        env.step_user_wait(idx, w)
                        user_trajs_wait[idx].append((obs_u, w, 0.0, 0.0, 0.0, gs, 0, 0, False))
                    elif phase == 'assign':
                        act, _ = agent_u_assign.select(obs_u)
                        serv_idx = env.decode_user_assign_action(int(round(act)))
                        env.step_user_assign(idx, serv_idx)
                        user_trajs_assign[idx].append((obs_u, serv_idx, 0.0, 0.0, 0.0, gs, 0, 0, False))
                elif isinstance(event, tuple) and len(event) == 3 and event[0] == 'server':
                    _, idx, phase = event
                    obs_s = env.get_server_obs(idx)
                    gs = env.get_obs_all()
                    if phase == 'start':
                        act, _ = agent_s.select(obs_s)
                        act_d = int(round(act))
                        env.step_server_start(idx, act_d)
                        server_trajs[idx].append((obs_s, act_d, 0.0, 0.0, 0.0, gs, 0, 0, False))
                    elif phase == 'end':
                        server_reward, u_rewards = env.step_server_end(idx)
                        from utils import scale_rewards
                        server_reward_norm, u_rewards_norm = scale_rewards(server_reward, u_rewards, 1)
                        server_trajs[idx][-1] = server_trajs[idx][-1][:3] + (server_reward_norm,) + server_trajs[idx][-1][4:]
                        for uid, reward in u_rewards_norm.items():
                            # 回填用户奖励到最近一次 assign/wait 动作
                            for buffer_trajs in (user_trajs_assign[uid], user_trajs_wait[uid]):
                                for i in reversed(range(len(buffer_trajs))):
                                    if buffer_trajs[i][3] == 0.0:
                                        buffer_trajs[i] = buffer_trajs[i][:3] + (reward,) + buffer_trajs[i][4:]
                                        break

            # 标记 done
            for i in range(env.num_users):
                if user_trajs_wait[i]:
                    user_trajs_wait[i][-1] = user_trajs_wait[i][-1][:-1] + (True,)
                if user_trajs_assign[i]:
                    user_trajs_assign[i][-1] = user_trajs_assign[i][-1][:-1] + (True,)
            for i in range(env.num_servers):
                if server_trajs[i]:
                    server_trajs[i][-1] = server_trajs[i][-1][:-1] + (True,)

            # 写入 replay buffer
            for i in range(env.num_users):
                push_episode_to_buffer(user_trajs_wait[i], agent_u_wait)
                push_episode_to_buffer(user_trajs_assign[i], agent_u_assign)
            for i in range(env.num_servers):
                push_episode_to_buffer(server_trajs[i], agent_s)

            # 训练 (按 experience 数量决定更新次数)
            agent_u_wait.train(batch_size=train_batch_size, updates=len(user_trajs_wait) * updates_per_step)
            agent_u_assign.train(batch_size=train_batch_size, updates=len(user_trajs_assign) * updates_per_step)
            agent_s.train(batch_size=train_batch_size, updates=len(server_trajs) * updates_per_step)

            # 统计 reward
            train_r_user = sum([x[3] for buf in user_trajs_assign for x in buf])
            train_r_serv = sum([x[3] for buf in server_trajs for x in buf])
            reward_hist.append(train_r_user)

            # 日志
            if ep % log_interval == 0 or ep == 1:
                aoi_per_user, train_mean_aoi = evaluate_aoi(env.completed_tasks, env.t, env.num_users)
                append_csv_row(train_logfile, [ep, train_r_user, train_r_serv, 0, 0, train_mean_aoi, env.lamda], header if ep == 1 else None)
                print(f"E{ep} | TrainU={train_r_user / env.num_users:.2f} TrainS={train_r_serv:.2f} MeanAoI={train_mean_aoi:.4f} Lamda={env.lamda:.4f}")
                # 周期性保存 completed_tasks，与 train.py 保持一致
                if ep % 5 == 0 or ep == 1:
                    completed_dir = os.path.join(outdir, "completed_tasks")
                    Path(completed_dir).mkdir(parents=True, exist_ok=True)
                    save_completed_tasks_to_csv(env.completed_tasks,
                                                os.path.join(completed_dir, f"it_{outer_iter}_ep_{ep}.csv"))

            # 收敛检测
            if len(reward_hist) >= reward_window:
                window = reward_hist[-reward_window:]
                if np.max(window) - np.min(window) < reward_tol:
                    converged = True

        # lamda 更新
        numer, denom, count = 0.0, 0.0, 0
        for user_tasks in env.completed_tasks:
            for rec in user_tasks:
                t_km1, q_km1, t_k, q_k, w = rec[0], rec[1], rec[2], rec[3], rec[4]
                d = (q_km1 - t_km1 + w)
                if abs(d) > 1e-8:
                    numer += 0.5 * ((q_k - t_km1) ** 2 - (q_k - t_k) ** 2)
                    denom += d
                    count += 1
        new_lamda = numer / denom if denom != 0 else env.lamda
        print(f"Update lamda: {env.lamda:.6f} -> {new_lamda:.6f} (samples={count})")
        if abs(new_lamda - env.lamda) < lamda_tol or outer_iter >= max_outer_iter:
            print("Lamda converged, training finished.")
            break
        env.lamda = alpha * new_lamda + (1 - alpha) * env.lamda

    save_model(agent_u_wait, model_path + "_user_wait")
    save_model(agent_u_assign, model_path + "_user_assign")
    save_model(agent_s, model_path + "_serv")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    NUM_USERS = config['env']['Num_users']
    NUM_SERVERS = config['env']['Num_servers']
    MAX_QUEUE = config['env']['Max_queue_len']
    EPISODE_LIMIT = config['env']['Episode_limit']
    MAX_BATCH_SIZE = config['env']['Max_batch_size']
    MAX_WAITING_TIME = config['env']['Max_waiting_time']

    LAMDA = 1
    USER_OBS_DIM = 3 + NUM_SERVERS * 2
    SVR_OBS_DIM = 1 + MAX_QUEUE
    GS_DIM = NUM_SERVERS * (2 + MAX_QUEUE) + NUM_USERS * 3

    ENV = EdgeBatchEnv(num_users=NUM_USERS,
                       num_servers=NUM_SERVERS,
                       max_queue_len=MAX_QUEUE,
                       episode_limit=EPISODE_LIMIT,
                       lamda=LAMDA)

    user_agent_wait = MASACAgent(obs_dim=USER_OBS_DIM, act_dim=1, global_dim=GS_DIM,
                                 action_low=1.0, action_high=MAX_WAITING_TIME)
    user_agent_assign = MASACAgent(obs_dim=USER_OBS_DIM, act_dim=1, global_dim=GS_DIM,
                                   action_low=0, action_high=ENV.user_assign_action_space() - 1)
    server_agent = MASACAgent(obs_dim=SVR_OBS_DIM, act_dim=1, global_dim=GS_DIM,
                              action_low=0, action_high=ENV.server_action_space() - 1)

    train_sac_with_log(ENV,
                       user_agent_wait,
                       user_agent_assign,
                       server_agent,
                       lamda_init=10,
                       reward_tol=5,
                       reward_window=5,
                       lamda_tol=0.05,
                       max_outer_iter=20,
                       epochs_per_lamda=50,
                       log_interval=1,
                       outdir="./trainlog/train_sac") 