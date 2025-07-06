from env import EdgeBatchEnv
from model import MATD3Agent
from utils import save_model, append_csv_row, evaluate_aoi, save_completed_tasks_to_csv
import numpy as np
import yaml, os, csv
from pathlib import Path


def push_episode_to_buffer(traj_list, agent):
    """把 episode 轨迹写入 replay buffer（MATD3 与 MASAC 接口一致）"""
    if not traj_list:
        return
    for i, trans in enumerate(traj_list):
        obs, act, _, reward, _, gs, _, _, done_flag = trans
        if i < len(traj_list) - 1:
            next_obs = traj_list[i + 1][0]
            next_gs = traj_list[i + 1][5]
            done = False
        else:
            next_obs = np.zeros_like(obs, dtype=np.float32)
            next_gs = np.zeros_like(gs, dtype=np.float32)
            done = True
        agent.store(obs, act, reward, next_obs, done, gs, next_gs)


def train_td3_with_log(env,
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
                       outdir="./trainlog_td3/sub"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    train_logfile = os.path.join(outdir, "train_history.csv")
    model_path = os.path.join(outdir, "matd3_checkpoint")
    header = ['epoch', 'train_user_reward', 'train_server_reward', 'test_user_reward', 'test_server_reward', 'test_mean_AoI', 'lamda']

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyper = config.get('hyperparameters', {})

    # 写 CSV 头
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
    with open(train_logfile, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in param_info.items():
            writer.writerow([f"# {k}", v])
        for k, v in hyper.items():
            writer.writerow([f"# hyper_{k}", v])
        writer.writerow(header)

    batch_size = hyper.get('batch_size', 256)
    updates_per_step = hyper.get('updates_per_step', 1)
    start_timesteps = hyper.get('start_timesteps', 500000)

    env.lamda = lamda_init
    alpha = 0.2
    total_timesteps = 0
    outer_iter = 0

    print(f"Populating replay buffer with {start_timesteps} random steps...")
    while total_timesteps < start_timesteps:
        _ = env.reset()
        user_trajs_wait = [[] for _ in range(env.num_users)]
        user_trajs_assign = [[] for _ in range(env.num_users)]
        server_trajs = [[] for _ in range(env.num_servers)]

        while not env.is_done() and total_timesteps < start_timesteps:
            event = env.get_next_event()
            if event is None:
                break
            if event[0] == 'user':
                _, idx, phase = event
                obs_u = env.get_user_obs(idx)
                gs = env.get_obs_all()
                if phase == 'wait':
                    w = np.random.uniform(agent_u_wait.action_low, agent_u_wait.action_high)
                    env.step_user_wait(idx, w)
                    user_trajs_wait[idx].append((obs_u, w, 0.0, 0.0, 0.0, gs, 0, 0, False))
                else:  # assign
                    act = np.random.uniform(agent_u_assign.action_low, agent_u_assign.action_high)
                    serv_idx = env.decode_user_assign_action(int(round(act)))
                    env.step_user_assign(idx, serv_idx)
                    user_trajs_assign[idx].append((obs_u, serv_idx, 0.0, 0.0, 0.0, gs, 0, 0, False))
                total_timesteps += 1
            elif event[0] == 'server':
                _, idx, phase = event
                obs_s = env.get_server_obs(idx)
                gs = env.get_obs_all()
                if phase == 'start':
                    act = np.random.uniform(agent_s.action_low, agent_s.action_high)
                    act_d = int(round(act))
                    env.step_server_start(idx, act_d)
                    server_trajs[idx].append((obs_s, act_d, 0.0, 0.0, 0.0, gs, 0, 0, False))
                else:  # end
                    server_reward, u_rewards = env.step_server_end(idx)
                    from utils import scale_rewards
                    sr_norm, ur_norm = scale_rewards(server_reward, u_rewards, 1)
                    server_trajs[idx][-1] = server_trajs[idx][-1][:3] + (sr_norm,) + server_trajs[idx][-1][4:]
                    for uid, r in ur_norm.items():
                        for cache in (user_trajs_assign[uid], user_trajs_wait[uid]):
                            for i in reversed(range(len(cache))):
                                if cache[i][3] == 0.0:
                                    cache[i] = cache[i][:3] + (r,) + cache[i][4:]
                                    break
                total_timesteps += 1

        for lst in (*user_trajs_wait, *user_trajs_assign, *server_trajs):
            if lst:
                lst[-1] = lst[-1][:-1] + (True,)
        
        for i in range(env.num_users):
            ################################################
            # 强制修改 wait time 用于调试
            # for j in range(len(user_trajs_wait[i])):
            #     obs, _, logp, reward, val, gs, adv, ret, done = user_trajs_wait[i][j]
            #     user_trajs_wait[i][j] = (obs, 10.0, logp, reward, val, gs, adv, ret, done)
            ################################################
            push_episode_to_buffer(user_trajs_wait[i], agent_u_wait)
            push_episode_to_buffer(user_trajs_assign[i], agent_u_assign)
        for i in range(env.num_servers):
            push_episode_to_buffer(server_trajs[i], agent_s)

    print("Replay buffer populated. Starting training...")

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
                if event[0] == 'user':
                    _, idx, phase = event
                    obs_u = env.get_user_obs(idx)
                    gs = env.get_obs_all()
                    if phase == 'wait':
                        w, _ = agent_u_wait.select(obs_u)
                        env.step_user_wait(idx, w)
                        user_trajs_wait[idx].append((obs_u, w, 0.0, 0.0, 0.0, gs, 0, 0, False))
                    else:  # assign
                        act, _ = agent_u_assign.select(obs_u)
                        serv_idx = env.decode_user_assign_action(int(round(act)))
                        env.step_user_assign(idx, serv_idx)
                        user_trajs_assign[idx].append((obs_u, serv_idx, 0.0, 0.0, 0.0, gs, 0, 0, False))
                elif event[0] == 'server':
                    _, idx, phase = event
                    obs_s = env.get_server_obs(idx)
                    gs = env.get_obs_all()
                    if phase == 'start':
                        act, _ = agent_s.select(obs_s)
                        act_d = int(round(act))
                        env.step_server_start(idx, act_d)
                        server_trajs[idx].append((obs_s, act_d, 0.0, 0.0, 0.0, gs, 0, 0, False))
                    else:  # end
                        server_reward, u_rewards = env.step_server_end(idx)
                        from utils import scale_rewards
                        sr_norm, ur_norm = scale_rewards(server_reward, u_rewards, 1)
                        server_trajs[idx][-1] = server_trajs[idx][-1][:3] + (sr_norm,) + server_trajs[idx][-1][4:]
                        for uid, r in ur_norm.items():
                            for cache in (user_trajs_assign[uid], user_trajs_wait[uid]):
                                for i in reversed(range(len(cache))):
                                    if cache[i][3] == 0.0:
                                        cache[i] = cache[i][:3] + (r,) + cache[i][4:]
                                        break
            # 标记 done
            for lst in (*user_trajs_wait, *user_trajs_assign, *server_trajs):
                if lst:
                    lst[-1] = lst[-1][:-1] + (True,)

            # 写 buffer
            for i in range(env.num_users):
                push_episode_to_buffer(user_trajs_wait[i], agent_u_wait)
                push_episode_to_buffer(user_trajs_assign[i], agent_u_assign)
            for i in range(env.num_servers):
                push_episode_to_buffer(server_trajs[i], agent_s)

            # 训练
            if total_timesteps >= start_timesteps:
                agent_u_wait.train(batch_size=batch_size, updates=updates_per_step)
                agent_u_assign.train(batch_size=batch_size, updates=updates_per_step)
                agent_s.train(batch_size=batch_size, updates=updates_per_step)

            train_r_user = sum(x[3] for buf in user_trajs_assign for x in buf)
            train_r_serv = sum(x[3] for buf in server_trajs for x in buf)
            reward_hist.append(train_r_user)
            if ep % log_interval == 0 or ep == 1:
                _, mean_aoi = evaluate_aoi(env.completed_tasks, env.t, env.num_users)
                append_csv_row(train_logfile, [ep, train_r_user, train_r_serv, 0, 0, mean_aoi, env.lamda], header if ep == 1 else None)
                print(f"E{ep} | TrainU={train_r_user / env.num_users:.2f} TrainS={train_r_serv:.2f} MeanAoI={mean_aoi:.4f} Lamda={env.lamda:.4f}")
                # 周期性保存 completed_tasks
                if ep % 5 == 0 or ep == 1:
                    completed_dir = os.path.join(outdir, "completed_tasks")
                    Path(completed_dir).mkdir(parents=True, exist_ok=True)
                    save_completed_tasks_to_csv(env.completed_tasks,
                                                os.path.join(completed_dir, f"it_{outer_iter}_ep_{ep}.csv"))
            if len(reward_hist) >= reward_window and (max(reward_hist[-reward_window:]) - min(reward_hist[-reward_window:]) < reward_tol):
                converged = True

        # lamda 更新
        numer = denom = 0.0
        for tasks in env.completed_tasks:
            for rec in tasks:
                t_km1, q_km1, t_k, q_k, w = rec[:5]
                d = q_km1 - t_km1 + w
                if abs(d) > 1e-8:
                    numer += 0.5 * ((q_k - t_km1) ** 2 - (q_k - t_k) ** 2)
                    denom += d
        new_lam = numer / denom if denom != 0 else env.lamda
        print(f"Update lamda: {env.lamda:.6f} -> {new_lam:.6f}")
        if abs(new_lam - env.lamda) < lamda_tol or outer_iter >= max_outer_iter:
            print("Lamda converged, training finished.")
            break
        env.lamda = alpha * new_lam + (1 - alpha) * env.lamda

    save_model(agent_u_wait, model_path + "_user_wait")
    save_model(agent_u_assign, model_path + "_user_assign")
    save_model(agent_s, model_path + "_serv")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    NUM_USERS = cfg['env']['Num_users']
    NUM_SERVERS = cfg['env']['Num_servers']
    MAX_QUEUE = cfg['env']['Max_queue_len']
    EPISODE_LIMIT = cfg['env']['Episode_limit']
    MAX_WAITING_TIME = cfg['env']['Max_waiting_time']

    GS_DIM = NUM_SERVERS * (2 + MAX_QUEUE) + NUM_USERS * 3
    USER_OBS_DIM = 3 + NUM_SERVERS * 2
    SVR_OBS_DIM = 1 + MAX_QUEUE

    ENV = EdgeBatchEnv(NUM_USERS, NUM_SERVERS, MAX_QUEUE, EPISODE_LIMIT, lamda=1)

    hyper = cfg.get('hyperparameters', {})
    user_agent_wait = MATD3Agent(obs_dim=USER_OBS_DIM, act_dim=1, global_dim=GS_DIM,
                                 action_low=1.0, action_high=MAX_WAITING_TIME,
                                 expl_noise=hyper.get('expl_noise', 0.1),
                                 policy_noise=hyper.get('policy_noise', 0.2),
                                 noise_clip=hyper.get('noise_clip', 0.3))
    user_agent_assign = MATD3Agent(obs_dim=USER_OBS_DIM, act_dim=1, global_dim=GS_DIM,
                                   action_low=0, action_high=ENV.user_assign_action_space() - 1,
                                   expl_noise=hyper.get('expl_noise', 0.1),
                                   policy_noise=hyper.get('policy_noise', 0.2),
                                   noise_clip=hyper.get('noise_clip', 0.3))
    server_agent = MATD3Agent(obs_dim=SVR_OBS_DIM, act_dim=1, global_dim=GS_DIM,
                              action_low=0, action_high=ENV.server_action_space() - 1,
                              expl_noise=hyper.get('expl_noise', 0.1),
                              policy_noise=hyper.get('policy_noise', 0.2),
                              noise_clip=hyper.get('noise_clip', 0.3))

    train_td3_with_log(ENV, user_agent_wait, user_agent_assign, server_agent,
                       lamda_init=10, reward_tol=5, reward_window=5, lamda_tol=0.05,
                       max_outer_iter=60, epochs_per_lamda=100, log_interval=1,
                       outdir="./trainlog/train_td3") 