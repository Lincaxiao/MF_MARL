from env import EdgeBatchEnv
from model import MAPPOAgent, compute_gae
from utils import plot_train_test_curves, save_model, append_csv_row, save_completed_tasks_to_csv
import numpy as np
import yaml
import os
import csv
from pathlib import Path


def train_with_log(env, agent_u_wait, agent_u_assign, agent_s, lamda_init=1.0, reward_tol=1e-2, reward_window=10,
                   lamda_tol=1e-3,
                   max_outer_iter=20, epochs_per_lamda=1000, log_interval=10, outdir="./trainlog/sub"):
    import numpy as np
    import os
    import csv
    import yaml
    from pathlib import Path
    Path(outdir).mkdir(parents=True, exist_ok=True)
    train_logfile = os.path.join(outdir, "train_history.csv")
    model_path = os.path.join(outdir, "mappo_checkpoint")
    header = ['epoch', 'train_user_reward', 'train_server_reward', 'test_user_reward', 'test_server_reward',
              'test_mean_AoI', 'lamda']
    # 读取config.yaml中的hyperparameters
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyperparams = config.get('hyperparameters', {})
    # 保存参数信息到train_history.csv开头
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
    # 若train_logfile父目录不存在则创建
    Path(os.path.dirname(train_logfile)).mkdir(parents=True, exist_ok=True)
    with open(train_logfile, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in param_info.items():
            writer.writerow([f"# {k}", v])
        for k, v in hyperparams.items():
            writer.writerow([f"# hyper_{k}", v])
        writer.writerow(header)

    alpha = 0.2  # lamda更新的缩放系数

    # hyperparams在训练循环内使用
    train_batch_size = hyperparams.get('batch_size', 64)
    gamma = hyperparams.get('gamma', 0.99)
    lam = hyperparams.get('lam', 0.95)
    clip_param = hyperparams.get('clip_param', 0.2)
    entropy_coef = hyperparams.get('entropy_coef', 0.01)
    num_epochs = hyperparams.get('num_epochs', 4)

    env.lamda = lamda_init
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
                if event is None: break
                if isinstance(event, tuple) and len(event) == 3 and event[0] == 'user':
                    _, idx, phase = event
                    obs_u = env.get_user_obs(idx)
                    gs = env.get_obs_all()
                    if phase == 'wait':
                        val = agent_u_wait.get_value(gs)
                        w, logp = agent_u_wait.select(obs_u)
                        env.step_user_wait(idx, w)
                        user_trajs_wait[idx].append((obs_u, w, logp, 0, val, gs, 0, 0,
                                                     False))  # [obs, act, logp, reward, val, gs, adv, return, done]
                    elif phase == 'assign':
                        val = agent_u_assign.get_value(gs)
                        act, logp = agent_u_assign.select(obs_u)
                        serv_idx = env.decode_user_assign_action(act)
                        env.step_user_assign(idx, serv_idx)
                        user_trajs_assign[idx].append((obs_u, act, logp, 0, val, gs, 0, 0, False))
                elif isinstance(event, tuple) and len(event) == 3 and event[0] == 'server':
                    _, idx, phase = event
                    obs_s = env.get_server_obs(idx)
                    gs = env.get_obs_all()
                    if phase == 'start':
                        act, logp = agent_s.select(obs_s)
                        val = agent_s.get_value(gs)

                        env.step_server_start(idx, act)

                        if act == 0:
                            server_trajs[idx].append((obs_s, act, logp, 0, val, gs, 0, 0, False))  # 不处理的奖励=0
                        else:
                            server_trajs[idx].append((obs_s, act, logp, 0, val, gs, 0, 0, False))  # 处理的奖励等待回填

                    elif phase == 'end':
                        # 只有end事件才有奖励
                        server_reward, u_rewards = env.step_server_end(idx)
                        from utils import scale_rewards
                        server_reward_norm, u_rewards_norm = scale_rewards(server_reward, u_rewards, 1)

                        # 回填reward
                        server_trajs[idx][-1] = server_trajs[idx][-1][:3] + (server_reward_norm,) + server_trajs[idx][
                                                                                                        -1][4:]

                        for uid, reward in u_rewards_norm.items():
                            for i in reversed(range(len(user_trajs_assign[uid]))):
                                if user_trajs_assign[uid][i][3] == 0:
                                    user_trajs_assign[uid][i] = user_trajs_assign[uid][i][:3] + (reward,) + \
                                                                user_trajs_assign[uid][i][4:]
                                    break

                            for i in reversed(range(len(user_trajs_wait[uid]))):
                                if user_trajs_wait[uid][i][3] == 0:
                                    user_trajs_wait[uid][i] = user_trajs_wait[uid][i][:3] + (reward,) + \
                                                              user_trajs_wait[uid][i][4:]
                                    break
                    # else:
                    #     env.server_need_decision[idx] = False
                    #     env.t = max(env.t, env.server_can_act_time[idx]) + 0.01
                    #     env.server_can_act_time[idx] = env.t

            # buffer中最后一个样本标记为done
            for i in range(env.num_users):
                if user_trajs_wait[i]:
                    user_trajs_wait[i][-1] = user_trajs_wait[i][-1][:-1] + (True,)
                if user_trajs_assign[i]:
                    user_trajs_assign[i][-1] = user_trajs_assign[i][-1][:-1] + (True,)
            for i in range(env.num_servers):
                if server_trajs[i]:
                    server_trajs[i][-1] = server_trajs[i][-1][:-1] + (True,)

            # 计算GAE
            big_wait_buf, big_assign_buf, big_sbuf = [], [], []
            for i in range(env.num_users):
                if user_trajs_wait[i]:
                    traj = compute_gae(user_trajs_wait[i], gamma=gamma, lam=lam)
                    big_wait_buf.extend(traj)
                if user_trajs_assign[i]:
                    traj = compute_gae(user_trajs_assign[i], gamma=gamma, lam=lam)
                    big_assign_buf.extend(traj)
            for i in range(env.num_servers):
                if server_trajs[i]:
                    traj = compute_gae(server_trajs[i], gamma=gamma, lam=lam)
                    big_sbuf.extend(traj)

            agent_u_wait.train(big_wait_buf, clipr=clip_param, epochs=num_epochs, ent_coef=entropy_coef,
                               batch_size=train_batch_size)
            agent_u_assign.train(big_assign_buf, clipr=clip_param, epochs=num_epochs, ent_coef=entropy_coef,
                                 batch_size=train_batch_size)
            agent_s.train(big_sbuf, clipr=clip_param, epochs=num_epochs, ent_coef=entropy_coef,
                          batch_size=train_batch_size)

            # 统计reward
            train_r_user = np.sum([x[3] for x in big_assign_buf])
            train_r_serv = np.sum([x[3] for x in big_sbuf])
            reward_hist.append(train_r_user)

            # 日志记录
            if ep % log_interval == 0 or ep == 1:
                # test_r_u, test_r_s, test_mean_aoi = run_test_log(env, agent_u_wait, agent_u_assign, agent_s, repeat=5)
                from utils import evaluate_aoi
                aoi_per_user, train_mean_aoi = evaluate_aoi(env.completed_tasks, env.t, env.num_users)
                append_csv_row(train_logfile, [ep, train_r_user, train_r_serv, train_mean_aoi, env.lamda],
                               header if ep == 1 else None)
                print(
                    f"E{ep} | TrainU={train_r_user / env.num_users:.2f} TrainS={train_r_serv:.2f} MeanAoI={train_mean_aoi:.4f} Lamda={env.lamda:.4f}")
            if ep % 5 == 0 or ep == 1:
                completed_dir = os.path.join(outdir, "completed_tasks")
                Path(completed_dir).mkdir(parents=True, exist_ok=True)
                save_completed_tasks_to_csv(env.completed_tasks,
                                            os.path.join(completed_dir, f"it_{outer_iter}_ep_{ep}.csv"))

            # 收敛判据：reward_window内最大最小差小于reward_tol
            if len(reward_hist) >= reward_window:
                window = reward_hist[-reward_window:]
                if np.max(window) - np.min(window) < reward_tol:
                    converged = True

        # lamda更新
        # 收集所有用户的completed_tasks，按公式期望更新lamda
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
            print(f"Lamda converged: {env.lamda:.6f} -> {new_lamda:.6f}, training finished.")
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
    ENV = EdgeBatchEnv(num_users=NUM_USERS, num_servers=NUM_SERVERS, max_queue_len=MAX_QUEUE,
                       episode_limit=EPISODE_LIMIT, lamda=LAMDA)

    user_agent_wait = MAPPOAgent(USER_OBS_DIM, 1, GS_DIM, continuous=True, action_low=1.0, action_high=MAX_WAITING_TIME)
    user_agent_assign = MAPPOAgent(USER_OBS_DIM, ENV.user_assign_action_space(), GS_DIM)
    server_agent = MAPPOAgent(SVR_OBS_DIM, ENV.server_action_space(), GS_DIM)

    train_with_log(ENV, user_agent_wait, user_agent_assign, server_agent, lamda_init=10, reward_tol=5, reward_window=5,
                   lamda_tol=0.05,
                   max_outer_iter=20, epochs_per_lamda=50, log_interval=1, outdir="./trainlog/train_2")
    # plot_train_test_curves("./trainlog/train_history.csv")