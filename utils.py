import math
import os, torch, csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_zeta(batch_size):
    a = 0.5
    b = 2
    return int(a * batch_size + b)


def log_zeta(batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return int(math.log(batch_size, base=math.e))


def save_model(agent, path):
    """根据 agent 类型保存参数。默认保存 actor；
    若有 critic 保存 critic；若有 q1/q2 保存这两支 critic。"""
    torch.save(agent.actor.state_dict(), path + "_actor.pth")
    if hasattr(agent, 'critic'):
        torch.save(agent.critic.state_dict(), path + "_critic.pth")
    elif hasattr(agent, 'q1') and hasattr(agent, 'q2'):
        torch.save(agent.q1.state_dict(), path + "_q1.pth")
        torch.save(agent.q2.state_dict(), path + "_q2.pth")


def load_model(agent, path):
    agent.actor.load_state_dict(torch.load(path + "_actor.pth", map_location="cpu"))
    if hasattr(agent, 'critic'):
        agent.critic.load_state_dict(torch.load(path + "_critic.pth", map_location="cpu"))
    elif hasattr(agent, 'q1') and hasattr(agent, 'q2'):
        agent.q1.load_state_dict(torch.load(path + "_q1.pth", map_location="cpu"))
        agent.q2.load_state_dict(torch.load(path + "_q2.pth", map_location="cpu"))


def append_csv_row(filename, row, header=None):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if (header is not None) and not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def evaluate_aoi(completed_tasks, total_time, n_users):
    # 统计每个用户的AoI面积
    aoi_per_user, aoi_all = [], 0.
    for tasks in completed_tasks:
        sum_aoi = 0.
        for t_km1, q_km1, t_k, q_k, w, reward, batch_size, _ in tasks:
            sum_aoi += 0.5 * ((q_k - t_km1) ** 2 - (q_k - t_k) ** 2)
        mean_aoi = sum_aoi / total_time if total_time > 0 else 0.
        aoi_per_user.append(mean_aoi)
        aoi_all += mean_aoi
    return aoi_per_user, aoi_all / n_users if n_users > 0 else 0.


def plot_train_test_curves(logfile, save_path='train_test_curves.png'):
    data = pd.read_csv(logfile)
    plt.figure(figsize=(14, 6))
    plt.subplot(131)
    plt.plot(data['epoch'], data['train_user_reward'], label="Train UserReward")
    plt.plot(data['epoch'], data['train_server_reward'], label="Train ServerReward")
    plt.legend();
    plt.title("Train Rewards")
    plt.subplot(132)
    plt.plot(data['epoch'], data['test_user_reward'], label="Test UserReward")
    plt.plot(data['epoch'], data['test_server_reward'], label="Test ServerReward")
    plt.legend();
    plt.title("Test Rewards")
    plt.subplot(133)
    plt.plot(data['epoch'], data['test_mean_AoI'], label="Test Mean AoI")
    plt.legend();
    plt.title("Test MeanAoI")
    plt.tight_layout();
    plt.savefig(save_path)


def plot_train_log_with_lamda(logfile, save_path='train_test_curves.png'):
    data = pd.read_csv(logfile)
    # 构造累计step列和lambda分段
    step = []
    cur = 0
    prev_lamda = None
    seg_idx = []
    seg = 0
    for i, row in data.iterrows():
        if prev_lamda is not None and abs(row['lamda'] - prev_lamda) > 1e-8:
            cur += 1
            seg += 1
        step.append(cur + row['epoch'])
        seg_idx.append(seg)
        prev_lamda = row['lamda']
    data['step'] = step
    data['seg'] = seg_idx
    # 颜色列表
    color_list = plt.cm.get_cmap('tab10', data['seg'].max()+1)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for s in range(data['seg'].min(), data['seg'].max()+1):
        seg_data = data[data['seg']==s]
        color = color_list(s)
        axs[0].plot(seg_data['step'], seg_data['train_user_reward'], label=f"Train UserReward λ{s}", color=color)
        axs[0].plot(seg_data['step'], seg_data['train_server_reward'], label=f"Train ServerReward λ{s}", linestyle='--', color=color)
        axs[1].plot(seg_data['step'], seg_data['test_user_reward'], label=f"Test UserReward λ{s}", color=color)
        axs[1].plot(seg_data['step'], seg_data['test_server_reward'], label=f"Test ServerReward λ{s}", linestyle='--', color=color)
        axs[2].plot(seg_data['step'], seg_data['test_mean_AoI'], label=f"Test Mean AoI λ{s}", color=color)
        if 'lamda' in seg_data.columns:
            axs[3].plot(seg_data['step'], seg_data['lamda'], label=f"Lamda λ{s}", color=color)
    axs[0].set_title("Train Rewards")
    axs[0].legend()
    axs[1].set_title("Test Rewards")
    axs[1].legend()
    axs[2].set_title("Test MeanAoI")
    axs[2].legend()
    if 'lamda' in data.columns:
        axs[3].set_title("Lamda vs Step")
        axs[3].legend()
    else:
        axs[3].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path)


def run_test_log(env, agent_u, agent_s, repeat=10):
    sum_ur, sum_sr, all_aoi, totT = 0., 0., 0., 0.
    n = env.num_users
    for i in range(repeat):
        _ = env.reset()
        episode_ur, episode_sr = 0., 0.
        while not env.is_done():
            event = env.get_next_event()
            if event is None: break
            who, idx = event
            gs = env.get_obs_all()
            if who=='user':
                obs_u = env.get_user_obs(idx)
                act, _ = agent_u.select(obs_u)
                wait_idx, serv_idx = env.decode_user_action(act)
                env.step_user(idx, wait_idx, serv_idx)
            elif who=='server':
                obs_s = env.get_server_obs(idx)
                act, _ = agent_s.select(obs_s)
                qlen = len(env.server_queues[idx])
                do_batch, batch_size = env.decode_server_action(act, qlen)
                sr, u_rewards = env.step_server(idx, do_batch, batch_size)
                episode_sr += sr
                for r in u_rewards.values():
                    episode_ur += r
        # AoI统计
        aoi_per_user, mean_aoi = evaluate_aoi(env.completed_tasks, env.t, n)
        save_completed_tasks_to_csv(env.completed_tasks, f"./trainlog/completed_tasks/ep_{i}.csv")
        sum_ur += episode_ur
        sum_sr += episode_sr
        all_aoi += mean_aoi
        totT += env.t
    return sum_ur/repeat, sum_sr/repeat, all_aoi/repeat


def save_completed_tasks_to_csv(completed_tasks, filename):
    """
    :param completed_tasks: env.completed_tasks 原始数据
    :param filename: 输出文件名
    """
    # 字段名
    fieldnames = ['user_id', 't_km1', 'q_km1', 't_k', 'q_k', 'w', 'reward', 'batch_size', 'batch_idx']
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for user_id, tasks in enumerate(completed_tasks):
            for t in tasks:
                row = [user_id] + t
                writer.writerow(row)


def get_batch_reward_coef(t_arrival, alpha=2.0):
    """
    给定到达时间数组t_arrival，返回归一化reward系数数组，范围[1, alpha]，越晚到达系数越大。
    """
    t_arrival = np.array(t_arrival)
    t_min, t_max = t_arrival.min(), t_arrival.max()
    if t_max > t_min:
        coef_arr = 1.0 + (alpha - 1.0) * (t_arrival - t_min) / (t_max - t_min)
    else:
        coef_arr = np.ones_like(t_arrival)
    return coef_arr


def normalize_rewards(server_reward, u_rewards):
    """
    对 server_reward (标量) 和 u_rewards (dict) 做标准化归一化。
    server_reward: float
    u_rewards: dict {uid: reward}
    返回: server_reward_norm, u_rewards_norm
    """
    # u_rewards 是字典，对本 batch 内 reward 做标准化
    u_r_list = list(u_rewards.values())
    if len(u_r_list) > 1:
        mean_u = np.mean(u_r_list)
        std_u = np.std(u_r_list) + 1e-8
        u_rewards_norm = {k: (v - mean_u) / std_u for k, v in u_rewards.items()}
    else:
        u_rewards_norm = {k: 0.0 for k in u_rewards}
    # server_reward_norm 改为标准化后的 u_rewards_norm 的平均值
    server_reward_norm = np.mean(list(u_rewards_norm.values())) if u_rewards_norm else 0.0
    return server_reward_norm, u_rewards_norm


def scale_rewards(server_reward, u_rewards, scale=0.01):
    u_rewards_scaled = {k: v * scale for k, v in u_rewards.items()}
    server_reward_scaled = server_reward * scale
    return server_reward_scaled, u_rewards_scaled


def plot_aoi_user_reward_vs_epoch(train_history_csv, save_path='aoi_user_reward_vs_epoch.png'):
    """
    画出train_history.csv中mean-AoI和user_reward随epoch变化的曲线。
    忽略多轮次间的epoch断点，假设epoch连续递增。
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    # 读取csv，跳过以#开头的参数行
    df = pd.read_csv(train_history_csv, comment='#')
    # 构造连续的全局epoch
    df = df.reset_index(drop=True)
    df['global_epoch'] = df.index + 1
    # 画图
    plt.figure(figsize=(10,5))
    plt.plot(df['global_epoch'], df['train_user_reward'], label='Train User Reward')
    if 'test_mean_AoI' in df.columns:
        plt.plot(df['global_epoch'], df['test_mean_AoI'], label='Test Mean AoI')
    plt.xlabel('Epoch (global)')
    plt.ylabel('Value')
    plt.title('User Reward and Mean AoI vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def plot_aoi_and_user_reward_vs_epoch(train_history_csv, save_path):
    """
    分别画出train_history.csv中mean-AoI、user_reward和lamda随epoch变化的曲线。
    忽略多轮次间的epoch断点，假设epoch连续递增。
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    # 读取csv，跳过以#开头的参数行
    df = pd.read_csv(train_history_csv, comment='#')
    # 构造连续的全局epoch
    df = df.reset_index(drop=True)
    df['global_epoch'] = df.index + 1
    save_path_aoi = save_path + 'aoi_vs_epoch.png'
    save_path_user_reward = save_path + 'user_reward_vs_epoch.png'
    save_path_lamda = save_path + 'lamda_vs_epoch.png'
    # 画user_reward
    plt.figure(figsize=(8,5))
    plt.plot(df['global_epoch'], df['train_user_reward'], label='Train User Reward')
    plt.xlabel('Epoch (global)')
    plt.ylabel('User Reward')
    plt.title('User Reward vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_user_reward)
    plt.close()
    # 画mean-AoI
    if 'test_mean_AoI' in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df['global_epoch'], df['test_mean_AoI'], label='Test Mean AoI', color='orange')
        plt.xlabel('Epoch (global)')
        plt.ylabel('Mean AoI')
        plt.title('Mean AoI vs. Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_aoi)
        plt.close()
    # 画lamda
    if 'lamda' in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df['global_epoch'], df['lamda'], label='Lamda', color='green')
        plt.xlabel('Epoch (global)')
        plt.ylabel('Lamda')
        plt.title('Lamda vs. Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_lamda)
        plt.close()


def plot_aoi_and_user_reward_vs_epoch_smooth(train_history_csv, save_path, window=11):
    """
    分别画出train_history.csv中mean-AoI、user_reward和lamda随epoch变化的平滑曲线。
    忽略多轮次间的epoch断点，假设epoch连续递增。
    window: 平滑窗口大小（滑动平均）
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    # 读取csv，跳过以#开头的参数行
    df = pd.read_csv(train_history_csv, comment='#')
    # 构造连续的全局epoch
    df = df.reset_index(drop=True)
    df['global_epoch'] = df.index + 1
    save_path_aoi = save_path + 'aoi_vs_epoch_smooth.png'
    save_path_user_reward = save_path + 'user_reward_vs_epoch_smooth.png'
    save_path_lamda = save_path + 'lamda_vs_epoch_smooth.png'
    # 平滑函数
    def smooth(y, window):
        if window < 2:
            return y
        return pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
    # 画user_reward
    plt.figure(figsize=(8,5))
    plt.plot(df['global_epoch'], smooth(df['train_user_reward'], window), label='Train User Reward (Smooth)')
    plt.xlabel('Epoch (global)')
    plt.ylabel('User Reward')
    plt.title('User Reward vs. Epoch (Smooth)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_user_reward)
    plt.close()
    # 画mean-AoI
    if 'test_mean_AoI' in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df['global_epoch'], smooth(df['test_mean_AoI'], window), label='Test Mean AoI (Smooth)', color='orange')
        plt.xlabel('Epoch (global)')
        plt.ylabel('Mean AoI')
        plt.title('Mean AoI vs. Epoch (Smooth)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_aoi)
        plt.close()
    # 画lamda
    if 'lamda' in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df['global_epoch'], smooth(df['lamda'], window), label='Lamda (Smooth)', color='green')
        plt.xlabel('Epoch (global)')
        plt.ylabel('Lamda')
        plt.title('Lamda vs. Epoch (Smooth)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_lamda)
        plt.close()


if __name__ == "__main__":
    # plot_train_test_curves("trainlog/train_2/train_history.csv", save_path='trainlog/train_2/train_test_curves.png')
    # plot_train_log_with_lamda("trainlog/train_2/train_history.csv", save_path='trainlog/train_2/train_test_curves.png')
    # plot_aoi_and_user_reward_vs_epoch_smooth("trainlog/train_2/train_history.csv", save_path='trainlog/train_2/')
    # plot_aoi_and_user_reward_vs_epoch_smooth("trainlog/train_sac/train_history.csv", save_path='trainlog/train_sac/')
    plot_aoi_and_user_reward_vs_epoch_smooth("trainlog/train_td3/train_history.csv", save_path='trainlog/train_td3/')