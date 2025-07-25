import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_avg_aoi(log_file: str, smooth_window: int = 1, max_aoi: float = 30):
    """读取指定日志文件并绘制 Avg AoI 曲线，可选滑动平均平滑。

    参数
    ------
    log_file : str
        training.log 的路径。
    smooth_window : int, default 1
        平滑窗口大小，=1 表示不平滑；>1 使用简单滑动平均。
    """
    if not os.path.isfile(log_file):
        print(f"日志文件不存在: {log_file}")
        return

    pattern = re.compile(r"Episode:\s*(\d+)/(\d+).*Avg AoI:\s*([\-\d\.]+)")
    episodes, avg_aois = [], []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep_idx = int(match.group(1))
                aoi_val = float(match.group(3))
                episodes.append(ep_idx)
                if aoi_val <= max_aoi:
                    avg_aois.append(aoi_val)
                else:
                    avg_aois.append(np.nan)  # 使用 nan 代替，以在图中产生断点

    if not episodes:
        print("未解析到任何 Avg AoI 信息。")
        return

    # -------------------- 平滑处理 --------------------
    plot_eps, plot_aois = episodes, avg_aois
    if smooth_window > 1:
        # 使用 pandas 进行滑动平均，可以更好地处理 NaN 值
        aoi_series = pd.Series(avg_aois)
        smoothed = aoi_series.rolling(window=smooth_window, min_periods=1, center=False).mean()
        plot_aois = smoothed.tolist()
        print(f"---- 使用窗口 {smooth_window} 的滑动平均后，数据点数: {len(plot_aois)} ----")

    # 如果 matplotlib 可用则绘图并保存
    if plt is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(plot_eps, plot_aois, marker='o', linewidth=1.0)
        plt.xlabel('Episode')
        plt.ylabel('Average AoI')
        plt.title(f'Average AoI (≤{max_aoi}) over Episodes')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # --- 保存 PNG ---
        save_dir = os.path.dirname(log_file)
        # 根据是否平滑决定文件名
        suffix = f"_smooth{smooth_window}" if smooth_window > 1 else ""
        png_path = os.path.join(save_dir, f'avg_aoi_curve{suffix}.png')
        plt.savefig(png_path, dpi=300)
        print(f"已保存曲线图到 {png_path}")

        # plt.show()
    else:
        print("未安装 matplotlib")

def visualize_compare_avg_aoi(
    dir_mappo: str,
    dir_masac: str,
    dir_ippo: str,
    dir_mfppo: str,
    smooth_window: int = 1,
    max_aoi: float = 30,
    max_episodes: int = -1,
):
    """在同一张图中比较四种实验 (Global, Global+MF, Local, Local+MF) 的 Avg AoI 曲线。

    参数
    ------
    dir_mappo : str
        不带均场的 Global 版本日志目录。
    dir_masac : str
        Global + Mean-Field 版本日志目录。
    dir_ippo : str
        不带均场的 Local 版本日志目录。
    dir_mfppo : str
        Local + Mean-Field 版本日志目录。
    smooth_window : int, default 1
        滑动平均窗口大小，=1 表示不平滑。
    max_episodes : int, default -1
        要绘制的最大回合数，-1 表示绘制所有回合。
    """
    pattern = re.compile(r"Episode:\s*(\d+)/(\d+).*Avg AoI:\s*([\-\d\.]+)")

    def _read_and_smooth(log_path: str):
        """读取 log 并返回 (episodes, aoi_values)；根据 smooth_window 进行平滑."""
        if not os.path.isfile(log_path):
            print(f"日志文件不存在: {log_path}")
            return [], []

        eps, aois = [], []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    ep_idx = int(m.group(1))
                    if max_episodes != -1 and ep_idx > max_episodes:
                        break  # 如果回合数超过限制，则停止读取
                    
                    aoi_val = float(m.group(3))
                    if aoi_val <= max_aoi:
                        aois.append(aoi_val)
                    else:
                        aois.append(np.nan)  # 使用 nan 代替
                    eps.append(ep_idx)
        if not eps:
            return [], []

        # 平滑
        if smooth_window > 1:
            aoi_series = pd.Series(aois)
            smoothed = aoi_series.rolling(window=smooth_window, min_periods=1, center=False).mean()
            aois = smoothed.tolist()
        return eps, aois

    logs = [
        ("MAPPO", os.path.join(dir_mappo, "training.log")),
        ("MASAC", os.path.join(dir_masac, "training.log")),
        ("IPPO", os.path.join(dir_ippo, "training.log")),
        ("MFPPO", os.path.join(dir_mfppo, "training.log")),
    ]

    all_data = []
    for label, path in logs:
        ep, aoi = _read_and_smooth(path)
        if not ep:
            print(f"跳过 {label}，未找到有效数据。")
            continue
        all_data.append((label, ep, aoi))

    if not all_data:
        print("三个日志均无有效数据，终止绘图。")
        return

    # 绘图
    if plt is None:
        print("未安装 matplotlib")
        return

    plt.figure(figsize=(8, 4))
    for label, ep, aoi in all_data:
        plt.plot(ep, aoi, marker='o', linewidth=1.0, label=label)

    plt.xlabel('Episode')
    plt.ylabel('Average AoI')
    title_suffix = f' (First {max_episodes} Episodes)' if max_episodes != -1 else ''
    plt.title(f'Average AoI Comparison (≤{max_aoi}){title_suffix}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存图像（保存在第一个目录）
    save_dir = dir_mappo
    suffix = f"_smooth{smooth_window}" if smooth_window > 1 else ""
    png_path = os.path.join(save_dir, f'avg_aoi_compare{suffix}.png')
    plt.savefig(png_path, dpi=300)
    print(f"已保存比较曲线图到 {png_path}")
# 每 60 秒调用一次
while True:
    visualize_avg_aoi("logs/Real_mappo_mf_35_5_Global_NoMF_GMM2_5e-05/training.log", 3, 500)
    visualize_avg_aoi("logs/Real_mappo_mf_35_5_Local_NoMF_GMM2_4e-05/training.log", 3, 500)
    # time.sleep(20)
    visualize_compare_avg_aoi(
        "logs/Real_mappo_mf_35_5_Shared_Global_NoMF_GMM2_4e-05/",
        "logs/Real_masac_35_5_Global_NoMF_5e-07/",
        "logs/Real_mappo_mf_35_5_Local_NoMF_GMM2_4e-05/",
        "logs/Real_mappo_mf_35_5_Local_MF_GMM2_5e-05/",
        3,
        150,
        max_episodes=500, # 示例：只绘制前1000个回合
    )
    time.sleep(20)
