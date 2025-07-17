import os
import re
import time
import matplotlib.pyplot as plt 
def visualize_avg_aoi(log_file: str, smooth_window: int = 1):
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
                avg_aois.append(aoi_val)

    if not episodes:
        print("未解析到任何 Avg AoI 信息。")
        return

    # 打印简单统计（取原始数据，不使用平滑值）
    # print("---- 平均 AoI 变化 (原始) ----")
    # for ep, aoi in zip(episodes, avg_aois):
    #     print(f"Episode {ep}: Avg AoI {aoi:.2f}")

    # -------------------- 平滑处理 --------------------
    plot_eps, plot_aois = episodes, avg_aois
    if smooth_window > 1:
        if smooth_window > len(avg_aois):
            print(f"smooth_window({smooth_window}) 大于数据长度，调整为 {len(avg_aois)}")
            smooth_window = len(avg_aois)
        import numpy as np
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(avg_aois, kernel, mode='valid')
        plot_eps = episodes[smooth_window - 1:]
        plot_aois = smoothed.tolist()
        print(f"---- 使用窗口 {smooth_window} 的滑动平均后，数据点数: {len(plot_aois)} ----")

    # 如果 matplotlib 可用则绘图并保存
    if plt is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(plot_eps, plot_aois, marker='o', linewidth=1.0)
        plt.xlabel('Episode')
        plt.ylabel('Average AoI')
        plt.title('Average AoI over Episodes')
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

def visualize_compare_avg_aoi(dir_global: str, dir_global_mf: str, dir_local_mf: str, smooth_window: int = 1):
    """在同一张图中比较三种实验 (Global, Global+MF, Local+MF) 的 Avg AoI 曲线。

    参数
    ------
    dir_global : str
        不带均场的 Global 版本日志目录。
    dir_global_mf : str
        Global + Mean-Field 版本日志目录。
    dir_local_mf : str
        Local + Mean-Field 版本日志目录。
    smooth_window : int, default 1
        滑动平均窗口大小，=1 表示不平滑。
    """
    pattern = re.compile(r"Episode:\s*(\d+)/(\d+).*Avg AoI:\s*([\-\d\.]+)")

    def _read_and_smooth(log_path: str):
        """读取 log 并返回 (episodes, aoi_values)；根据 smooth_window 进行平滑。"""
        if not os.path.isfile(log_path):
            print(f"日志文件不存在: {log_path}")
            return [], []

        eps, aois = [], []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    eps.append(int(m.group(1)))
                    aois.append(float(m.group(3)))
        if not eps:
            return [], []

        # 平滑
        if smooth_window > 1:
            import numpy as np
            if smooth_window > len(aois):
                print(f"smooth_window({smooth_window}) 大于数据长度，调整为 {len(aois)}")
                sw = len(aois)
            else:
                sw = smooth_window
            kernel = np.ones(sw) / sw
            smoothed = np.convolve(aois, kernel, mode='valid')
            eps = eps[sw - 1:]
            aois = smoothed.tolist()
        return eps, aois

    logs = [
        ("Global", os.path.join(dir_global, "training.log")),
        ("Global+MF", os.path.join(dir_global_mf, "training.log")),
        ("Local+MF", os.path.join(dir_local_mf, "training.log")),
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
    plt.title('Average AoI Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存图像（保存在第一个目录）
    save_dir = dir_global
    suffix = f"_smooth{smooth_window}" if smooth_window > 1 else ""
    png_path = os.path.join(save_dir, f'avg_aoi_compare{suffix}.png')
    plt.savefig(png_path, dpi=300)
    print(f"已保存比较曲线图到 {png_path}")
# 每 60 秒调用一次
while True:
    visualize_avg_aoi("logs/mappo_mf_21_9_Global_NoMF_1e-05/training.log", 3)
    visualize_avg_aoi("logs/mappo_mf_21_9_Global_NoMF_4e-05/training.log",3)
    time.sleep(60)
# visualize_compare_avg_aoi("logs/mappo_20250715_114328/",
#                           "logs/mappo_mf_20250715_121311/",
#                           "logs/mappo_mf_20250715_125350/",
#                           1)
