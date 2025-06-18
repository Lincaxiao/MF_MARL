# `train.py` 代码实现详解

## 1. 概述

`train.py` 是整个项目的执行入口和总指挥。它负责编排整个训练流程，包括初始化环境和智能体、管理双层训练循环（外层更新 `lamda`，内层训练模型）、驱动事件模拟、收集和处理训练数据、以及记录日志和保存模型。

## 2. 主程序入口 (`if __name__ == "__main__"`)

这是脚本启动时首先执行的部分。

```python
if __name__ == "__main__":
    # 1. 加载配置
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. 从配置中提取环境参数
    NUM_USERS = config['env']['Num_users']
    # ... 其他参数 ...

    # 3. 定义观察和动作空间的维度
    USER_OBS_DIM = 3 + NUM_SERVERS * 2
    SVR_OBS_DIM = 1 + MAX_QUEUE
    GS_DIM = NUM_SERVERS * (2 + MAX_QUEUE) + NUM_USERS * 3

    # 4. 实例化环境和智能体
    ENV = EdgeBatchEnv(...)
    user_agent_wait = MAPPOAgent(..., continuous=True, ...)
    user_agent_assign = MAPPOAgent(...)
    server_agent = MAPPOAgent(...)

    # 5. 调用主训练函数
    train_with_log(ENV, user_agent_wait, user_agent_assign, server_agent, ...)
```
*   **实现方式**:
    1.  **加载配置**: 使用 `yaml.safe_load` 从 `config.yaml` 文件中安全地加载所有参数。
    2.  **参数提取**: 将 `env` 部分的配置项提取到局部变量中，使代码更具可读性。
    3.  **维度计算**: 根据环境参数，硬编码计算出用户观察维度 `USER_OBS_DIM`、服务器观察维度 `SVR_OBS_DIM` 和全局状态维度 `GS_DIM`。这些维度必须与 `env.py` 中 `get_*_obs` 函数返回的向量长度严格匹配。
    4.  **对象实例化**:
        *   创建 `EdgeBatchEnv` 环境实例。
        *   创建三个独立的 `MAPPOAgent` 实例：
            *   `user_agent_wait`: 负责用户“等待”决策。这是一个**连续动作**智能体，因为它需要决定一个具体的等待时长。
            *   `user_agent_assign`: 负责用户“分配”决策。这是一个**离散动作**智能体，它从可用的服务器中选择一个。
            *   `server_agent`: 负责服务器决策，也是一个离散动作智能体。
    5.  **启动训练**: 调用核心的 `train_with_log` 函数，并将所有初始化好的对象和训练超参数传入。

## 3. 主训练函数 `train_with_log`

这个函数包含了整个训练的核心逻辑。

### 3.1. 初始化与日志设置

```python
def train_with_log(env, agent_u_wait, agent_u_assign, agent_s, ...):
    # 创建输出目录
    Path(outdir).mkdir(parents=True, exist_ok=True)
    train_logfile = os.path.join(outdir, "train_history.csv")
    model_path = os.path.join(outdir, "mappo_checkpoint")
    
    # ... 提取超参数 ...

    # 将所有参数写入日志文件头部
    with open(train_logfile, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in param_info.items():
            writer.writerow([f"# {k}", v])
        # ...
        writer.writerow(header)
```
*   **实现方式**:
    *   **目录创建**: 使用 `pathlib.Path` 的 `mkdir` 方法创建输出目录。`parents=True` 确保可以创建多级目录，`exist_ok=True` 确保如果目录已存在则不会报错。
    *   **日志文件初始化**: 以写入模式 (`'w'`) 打开日志文件。在文件开头，将本次训练的所有相关参数（环境参数、训练超参数）以 `#` 注释行的形式写入。这样做极大地增强了实验的可复现性，因为任何人都可以通过查看日志文件来了解实验的完整配置。

### 3.2. 双层训练循环

代码采用了一个双层循环结构。

```python
    env.lamda = lamda_init
    outer_iter = 0
    while True: # 外层循环：更新 lamda
        outer_iter += 1
        # ...
        
        ep = 0
        while not converged and ep < epochs_per_lamda: # 内层循环：训练模型
            ep += 1
            # ... 模拟与训练 ...

        # ... lamda 更新逻辑 ...
        if abs(new_lamda - env.lamda) < lamda_tol or outer_iter >= max_outer_iter:
            break # 结束外层循环
        env.lamda = alpha * new_lamda + (1 - alpha) * env.lamda
```
*   **外层循环**: 一个无限循环 `while True`，负责迭代地调整 `lamda` 参数。它只有在 `lamda` 收敛或达到最大迭代次数时才会通过 `break` 退出。
*   **内层循环**: 在一个固定的 `lamda` 值下，训练模型 `epochs_per_lamda` 个 epoch，或者直到满足某个收敛条件 `converged`。

### 3.3. 事件驱动的 Episode 模拟

这是内层循环的核心，负责智能体与环境的交互。

```python
            _ = env.reset()
            # 为每个智能体创建空的轨迹列表
            user_trajs_wait = [[] for _ in range(env.num_users)]
            # ...

            while not env.is_done():
                event = env.get_next_event()
                if event is None: break

                if event[0] == 'user':
                    _, idx, phase = event
                    obs_u = env.get_user_obs(idx)
                    gs = env.get_obs_all()
                    if phase == 'wait':
                        val = agent_u_wait.get_value(gs)
                        w, logp = agent_u_wait.select(obs_u)
                        env.step_user_wait(idx, w)
                        # 记录轨迹
                        user_trajs_wait[idx].append((obs_u, w, logp, 0, val, gs, 0, 0, False))
                    # ... 'assign' 阶段类似 ...
                
                elif event[0] == 'server':
                    # ... 'start' 阶段类似 ...
                    elif phase == 'end':
                        # 关键：奖励产生和回填
                        server_reward, u_rewards = env.step_server_end(idx)
                        # ...
                        # 回填服务器奖励
                        server_trajs[idx][-1] = server_trajs[idx][-1][:3] + (server_reward_norm,) + ...
                        # 回填用户奖励
                        for uid, reward in u_rewards_norm.items():
                            # ...
```
*   **实现方式**:
    1.  **重置与初始化**: 每个 epoch 开始时，调用 `env.reset()`，并为每个智能体创建空的轨迹列表 `trajs`。
    2.  **事件循环**: `while not env.is_done()` 持续运行直到 episode 结束。在循环中，`env.get_next_event()` 是驱动力，它返回时间上最早发生的事件。
    3.  **事件处理**:
        *   根据 `event` 的类型（`'user'` 或 `'server'`）和阶段（`'wait'`, `'assign'`, `'start'`, `'end'`），调用对应的智能体进行决策。
        *   获取**局部观察** (`obs_u`, `obs_s`) 和**全局状态** (`gs`)。
        *   调用 `agent.select(obs)` 获取动作，调用 `agent.get_value(gs)` 获取价值评估。
        *   将动作应用到环境中 (`env.step_*`)。
        *   将这次交互的完整信息 `(obs, act, logp, reward, val, gs, adv, return, done)` 作为一个元组存入对应的轨迹列表。注意，此时 `reward` 和其他一些字段暂时用 0 占位。
    4.  **奖励回填 (Credit Assignment)**:
        *   这是事件驱动模型中的一个关键且巧妙的设计。奖励只在服务器 `'end'` 事件中产生。
        *   当奖励产生后，代码会找到触发这次奖励的服务器动作和用户动作所对应的轨迹记录（通常是列表中的最后一个元素 `[-1]`），然后用真实的奖励值替换掉之前占位的 0。
        *   这种**延迟回填**机制确保了动作和其导致的未来奖励能够被正确地关联起来。

### 3.4. GAE 计算与模型训练

在一次完整的 episode 模拟结束后，进行数据处理和模型更新。

```python
            # ... 标记轨迹最后一步为 done ...

            # 计算 GAE
            big_wait_buf, big_assign_buf, big_sbuf = [], [], []
            for i in range(env.num_users):
                if user_trajs_wait[i]:
                    traj = compute_gae(user_trajs_wait[i], ...)
                    big_wait_buf.extend(traj)
                # ...
            
            # 训练所有智能体
            agent_u_wait.train(big_wait_buf, ...)
            agent_u_assign.train(big_assign_buf, ...)
            agent_s.train(big_sbuf, ...)
```
*   **实现方式**:
    1.  **合并轨迹**: 将每个智能体单独的轨迹列表，通过 `extend` 方法合并成一个大的 buffer (`big_*_buf`)。
    2.  **计算 GAE**: 对每个智能体的轨迹调用 `compute_gae` 函数，该函数会计算优势和回报，并更新 buffer 中的相应字段。
    3.  **训练**: 将处理好的、包含完整信息的 buffer 传递给每个智能体的 `.train()` 方法，执行 PPO 算法更新。

### 3.5. Lamda 更新逻辑

在内层循环结束后，执行外层循环的 `lamda` 更新。

```python
        numer, denom, count = 0.0, 0.0, 0
        for user_tasks in env.completed_tasks:
            for rec in user_tasks:
                # ...
                numer += 0.5 * ((q_k - t_km1) ** 2 - (q_k - t_k) ** 2)
                denom += (q_km1 - t_km1 + w)
        
        new_lamda = numer / denom if denom != 0 else env.lamda
        # ...
        env.lamda = alpha * new_lamda + (1 - alpha) * env.lamda
```
*   **实现方式**:
    *   **数据收集**: 遍历 `env.completed_tasks` 中记录的所有已完成任务。
    *   **公式计算**: 根据奖励函数的形式，分别累加计算 `new_lamda` 的分子（与 AoI 面积相关）和分母（与任务生命周期相关）。
    *   **平滑更新**: 使用一个平滑因子 `alpha` 来更新 `env.lamda`，即 `new_lamda = alpha * calculated_lamda + (1 - alpha) * old_lamda`。这可以防止 `lamda` 值因单次内层循环的随机性而剧烈波动，使整个寻优过程更稳定。