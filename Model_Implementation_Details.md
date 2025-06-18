# `model.py` 代码实现详解

## 1. 概述

本文件详细阐述 `model.py` 的代码实现细节。该文件使用 PyTorch 库构建了多智能体强化学习（MAPPO）所需的核心组件，包括神经网络模型、智能体类以及优势计算函数。

## 2. 全局设置

文件开头部分负责导入必要的库、加载配置并设置计算设备。

```python
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

# 加载配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
lr = config['hyperparameters']['learning_rate']

# 设置计算设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)
```
*   **库导入**: 导入了 PyTorch 的核心模块 `nn` (用于构建网络层)、`optim` (用于优化器) 以及 `distributions` (用于处理动作概率分布)。
*   **配置加载**: 从 `config.yaml` 中读取超参数，特别是学习率 `lr`。
*   **设备选择**: 通过 `torch.cuda.is_available()` 检查是否存在可用的 GPU。如果存在，`DEVICE` 被设置为 `'cuda'`，否则为 `'cpu'`。所有后续创建的张量和模型都将被显式地移动到这个设备上，以利用 GPU 加速。

## 3. 神经网络实现

### 3.1. `Actor` (离散动作)

这是用于离散动作空间的策略网络。

```python
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        return self.out(self.fc(x))
```
*   **实现方式**:
    *   继承自 `torch.nn.Module`，这是所有 PyTorch 模型的基类。
    *   `__init__`: 构造函数中，使用 `nn.Sequential` 容器来快速、简洁地搭建一个多层感知机（MLP）。它将 `nn.Linear` (全连接层) 和 `nn.ReLU` (激活函数) 按顺序串联起来。最后，一个单独的 `self.out` 线性层输出最终的 logits。
    *   `forward`: 定义了数据的前向传播路径。输入张量 `x` 首先通过 `self.fc` 序列，然后将结果送入 `self.out` 层，最终返回代表每个动作分数的 logits。

### 3.2. `Critic` (价值网络)

用于评估全局状态的价值。

```python
class Critic(nn.Module):
    def __init__(self, global_dim, hidden_dim=64):
        # ...
        self.fc1 = nn.Linear(global_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
```
*   **实现方式**:
    *   与 `Actor` 结构类似，但输入维度是 `global_dim`。
    *   `forward` 方法中，它使用了 `torch.nn.functional.relu` (通常导入为 `F.relu`)，这是 `nn.ReLU` 模块的函数式版本。两者功能相同，但函数式版本在编写自定义或更复杂的 `forward` 逻辑时可能更灵活。

### 3.3. `GaussianActor` (连续动作)

这是为连续动作空间设计的策略网络。

```python
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_low, action_high, hidden_dim=64):
        # ...
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        # ...

    def forward(self, x):
        # ...
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        action_clipped = torch.clamp(action, self.action_low, self.action_high)
        log_prob = dist.log_prob(action)
        return action_clipped.squeeze(-1), log_prob.squeeze(-1), mu.squeeze(-1), std.squeeze(-1)
```
*   **实现方式**:
    *   `self.log_std = nn.Parameter(...)`: 将一个张量包装成 `nn.Parameter`，意味着它会成为模型的可学习参数，在 `model.parameters()` 中可见，并能被优化器更新。
    *   `std = torch.exp(self.log_std)`: 通过对 `log_std` 取指数来计算标准差 `std`。这样做可以保证 `std` 永远是正数，这是标准差的数学要求。
    *   `dist.rsample()`: 这是**重参数化技巧 (Reparameterization Trick)** 的实现。直接从一个分布中采样是不可导的，但 `rsample()` 将随机性分离出来（例如，`sample = mu + std * epsilon`，其中 `epsilon` 从标准正态分布采样），使得梯度可以从 `action` 一路回传到 `mu` 和 `std`，从而能够训练网络。
    *   `dist.log_prob(action)`: PyTorch 的分布对象提供了直接计算给定动作的对数概率的方法，非常方便。

## 4. `MAPPOAgent` 实现

这个类是智能体的控制器，整合了模型并实现了完整的选择动作和训练的逻辑。

### 4.1. `select(self, obs)`

```python
def select(self, obs):
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        if self.continuous:
            action, logp, mu, std = self.actor.sample(obs)
            return float(action.cpu().item()), float(logp.cpu().item())
        else:
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            return int(a.cpu().detach().item()), logp.cpu().detach().item()
```
*   **实现方式**:
    *   `torch.tensor(...).unsqueeze(0)`: 将 NumPy 数组或列表格式的 `obs` 转换为 PyTorch 张量，并使用 `unsqueeze(0)` 在第0维增加一个维度，将其从 `[D_obs]` 变为 `[1, D_obs]`，以符合网络对批处理输入的要求。
    *   `with torch.no_grad()`: 这是一个上下文管理器，它会临时禁用梯度计算。在推理（选择动作）阶段，这样做可以显著减少内存消耗并加速计算。
    *   `.cpu().detach().item()`: 这是一个标准操作序列，用于从一个单元素的 GPU 张量中提取一个 Python 标量值。`.detach()` 将其从计算图中分离，`.cpu()` 将其移至 CPU，`.item()` 提取其值。

### 4.2. `train(self, traj, ...)`

这是最核心的训练方法。

```python
def train(self, traj, ...):
    # 1. 将轨迹数据列表转换为 PyTorch 张量
    obs = torch.tensor(np.array([t[0] for t in traj]), ...)
    # ...

    n = len(traj)
    idlist = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idlist) # 2. 每轮 epoch 开始时打乱数据
        for b in range(0, n, batch_size): # 3. 小批量迭代
            bb = idlist[b:b + batch_size]
            
            # ... 计算新旧策略的 logp 和 ratio ...
            ratio = torch.exp(logps - logps_old[bb])

            # 4. 计算 PPO 裁剪损失
            surr1 = ratio * adv[bb]
            surr2 = torch.clamp(ratio, 1 - clipr, 1 + clipr) * adv[bb]
            actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

            # 5. 计算 Critic 价值损失
            values = self.critic(glo_obs[bb]).squeeze(-1)
            critic_loss = ((values - returns[bb]) ** 2).mean()

            # 6. 梯度更新
            loss = actor_loss + 0.2 * critic_loss
            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            loss.backward()
            self.optim_actor.step()
            self.optim_critic.step()
```
*   **实现方式**:
    1.  **数据转换**: 使用列表推导式和 `np.array` 将 Python 对象列表高效地转换为 NumPy 数组，然后用 `torch.tensor` 一次性将整个数组转换为 GPU 张量，这是非常高效的数据处理方式。
    2.  **数据打乱**: 在每个 `epoch` 开始时，通过 `np.random.shuffle` 打乱索引列表 `idlist`，然后用这个打乱后的索引来取小批量数据。这确保了每个小批量都是随机的，有助于模型训练的稳定性。
    3.  **小批量循环**: `for b in range(0, n, batch_size)` 是一个标准的 Python 循环，用于以 `batch_size` 为步长遍历所有数据。
    4.  **Actor 损失**: `torch.clamp` 函数完美地实现了 PPO 的裁剪操作。`torch.min` 则用于取 `surr1` 和 `surr2` 中较小的一个。最后 `.mean()` 对整个小批量的损失取平均。
    5.  **Critic 损失**: 这是一个标准的均方误差（MSE）损失实现。
    6.  **梯度更新**: `zero_grad()` 清除上一批的旧梯度；`backward()` 根据 `loss` 计算所有可学习参数的梯度；`step()` 应用优化算法（如 Adam）根据梯度更新参数。

## 5. `compute_gae` 实现

这是一个独立的辅助函数，用于计算 GAE 和回报。

```python
def compute_gae(traj, gamma=0.99, lam=0.95):
    # ...
    advs = []
    lastgaelam = 0
    for t in reversed(range(len(traj))):
        non_terminal = 1.0 - float(traj[t][8])  # done
        delta = rew[t] + gamma * v[t + 1] * non_terminal - v[t]
        lastgaelam = delta + gamma * lam * non_terminal * lastgaelam
        advs.insert(0, lastgaelam)
    rets = [a + v[i] for i, a in enumerate(advs)]
    # ...
    return traj
```
*   **实现方式**:
    *   **反向遍历**: `for t in reversed(range(len(traj)))` 从轨迹的最后一步开始向前计算，这是 GAE 算法的自然要求。
    *   **`non_terminal`**: 通过 `1.0 - done_flag` 的方式巧妙地处理了 episode 结束的情况。如果 `done` 为 `True` (1.0)，`non_terminal` 为 0，下一状态的价值 `v[t+1]` 就会被清零，符合 TD 误差的定义。
    *   **`lastgaelam`**: 这是一个累加器，在循环中不断更新，实现了 GAE 的递归计算。
    *   `advs.insert(0, ...)`: 由于是反向遍历，每次计算出的优势值需要插入到列表的**开头**，以保持与原始轨迹相同的时间顺序。