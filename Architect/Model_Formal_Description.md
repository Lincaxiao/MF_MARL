# `model.py` 形式化描述

## 1. 概述

`model.py` 文件定义了强化学习智能体的核心组件。它基于 **Actor-Critic** 架构，并实现了 **多智能体近端策略优化 (MAPPO)** 算法。该文件还包括了处理离散和连续动作空间的策略网络，以及用于稳定训练的 **泛化优势估计 (GAE)** 方法。

## 2. 核心组件

### 2.1. Actor 网络 (策略网络)

Actor 网络的功能是根据智能体的局部观察 $o$ 来学习一个策略 $\pi_\theta(a|o)$，该策略定义了在给定观察下采取各个动作 $a$ 的概率分布。

*   **输入**: 局部观察 $o \in \mathbb{R}^{D_{obs}}$，其中 $D_{obs}$ 是观察空间的维度。
*   **输出 (离散动作)**: 一个 logits 向量 $l \in \mathbb{R}^{D_{act}}$，其中 $D_{act}$ 是动作空间的维度。策略（即每个动作的概率）通过对 logits 应用 Softmax 函数得到：
    $$
    \pi_\theta(a|o) = \text{Softmax}(l)_a = \frac{e^{l_a}}{\sum_{k=1}^{D_{act}} e^{l_k}}
    $$
*   **网络结构**: 由参数为 $\theta$ 的多层感知机 (MLP) 实现。

### 2.2. Critic 网络 (价值网络)

Critic 网络的功能是评估给定**全局状态** $s_g$ 的价值 $V_\phi(s_g)$。这个价值函数估计了从状态 $s_g$ 开始，遵循当前策略所能获得的期望折扣回报。

*   **输入**: 全局状态 $s_g \in \mathbb{R}^{D_{global}}$，其中 $D_{global}$ 是全局状态空间的维度。
*   **输出**: 一个标量值，即状态价值 $V_\phi(s_g) \in \mathbb{R}$。
*   **网络结构**: 由参数为 $\phi$ 的多层感知机 (MLP) 实现。
*   **用途**: Critic 的输出用于计算优势函数，以指导 Actor 的更新。这是 MAPPO "中心化训练" 的核心。

### 2.3. GaussianActor (连续动作策略网络)

对于连续动作空间，策略被建模为一个高斯分布。

*   **功能**: `GaussianActor` 根据局部观察 $o$ 输出一个高斯分布的均值 $\mu_\theta(o)$ 和标准差 $\sigma_\theta(o)$。
*   **策略**: 动作从该高斯分布中采样：
    $$
    a \sim \pi_\theta(\cdot|o) = \mathcal{N}(\mu_\theta(o), \sigma_\theta(o)^2)
    $$
*   **网络结构**: 主体是一个 MLP，但有两个输出头，分别用于预测 $\mu$ 和 $\log(\sigma)$。标准差 $\sigma$ 通过对 $\log(\sigma)$ 取指数得到，以确保其为正。

## 3. 训练算法

### 3.1. 泛化优势估计 (GAE)

为了减少策略梯度估计的方差，我们使用 GAE 来计算优势函数 $A(s, a)$。对于在时间步 $t$ 的状态-动作对 $(s_t, a_t)$，其 GAE 优势 $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ 计算如下：

1.  首先计算时序差分误差 (TD-error) $\delta_t$:
    $$
    \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
    $$
    其中 $r_t$ 是奖励，$\gamma$ 是折扣因子。

2.  GAE 优势是所有未来 TD-error 的折扣累加和：
    $$
    \hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
    $$
    其中 $\lambda \in [0, 1]$ 是用于在偏差和方差之间进行权衡的参数。

### 3.2. MAPPO 损失函数

`MAPPOAgent` 的 `train` 方法通过优化一个复合损失函数来更新 Actor 和 Critic 的参数。

*   **Actor (策略) 损失 $L^{\text{CLIP}}(\theta)$**:
    PPO 使用一个裁剪的代理目标函数来限制策略更新的幅度。
    $$
    L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
    $$
    其中：
    *   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的概率比。
    *   $\hat{A}_t$ 是在时间步 $t$ 的优势估计 (来自 GAE)。
    *   $\epsilon$ 是裁剪参数 (例如 0.2)。

*   **Critic (价值) 损失 $L^{\text{VF}}(\phi)$**:
    Critic 的目标是使其预测的价值 $V_\phi(s_t)$ 尽可能接近蒙特卡洛回报（或目标价值） $V_t^{\text{targ}}$。这通常通过最小化均方误差来实现：
    $$
    L^{\text{VF}}(\phi) = \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - V_t^{\text{targ}})^2 \right]
    $$
    其中 $V_t^{\text{targ}} = \hat{A}_t + V_\phi(s_t)$。

*   **熵奖励 $S[\pi_\theta](s_t)$**:
    为了鼓励探索，可以在损失函数中加入一个熵奖励项。
    $$
    S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)
    $$

*   **总损失函数 $L(\theta, \phi)$**:
    最终需要优化的总损失是以上各项的加权和：
    $$
    L(\theta, \phi) = \hat{\mathbb{E}}_t \left[ -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 S[\pi_\theta](s_t) \right]
    $$
    其中 $c_1$ 和 $c_2$ 是权重系数。在代码中，优化的目标是 $-L(\theta, \phi)$。