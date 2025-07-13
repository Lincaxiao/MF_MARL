import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 假设 EdgeBatchEnv 在同一个目录下或已安装
from EdgeBatchEnv import EdgeBatchEnv


class SB3Wrapper(gym.Env):
    """
    A wrapper for the multi-agent EdgeBatchEnv to make it compatible with
    single-agent SB3 algorithms.

    This wrapper flattens the observation and action spaces.
    - The observation space is a single flat vector containing all local observations.
    - The action space is a MultiDiscrete space representing the joint action of all agents.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env: EdgeBatchEnv, algo: str):
        """
        Initializes the SB3 wrapper.

        Args:
            env (EdgeBatchEnv): An already instantiated EdgeBatchEnv object.
            algo (str): The name of the algorithm ('ppo', 'sac', 'td3') which
                        determines the action space type.
        """
        super(SB3Wrapper, self).__init__()
        self.env = env
        self.algo = algo.lower()

        # Define action and observation spaces
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

    def _define_action_space(self):
        """
        Defines the action space based on the algorithm.
        - PPO: Uses the native MultiDiscrete action space.
        - SAC/TD3: Uses a continuous Box space that mimics hierarchical decision-making.
        """
        if self.algo in ['sac', 'td3']:
            # For SAC/TD3, use a Box space that represents hierarchical choices.
            # User action: [alloc_gate] + [server_preferences] -> 1 + n_servers
            # Server action: [process_gate] -> 1
            user_action_dim = 1 + self.env.n_servers
            server_action_dim = 1
            total_dim = self.env.n_users * user_action_dim + self.env.n_servers * server_action_dim
            # Use [-1, 1] as it's common for policies with tanh activation
            return spaces.Box(low=-1, high=1, shape=(total_dim,), dtype=np.float32)
        else:
            # For PPO, use the standard MultiDiscrete space
            user_action_dims = [self.env.n_servers + 1] * self.env.n_users
            server_action_dims = [2] * self.env.n_servers
            return spaces.MultiDiscrete(user_action_dims + server_action_dims)

    def _define_observation_space(self) -> spaces.Box:
        """
        Defines the flattened observation space for all agents.
        - User observation: [h, aoi] (2 features)
        - Server observation: [q_len, b_size, t] (3 features)
        The final observation is a concatenation of all these features.
        """
        user_obs_dim = 2
        server_obs_dim = 3
        total_obs_dim = self.env.n_users * user_obs_dim + self.env.n_servers * server_obs_dim
        
        # Use -inf to inf for simplicity, can be refined if specific bounds are known
        low = -np.inf * np.ones(total_obs_dim, dtype=np.float32)
        high = np.inf * np.ones(total_obs_dim, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, shape=(total_obs_dim,), dtype=np.float32)

    def _flatten_obs(self, obs_list: list) -> np.ndarray:
        """
        Flattens a list of observations from the environment into a single numpy array.
        """
        return np.concatenate([np.array(obs) for obs in obs_list]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.
        """
        super().reset(seed=seed)
        initial_obs_list = self.env.reset()
        flat_obs = self._flatten_obs(initial_obs_list)
        return flat_obs, {}

    def step(self, action: np.ndarray):
        """
        Executes one time step within the environment.
        If the algo is SAC/TD3, it decodes the continuous action vector into
        discrete actions for the environment.

        Args:
            action (np.ndarray): An action from the SB3 agent.

        Returns:
            A tuple (observation, reward, terminated, truncated, info).
        """
        if self.algo in ['sac', 'td3']:
            discrete_actions = []
            current_idx = 0
            # Decode user actions
            user_action_dim = 1 + self.env.n_servers
            for _ in range(self.env.n_users):
                user_action_vec = action[current_idx : current_idx + user_action_dim]
                alloc_gate = user_action_vec[0]
                if alloc_gate > 0:
                    # Decide to allocate: choose server with highest preference
                    server_prefs = user_action_vec[1:]
                    chosen_server_idx = np.argmax(server_prefs)
                    discrete_actions.append(chosen_server_idx + 1) # +1 because 0 is "no-op"
                else:
                    # Decide not to allocate
                    discrete_actions.append(0)
                current_idx += user_action_dim
            
            # Decode server actions
            for i in range(self.env.n_servers):
                server_action_val = action[current_idx + i]
                if server_action_val > 0:
                    discrete_actions.append(1) # Process batch
                else:
                    discrete_actions.append(0) # Wait
            action_list = discrete_actions
        else:
            # For PPO, the action is already discrete
            action_list = action.tolist()

        next_obs_list, reward, done, info = self.env.step(action_list)
        
        flat_next_obs = self._flatten_obs(next_obs_list)
        
        # `done` from custom env corresponds to `terminated` in gymnasium
        terminated = done
        truncated = False # Assuming the env doesn't have a truncation condition like time limit

        return flat_next_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Renders the environment. For now, it prints basic info.
        """
        avg_aoi = self.env.completed_tasks_log[-1]['average_aoi'] if self.env.completed_tasks_log else 'N/A'
        print(f"Time: {self.env.time_step}, Average AoI: {avg_aoi}")

    def close(self):
        """
        Cleans up the environment's resources.
        """
        pass


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    print("--- Running Environment Checker ---")
    
    # 1. Create environment instance
    base_env = EdgeBatchEnv(
        n_users=5,
        n_servers=2,
        batch_proc_time={'base': 2, 'per_task': 1},
        max_batch_size=2
    )
    # Check with PPO (MultiDiscrete)
    print("--- Checking with PPO ---")
    env_ppo = SB3Wrapper(base_env, algo='ppo')
    check_env(env_ppo, warn=True)
    print("\n✅ PPO Wrapper passed the SB3 check!")

    # Check with SAC (Box)
    print("\n--- Checking with SAC ---")
    env_sac = SB3Wrapper(base_env, algo='sac')
    check_env(env_sac, warn=True)
    print("\n✅ SAC Wrapper passed the SB3 check!")

    print("\n--- Running Simple Interaction Test (SAC) ---")
    
    # 3. Optional: Simple interaction loop
    obs, _ = env_sac.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(10):
        action = env_sac.action_space.sample() # Sample continuous action
        obs, reward, terminated, truncated, info = env_sac.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            obs, _ = env_sac.reset()
            
    env_sac.close()
    print("\n✅ Simple interaction test completed.")