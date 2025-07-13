import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
from EdgeBatchEnv import EdgeBatchEnv

class EdgeBatchEnvWrapper(gym.Env):
    """
    A wrapper for the EdgeBatchEnv to make it compatible with Stable Baselines3.

    This wrapper converts the multi-agent environment into a single-agent environment
    with a unified observation space and a multi-discrete action space.
    """
    def __init__(self, env_config: dict):
        """
        Initializes the wrapper.

        Args:
            env_config (dict): Configuration dictionary for the EdgeBatchEnv.
        """
        super(EdgeBatchEnvWrapper, self).__init__()
        self.env = EdgeBatchEnv(**env_config)

        # Define action space
        action_dims = [self.env.user_action_dim] * self.env.n_users + \
                      [self.env.server_action_dim] * self.env.n_servers
        self.action_space = MultiDiscrete(action_dims)

        # Define observation space (without mean-field information)
        user_obs_dim = len(self.env._get_user_obs(0))
        server_obs_dim = len(self.env._get_server_obs(0))
        obs_dim = self.env.n_users * user_obs_dim + self.env.n_servers * server_obs_dim
        
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _flatten_obs(self, obs: list) -> np.ndarray:
        """
        Flattens the list of observations from the environment into a single NumPy array.

        Args:
            obs (list): The list of lists containing observations for each agent.

        Returns:
            np.ndarray: A flattened NumPy array of the observations.
        """
        flat_obs = [item for sublist in obs for item in sublist]
        return np.array(flat_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial flattened observation.
        """
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._flatten_obs(obs), {}

    def step(self, action: np.ndarray):
        """
        Takes a step in the environment.

        Args:
            action (np.ndarray): The action from the SB3 agent.

        Returns:
            tuple: A tuple containing the flattened next observation, reward, done flag, and info dict.
        """
        action_list = action.tolist()
        next_obs, reward, done, info = self.env.step(action_list)
        flattened_next_obs = self._flatten_obs(next_obs)
        # The 'done' flag in SB3 is now 'terminated' or 'truncated'
        terminated = done
        truncated = False # Assuming our environment doesn't have a truncation condition
        return flattened_next_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Rendering is not supported in this environment.
        """
        pass

    def close(self):
        """
        Closes the environment.
        """
        pass