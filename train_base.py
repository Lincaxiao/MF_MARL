# ======================================================================================
# Plan for train_base.py
# ======================================================================================
# 1.  **Imports**:
#     - Import necessary libraries: os, datetime, logging, torch, yaml.
#     - Import Stable Baselines3 components: PPO, SAC, TD3.
#     - Import custom environment and wrapper: EdgeBatchEnv, SB3Wrapper.
#     - Import SB3 utilities like callbacks for evaluation.
#
# 2.  **Configuration Loading**:
#     - Create a function `load_config(path)` to read the `config.yaml` file.
#     - This keeps the main script clean and allows for easy configuration changes.
#
# 3.  **Logger Setup**:
#     - Create a function `setup_logger(log_dir)` similar to `train_mf.py`.
#     - It will set up logging to both a file and the console to track training progress.
#
# 4.  **Main Training Class (`BaselineTrainer`)**:
#     -   **`__init__(self, config, algorithm_name)`**:
#         -   Store config and the chosen algorithm name (e.g., 'ppo').
#         -   Set up the device (CUDA or CPU).
#         -   Create the log directory based on the algorithm and current timestamp.
#         -   Initialize the logger using `setup_logger`.
#         -   Instantiate the base environment `EdgeBatchEnv` using parameters from the config file.
#         -   Wrap the environment with `SB3Wrapper`, passing the base env and algorithm name.
#         -   Log key information like algorithm, observation space, and action space details.
#
#     -   **`_create_model(self)`**:
#         -   A helper method to instantiate the correct SB3 model based on `self.algorithm_name`.
#         -   It will use a dictionary or if/elif/else structure to map the name ('ppo', 'sac', 'td3') to the corresponding class (PPO, SAC, TD3).
#         -   It will fetch hyperparameters from the loaded config file to initialize the model.
#         -   This makes the training script flexible and easy to extend.
#
#     -   **`train(self)`**:
#         -   The main training loop.
#         -   Call `_create_model()` to get the SB3 model instance.
#         -   Log the start of the training.
#         -   Execute `model.learn()`, providing the total number of timesteps from the config.
#         -   Implement a callback (e.g., `EvalCallback`) for periodic evaluation and saving the best model (optional but good practice).
#         -   After training, save the final model to the log directory.
#         -   Log the completion of training.
#
# 5.  **Execution Block (`if __name__ == '__main__'`)**:
#     -   Hardcode the algorithm to be tested, e.g., `ALGORITHM_TO_RUN = 'ppo'`. This makes it easy to switch between 'ppo', 'sac', and 'td3' for different runs.
#     -   Load the configuration from `config.yaml`.
#     -   Instantiate the `BaselineTrainer` with the config and the chosen algorithm.
#     -   Call the `trainer.train()` method to start the process.
# ======================================================================================

import os
import logging
from datetime import datetime
import torch
import csv

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from EdgeBatchEnv import EdgeBatchEnv
from sb3_wrapper import SB3Wrapper

def setup_logger(log_dir):
    """Sets up the logger for training."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logger initialized.")

class StatsCallback(BaseCallback):
    """A custom callback to collect episode statistics for PPO."""
    def __init__(self, trainer, verbose=0):
        super(StatsCallback, self).__init__(verbose)
        self.trainer = trainer

    def _on_step(self) -> bool:
        # `self.locals['infos']` is a list of info dicts from the envs
        info = self.locals['infos'][0]
        self.trainer.current_episode_reward += self.locals['rewards'][0]
        self.trainer.current_episode_aoi += info.get('average_aoi', 0)
        self.trainer.current_episode_steps += 1
        return True

class BaselineTrainer:
    """
    A trainer for running baseline algorithms (PPO, SAC, TD3) using Stable Baselines3.
    """
    def __init__(self, config, algorithm_name: str):
        self.config = config
        self.algorithm_name = algorithm_name.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Setup Logging ---
        log_dir_name = f'{self.algorithm_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.log_dir = os.path.join("logs", log_dir_name)
        setup_logger(self.log_dir)

        # --- Environment Setup ---
        env_params = self.config['env']
        base_env = EdgeBatchEnv(
            n_users=env_params['Num_users'],
            n_servers=env_params['Num_servers'],
            max_batch_size=env_params['Max_batch_size'],
            batch_proc_time=env_params['batch_proc_time']
        )
        
        self.env = SB3Wrapper(base_env, algo=self.algorithm_name)
        
        logging.info(f"Successfully created and wrapped environment for algorithm: {self.algorithm_name.upper()}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Observation Space: {self.env.observation_space}")
        logging.info(f"Action Space: {self.env.action_space}")

        # Vars for logging
        self.current_episode_reward = 0
        self.current_episode_aoi = 0
        self.current_episode_steps = 0

    def _create_model(self):
        """Creates the SB3 model with algorithm-specific hyperparameters."""
        hyperparams = self.config['hyperparameters']
        model_map = {'ppo': PPO, 'sac': SAC, 'td3': TD3}

        if self.algorithm_name not in model_map:
            raise ValueError(f"Algorithm '{self.algorithm_name}' is not supported. Choose from {list(model_map.keys())}")

        model_class = model_map[self.algorithm_name]

        # Base parameters common to all models
        model_params = {
            "policy": "MlpPolicy",
            "env": self.env,
            "verbose": 1,
            "device": self.device,
            "gamma": hyperparams.get('gamma', 0.99),
            "learning_rate": hyperparams.get('learning_rate', 3e-4)
        }

        # Add algorithm-specific parameters
        if self.algorithm_name == 'ppo':
            ppo_params = hyperparams.get('ppo', {})
            model_params.update({
                "batch_size": ppo_params.get('batch_size', 64),
                "n_steps": ppo_params.get('n_steps', 2048),
                "n_epochs": ppo_params.get('n_epochs', 10),
                "clip_range": ppo_params.get('clip_range', 0.2),
                "ent_coef": ppo_params.get('ent_coef', 0.01),
                "gae_lambda": ppo_params.get('gae_lambda', 0.95),
                "policy_kwargs": dict(net_arch=dict(pi= [128, 128], vf=[128, 128]))
            })
        elif self.algorithm_name in ['sac', 'td3']:
            off_policy_params = hyperparams.get(self.algorithm_name, {})
            model_params.update({
                "batch_size": hyperparams.get('batch_size', 256),
                "buffer_size": off_policy_params.get('buffer_size', 10000),
                "learning_starts": off_policy_params.get('learning_starts', 5000),
                "tau": off_policy_params.get('tau', 0.005),
                "policy_kwargs": dict(net_arch=[128, 128])
            })

        model = model_class(**model_params)
        return model

    def train(self):
        """Starts the training process by selecting the appropriate training loop."""
        model = self._create_model()
        logging.info(f"Starting training for {self.algorithm_name.upper()}...")

        if self.algorithm_name == 'ppo':
            self._train_on_policy(model)
        else:
            self._train_off_policy(model)

        model_path = os.path.join(self.log_dir, f"{self.algorithm_name}_final_model.zip")
        model.save(model_path)
        logging.info(f"Training finished. Model saved to {model_path}")

    def _train_on_policy(self, model):
        """Training loop for On-Policy algorithms like PPO."""
        hp = self.config['hyperparameters']
        num_episodes = hp.get('num_episodes', 10)
        episode_length = hp.get('episode_length', 1000)
        
        callback = StatsCallback(self)

        for episode in range(num_episodes):
            # Reset stats before each episode
            self.current_episode_reward = 0
            self.current_episode_aoi = 0
            self.current_episode_steps = 0
            
            # The model will internally call env.reset() when it detects a done signal,
            # or at the beginning of learn if reset_num_timesteps is True.
            # Since our env never returns done=True, we rely on the loop structure.
            # We need to manually reset the underlying env to reset its state.
            model.env.reset()

            model.learn(
                total_timesteps=episode_length,
                reset_num_timesteps=False, # Keep total timesteps consistent across episodes
                callback=callback
            )
            
            self.save_completed_tasks_log(episode)
            
            avg_aoi = self.current_episode_aoi / self.current_episode_steps if self.current_episode_steps > 0 else 0
            completed_tasks = len(self.env.env.completed_tasks_log)
            
            logging.info(f"Episode: {episode + 1}/{num_episodes}, "
                         f"Reward: {self.current_episode_reward:.2f}, Avg AoI: {avg_aoi:.2f}, Completed Tasks: {completed_tasks}")
            
            self.env.env.completed_tasks_log.clear()

    def _train_off_policy(self, model):
        """Training loop for Off-Policy algorithms like SAC and TD3."""
        hp = self.config['hyperparameters']
        num_episodes = hp.get('num_episodes', 10)
        episode_length = hp.get('episode_length', 1000)
        learning_starts = hp.get(self.algorithm_name, {}).get('learning_starts', 100)
        batch_size = hp.get(self.algorithm_name, {}).get('batch_size', 128)

        # SAC/TD3 需要像 PPO 一样在每个回合开始时 reset 环境，
        # 以避免状态在回合间持续累积造成指标不可比。
        for episode in range(num_episodes):
            obs, _ = self.env.reset()  # 每回合重置环境
            episode_reward = 0
            episode_aoi = 0
            episode_steps = 0
            for _ in range(episode_length):
                action, _ = model.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Manually add to replay buffer
                model.replay_buffer.add(obs, next_obs, action, reward, terminated, [info])
                obs = next_obs
                
                episode_reward += reward
                episode_aoi += info.get('average_aoi', 0)
                episode_steps += 1

                if model.num_timesteps > learning_starts:
                    model.train(batch_size=batch_size)

                if terminated or truncated:
                    # This part is likely not reached if env never terminates
                    obs, _ = self.env.reset()
            
            self.save_completed_tasks_log(episode)
            avg_aoi = episode_aoi / episode_steps if episode_steps > 0 else 0
            completed_tasks = len(self.env.env.completed_tasks_log)

            logging.info(f"Episode: {episode + 1}/{num_episodes}, "
                         f"Reward: {episode_reward:.2f}, Avg AoI: {avg_aoi:.2f}, Completed Tasks: {completed_tasks}")
            
            self.env.env.completed_tasks_log.clear()

    def save_completed_tasks_log(self, epoch):
        """Saves the completed tasks log to a CSV file."""
        log_path = self.log_dir
        # Accessing the base env's log
        completed_tasks_log = self.env.env.completed_tasks_log
        
        if not completed_tasks_log:
            # logging.info(f"No tasks completed in episode {epoch+1}. Log file not created.")
            return

        os.makedirs(log_path, exist_ok=True)
        filepath = os.path.join(log_path, f"completed_tasks_log_{epoch+1}.csv")
        headers = ['task_id', 'user_id', 'server_id', 'generation_time', 'completion_time', 'latency']

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for task in completed_tasks_log:
                row = [
                    task.get('task_id'),
                    task.get('user_id'),
                    task.get('server_id'),
                    task.get('generation_time'),
                    task.get('completion_time'),
                    task.get('latency')
                ]
                writer.writerow(row)

if __name__ == '__main__':
    # --- CHOOSE ALGORITHM TO RUN ---
    # Options: 'ppo', 'sac', 'td3'
    ALGORITHM_TO_RUN = 'ppo'
    # --------------------------------

    # --- Training Configuration ---
    config = {
        'env': {
            'Num_users': 28,
            'Num_servers': 12,
            'Max_batch_size': 12,
            'batch_proc_time': {'base': 3, 'per_task': 2}
        },
        'hyperparameters': {
            # Common params
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'num_episodes': 350,
            'episode_length': 3000,
            
            # PPO specific
            'ppo': {
                'batch_size': 64,
                'n_steps': 1000, # Steps to collect before update. Should be <= episode_length
                'n_epochs': 4,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'gae_lambda': 0.95
            },
            
            # SAC/TD3 specific (Off-policy)
            'sac': {
                'batch_size': 128,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'tau': 0.005
            },
            'td3': {
                'batch_size': 128,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'tau': 0.005
            }
        }
    }

    trainer = BaselineTrainer(config=config, algorithm_name=ALGORITHM_TO_RUN)
    trainer.train()