from autoloco_env import AutoLocoEnv
from stable_baselines3 import PPO

env = AutoLocoEnv(render=False)
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100_000)  # adjust as needed
model.save("ppo_autoloco")