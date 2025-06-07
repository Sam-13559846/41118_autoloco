import pybullet as p
import time
from stable_baselines3 import PPO
from autoloco_env import AutoLocoEnv

env = AutoLocoEnv(render=True)
model = PPO.load("ppo_autoloco")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    time.sleep(1./1000.)  # increase to slow down playback
    if done:
        obs = env.reset()