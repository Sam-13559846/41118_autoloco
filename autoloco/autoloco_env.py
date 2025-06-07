import pybullet as p
import pybullet_data
import gym
import numpy as np
from gym import spaces
import time

class AutoLocoEnv(gym.Env):
    def __init__(self, render=False):
        super(AutoLocoEnv, self).__init__()
        self.render_mode = render
        self.physics_client = p.connect(p.GUI if self.render_mode else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.time_step = 1. / 240.
        p.setTimeStep(self.time_step)

        self.quadruped_urdf = "laikago/laikago_toes.urdf"
        self.goal_position = np.array([2.0, 0.0, 0.3])  # 2m ahead in x-direction

        # Action space: one control per joint
        self.num_joints = 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

        # Observation space: joint states + robot base position + velocity
        obs_dim = self.num_joints * 2 + 6 + 6 + 3  # joint pos+vel, base pos+ori, base vel, goal pos
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.robot = None
        self.goal_visual = None
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")

        # Spawn goal
        self.goal_visual = p.loadURDF("sphere2.urdf", self.goal_position.tolist(), globalScaling=0.2, useFixedBase=True)

        # Spawn quadruped standing up
        start_pos = [0, 0, 0.55]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF(self.quadruped_urdf, start_pos, start_orientation)

        # Disable default motors
        for i in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(self.robot, i, controlMode=p.VELOCITY_CONTROL, force=0)

        return self._get_obs()

    def step(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot, i, controlMode=p.POSITION_CONTROL, targetPosition=action[i], force=20
            )

        p.stepSimulation()

        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, list(range(self.num_joints)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        base_vel_lin, base_vel_ang = p.getBaseVelocity(self.robot)

        obs = np.array(joint_positions + joint_velocities + list(base_pos) + list(base_ori) +
                       list(base_vel_lin) + self.goal_position.tolist(), dtype=np.float32)
        return obs

    def _compute_reward(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        dist = np.linalg.norm(np.array(base_pos[:2]) - self.goal_position[:2])

        reward = -dist
        done = False

        if dist < 0.5:
            reward += 100
            done = True

        if base_pos[2] < 0.3:  # fell
            reward -= 50
            done = True

        return reward, done

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()