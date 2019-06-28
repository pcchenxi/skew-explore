"""
This code is modified from the implementation of "stable_baseline" 
(https://github.com/hill-a/stable-baselines) to match the interface 
for goal proposing environment
"""

import numpy as np
from gym import spaces

class HERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.

    :param env: (gym.GoalEnv)
    """

    def __init__(self, env):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        # self.metadata = self.env.metadata
        self.action_space = env.action_space
        self.spaces = list(env.observation_space.spaces.values())

        if self.env.unwrapped.name == 'yumi':
            self.xyz_start = self.env.unwrapped.xyz_start
            self.xyz_end = self.env.unwrapped.xyz_end
            self.gripper_start = self.env.unwrapped.gripper_start
            self.gripper_end = self.env.unwrapped.gripper_end
            self.door_start = self.env.unwrapped.door_start
            self.door_end = self.env.unwrapped.door_end
        elif self.env.unwrapped.name == 'yumi_box_pick':
            self.xyz_start = self.env.unwrapped.xyz_start
            self.xyz_end = self.env.unwrapped.xyz_end
            self.gripper_start = self.env.unwrapped.gripper_start
            self.gripper_end = self.env.unwrapped.gripper_end
            self.door_l_start = self.env.unwrapped.door_l_start
            self.door_l_end = self.env.unwrapped.door_l_end
            self.door_r_start = self.env.unwrapped.door_r_start
            self.door_r_end = self.env.unwrapped.door_r_end

        # TODO: check that all spaces are of the same type
        # (current limitation of the wrapper)
        # TODO: check when dim > 1

        goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
        self.obs_dim = env.observation_space.spaces['observation'].shape[0]
        self.goal_dim = goal_space_shape[0]
        total_dim = self.obs_dim + 2 * self.goal_dim

        if len(goal_space_shape) == 2:
            assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
        else:
            assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            pass

        else:
            raise NotImplementedError()

    @staticmethod
    def convert_dict_to_obs(obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        # return np.concatenate([obs for obs in obs_dict.values()])
        return np.concatenate((obs_dict['observation'], obs_dict['achieved_goal'], obs_dict['desired_goal']))

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (dict<np.ndarray>)
        """
        return {
            'observation': observations[:self.obs_dim],
            'achieved_goal': observations[self.obs_dim:self.obs_dim + self.goal_dim],
            'desired_goal': observations[self.obs_dim + self.goal_dim:],
        }

    def set_density_estimator(self, density_estimator):
        self.env.unwrapped.density_estimator = density_estimator

    def set_goals(self, goals):
        self.env.unwrapped.set_goals(goals)

    def update_reward_scale(self, mean, std):
        self.env.unwrapped.update_reward_scale(mean, std)

    def get_extrinsic_reward(self, achieved_goals):
        return self.env.unwrapped.get_extrinsic_reward(achieved_goals)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.convert_dict_to_obs(obs), reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        return self.convert_dict_to_obs(self.env.reset())
    
    def reset_ep(self):
        return self.convert_dict_to_obs(self.env.reset_ep())

    def compute_reward(self, achieved_goal, desired_goal, trajectory, info):
        return self.env.compute_reward(achieved_goal, desired_goal, trajectory, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
