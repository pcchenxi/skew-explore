import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

class GaolDummyVecEnv(DummyVecEnv):
    """
    extend the original DummyVecEnv class with function set_goals for goal proposing
    """
    def __init__(self, env_fns):
        super(GaolDummyVecEnv, self).__init__(env_fns)
        if self.envs[0].unwrapped.name == 'yumi':
            self.xyz_start = self.envs[0].unwrapped.xyz_start
            self.xyz_end = self.envs[0].unwrapped.xyz_end
            self.gripper_start = self.envs[0].unwrapped.gripper_start
            self.gripper_end = self.envs[0].unwrapped.gripper_end
            self.door_start = self.envs[0].unwrapped.door_start
            self.door_end = self.envs[0].unwrapped.door_end
        elif self.envs[0].unwrapped.name == 'yumi_box_pick':
            self.xyz_start = self.envs[0].unwrapped.xyz_start
            self.xyz_end = self.envs[0].unwrapped.xyz_end
            self.gripper_start = self.envs[0].unwrapped.gripper_start
            self.gripper_end = self.envs[0].unwrapped.gripper_end
            self.door_l_start = self.envs[0].unwrapped.door_l_start
            self.door_l_end = self.envs[0].unwrapped.door_l_end
            self.door_r_start = self.envs[0].unwrapped.door_r_start
            self.door_r_end = self.envs[0].unwrapped.door_r_end

    def set_goals(self, goals):
        for env_idx in range(self.num_envs):
            self.envs[env_idx].unwrapped.set_goals(goals)

    def set_density_estimator(self, density_estimator):
        for env_idx in range(self.num_envs):
            self.envs[env_idx].unwrapped.set_density_estimator(density_estimator)        

    def set_reward_type(self, reward_type):
        for env_idx in range(self.num_envs):
            self.envs[env_idx].unwrapped.set_reward_type(reward_type)        

    def get_extrinsic_reward(self, achieved_goals):
        return self.envs[0].unwrapped.get_extrinsic_reward(achieved_goals)

    def update_reward_scale(self, mean, std):
        for env_idx in range(self.num_envs):
            self.envs[env_idx].unwrapped.update_reward_scale(mean, std)

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)

# class GoalSubprocVecEnv(SubprocVecEnv):
#     """
#     extend the original SubprocVecEnv class with function set_goals for goal proposing
#     """
#     def set_goals(self, goals):
#         for env_idx in range(self.num_envs):
#             self.envs[env_idx].set_goals()
#     def set_reward_type(self, reward_type):
#         for env_idx in range(self.num_envs):
#             self.envs[env_idx].unwrapped.set_reward_type(reward_type)              



