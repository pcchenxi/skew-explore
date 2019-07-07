"""
This environment is extended from the point maze environment of "rllab" 
(https://github.com/rll/rllab) to a goal proposing environment.
The point moves along x and y direction
"""
import copy
import collections
import numpy as np
from gym import utils, spaces
from envs.point_maze.maze_env import MazeEnv
from envs.point_maze.point_env import PointEnv
from rllab.envs.base import Step
from sklearn.neighbors.kde import KernelDensity

class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
    ORI_IND = 2

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    MANUAL_COLLISION = True

    def __init__(self):
        super(PointMazeEnv, self).__init__()
        obs = self.get_current_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    def get_current_obs(self):
        return np.array(self.wrapped_env.get_xy())

class GoalProposePointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
    ORI_IND = 2

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    MANUAL_COLLISION = True

    def __init__(self):
        super(GoalProposePointMazeEnv, self).__init__()
        self.name = 'point_maze'
        self.unwrapped = self
        goal_dim = 2
        self.goals = (np.random.rand(100, goal_dim) - 0.5) * 2 * 0.2
        self.selected_goal = self.sample_goal()
        self.reward_type = 'dense'

        self.ep_count = 0.0
        self.ep_length = 250.0
        self.ep_trajectory = []

        self.distance_threshold = 0.1
        self.use_index = False
        # self.set_observation_space()

        self.tra_extrinsic_rewards = []
        self.tra_count = 0
        self.task_evaluation = collections.deque(maxlen=100)
        self.hist_task_evaluation = []

    def set_args_params(self, args):
        self.args = args 
        self.use_index = self.args.use_index
        self.reward_type = self.args.reward_type
        self.kde_goal = KernelDensity(kernel='gaussian', bandwidth=self.args.goal_bandwidth)
        self.kde_tra = KernelDensity(kernel='gaussian', bandwidth=self.args.trajectory_bandwidth)
        # self.set_observation_space()

        self.ep_length = self.args.ep_length
        self.always_render = self.args.render

        print('use index', self.use_index)
        print('reward_type', self.reward_type)
        print('ep_length', self.ep_length)

    @property
    def observation_space(self):
        obs = self._get_obs()
        observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        return observation_space

    @property
    def action_space(self):
        return spaces.Box(np.array([-1, -1]), np.array([1, 1]))

    def step(self, action):
        if self.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            inner_next_obs, inner_rew, done, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            if self._is_in_collision(new_pos):
                self.wrapped_env.set_xy(old_pos)
                done = False
        else:
            inner_next_obs, inner_rew, done, info = self.wrapped_env.step(action)
        obs = self._get_obs()

        self.ep_trajectory.append(obs['observation'])
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], self.ep_trajectory, {})

        task_reward = self.compute_extrinsic_reward(obs['achieved_goal'])
        self.tra_extrinsic_rewards.append(task_reward)

        self.ep_count += 1
        if self.ep_count == self.ep_length:
            done = True 
            self.ep_count = 0
        else:
            done = False

        if self.always_render:
            self.render()
        return obs, reward, done, {}

    def _get_obs(self):
        obs = self.get_current_obs()
        obs = np.array(obs)

        achieved_goal = obs.copy()
        if self.use_index:
            index = [self.ep_count/self.ep_length]
            obs = np.concatenate((obs, index))

        final_obs = {
                        'observation': obs.copy(),
                        'achieved_goal': achieved_goal.copy(),
                        'desired_goal': self.selected_goal.copy(),
                    }
        return final_obs

    def get_current_obs(self):
        return np.array(self.wrapped_env.get_xy())

    def compute_reward(self, achieved_goal, desired_goal, trajectory, info):
        goal_mse = np.linalg.norm((achieved_goal - desired_goal))
        reward = 0
        # sparse and dense reward types are used for goal conditioned policy training, 
        # density reward type is used for goal distribution conditioned policy training
        # assert self.reward_type in  ['sparse', 'dense', 'density'], "reward type must be one of three types. ['sparse', 'dense', 'density']"
        if self.reward_type == 'sparse':
            if goal_mse < self.distance_threshold:
                reward = -1
            else:
                reward = 0
        elif self.reward_type == 'dense':
            reward = -goal_mse

        elif self.reward_type == 'density':
            self.kde_goal.fit([desired_goal])
            if self.use_index:
                self.kde_tra.fit(np.array(trajectory)[:,:-1])
            else:
                self.kde_tra.fit(np.array(trajectory))

            # log_g_density = self.kde_goal.score_samples([desired_goal])
            log_density_from_g = self.kde_goal.score_samples([achieved_goal])
            log_density_from_t = self.kde_tra.score_samples([achieved_goal])

            # if info is None:
            # print(log_density_from_g, log_density_from_t, (log_density_from_g - log_density_from_t), log_g_density)
            reward = (log_density_from_g - log_density_from_t) * 0.01
        else:
            pass

        task_reward = self.compute_extrinsic_reward(achieved_goal)

        return reward + task_reward

    def compute_extrinsic_reward(self, achieved_goal):
        task_target = np.array([-4, 0])
        task_weight = 10.0
        x = np.linalg.norm((achieved_goal - task_target))
        delta_alpha = 0.5
        if np.abs(x) < 0.15:
            task_reward = np.exp(-(x/delta_alpha)**2)/(np.abs(delta_alpha)*np.sqrt(np.pi))
        else:
            task_reward = 0

        return task_reward * task_weight  

    def get_extrinsic_reward(self, achieved_goal_list):
        task_rewards = []
        for achieved_goal in achieved_goal_list:
            task_reward = self.compute_extrinsic_reward(achieved_goal)
            task_rewards.append(task_reward)

        print('achieved_goal_list size', achieved_goal_list.shape, np.min(task_rewards))

        return np.array(task_rewards)

    def reset(self, *args, **kwargs):
        print('in reset')
        self.selected_goal = self.sample_goal()
        print('selected goal', self.selected_goal)
        self.wrapped_env.reset(*args, **kwargs)

        self.ep_trajectory = []
        if len(self.tra_extrinsic_rewards) > 0:
            for i in range(len(self.tra_extrinsic_rewards) - 10, len(self.tra_extrinsic_rewards), 1):
                if self.tra_extrinsic_rewards[i] > 0:
                    self.task_evaluation.append(1)
                else:
                    self.task_evaluation.append(0)
            mean_task_evaluation = np.mean(self.task_evaluation)
            self.hist_task_evaluation.append(mean_task_evaluation)
            # np.save(self.args.save_path + '/extrinsic_reward_' + str(self.tra_count), np.array(self.tra_extrinsic_rewards))
            # np.save(self.args.save_path + '/task_evaluation', np.array(self.hist_task_evaluation))

            # print(self.tra_extrinsic_rewards)
            # print(self.tra_count, mean_task_evaluation)
            
        self.tra_count += 1
        self.tra_extrinsic_rewards = []

        return self._get_obs()

    def reset_ep(self, *args, **kwargs):
        print('in reset ep')
        self.selected_goal = self.sample_goal()
        print('selected goal', self.selected_goal)
        self.ep_trajectory = []
        return self._get_obs()

    def set_goals(self, goals):
        self.goals = goals
        print('update goal proposing list !!', self.goals.shape)

    def sample_goal(self):
        print('in sample goal')
        selected_goal_index = np.random.randint(len(self.goals))
        return self.goals[selected_goal_index]