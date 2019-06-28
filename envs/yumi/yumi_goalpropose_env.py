import os
import collections
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from envs.yumi.yumi_env_base_new import YumiEnv

class GoalProposeYumiEnv(YumiEnv):
    """
    Goal proposing Yumi environment in Mujoco.
    Goal states are updated using "set_goals" function.

    :param xml_name: xml file of the robot model and the environment
    :param control_type: the control type of yumi robot, can be 'mocap' or 'controller'
    :param goal_dim: the dimension of the goal state
    :param reward_type: the type for reward computing. The reward type can be "dense", "sparse", "density" and "rnd" 
    :param mocap_high: the up-limit of the mocap position
    :param mocap_low: the low-limit of the mocap position
    :param mocap_init_pos: the initial position of mocap
    :param mocap_init_quat: the initial orientation of mocap
    """    
    def __init__(self, xml_name='yumi_reach.xml', control_type='mocap', goal_dim = 3, 
                    reward_type = 'sparse', mocap_high=None, mocap_low=None,
                    mocap_init_pos=None, mocap_init_quat=None):
        self.goal_dim = goal_dim
        self.goals = (np.random.rand(2, goal_dim) - 0.5) * 2 * 0.05
        self.selected_goal = self.sample_goal()
        self.reward_type = reward_type
        self.end_effector = 'gripper_r_base'

        self.ep_count = 0
        self.ep_length = 250
        self.ep_trajectory = []
        self.use_index = False
        super(GoalProposeYumiEnv, self).__init__(xml_name=xml_name, control_type=control_type, 
                                                mocap_high=mocap_high, mocap_low=mocap_low,
                                                mocap_init_pos=mocap_init_pos, mocap_init_quat=mocap_init_quat)

        self.tra_extrinsic_rewards = []
        self.tra_count = 0
        self.task_evaluation = collections.deque(maxlen=100)
        self.hist_task_evaluation = []

    def set_use_index(self, use_index):
        self.use_index = use_index
        print('set use_index', self.use_index)
        
    def set_observation_space(self):
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.obs_dim = obs['observation'].shape

    def set_goals(self, goals):
        self.goals = goals
        print('update goal proposing list !!', len(self.goals))

    def sample_goal(self):
        selected_goal_index = np.random.randint(len(self.goals))
        return self.goals[selected_goal_index]

    def _get_obs(self):
        obs = self.get_current_obs()
        obs = np.array(obs)

        achieved_goal = obs.copy()
        index_length = 50
        if self.use_index:
            index = [self.ep_count/self.ep_length]
            # index = [self.ep_count%index_length]
            obs = np.concatenate((obs, index))

        final_obs = {
                        'observation': obs.copy(),
                        'achieved_goal': achieved_goal.copy(),
                        'desired_goal': self.selected_goal.copy(),
                    }
        return final_obs

    def _reset_sim(self):
        self.ep_trajectory = []
        # self.sim.set_state(self.initial_state)
        self.set_state(self.initial_qpos, self.initial_qvel)

        # sample goal from the received goal list
        self.selected_goal = self.sample_goal()
        self.reset_goal()
        # self.data.set_mocap_pos('mocap', self.mocap_init_pos)
        # self.data.set_mocap_quat('mocap', self.mocap_init_quat)        
        self.sim.forward()
        return True

    def _reset_sim_ep(self):
        self.ep_trajectory = []
        # sample goal from the received goal list
        self.selected_goal = self.sample_goal()
        self.reset_goal()
        self.sim.forward()
        return True
    
    def reset_goal(self):
        pass

    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            np.clip(self.sim.data.qvel.flat[:7], -10, 10)
        ])

    def get_achieved_goal(self):
        return self.sim.data.get_body_xpos(self.end_effector) #self.get_body_com('gripper_r_finger_r')

    def step(self, action):
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        for _ in range(10):
            self.sim.step()
        obs = self._get_obs()

        if self.always_render:
            self.render()

        self.ep_trajectory.append(obs['observation'].copy())
        reward = self.compute_reward(obs['achieved_goal'].copy(), obs['desired_goal'].copy(), self.ep_trajectory, {})

        task_reward = self.compute_extrinsic_reward(obs['achieved_goal'])
        self.tra_extrinsic_rewards.append(task_reward)

        self.ep_count += 1
        if self.ep_count == self.ep_length:
            done = True 
            self.ep_count = 0
        else:
            done = False

        return obs, reward, done, {}

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.data.set_mocap_pos('mocap', self.mocap_init_pos)
        self.data.set_mocap_quat('mocap', self.mocap_init_quat)   

        if len(self.tra_extrinsic_rewards) > 0:
            for i in range(len(self.tra_extrinsic_rewards) - 10, len(self.tra_extrinsic_rewards), 1):
                if self.tra_extrinsic_rewards[i] > 0:
                    self.task_evaluation.append(1)
                else:
                    self.task_evaluation.append(0)
            mean_task_evaluation = np.mean(self.task_evaluation)
            self.hist_task_evaluation.append(mean_task_evaluation)
            np.save(self.args.save_path + '/extrinsic_reward_' + str(self.tra_count), np.array(self.tra_extrinsic_rewards))
            np.save(self.args.save_path + '/task_evaluation', np.array(self.hist_task_evaluation))

            print(self.tra_extrinsic_rewards)
            print(self.tra_count, mean_task_evaluation)
        self.tra_count += 1
        self.tra_extrinsic_rewards = []

        self.sim.step()
        obs = self._get_obs()
        return obs

    def compute_reward(self, achieved_goal, desired_goal, trajectory, info):
        raise NotImplementedError        