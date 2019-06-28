import os, copy, time
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from envs.yumi.yumi_env_base_new import YumiEnv
from envs.yumi.yumi_goalpropose_env import GoalProposeYumiEnv

class YumiMocapXYZGEnv(YumiEnv):
    """
    Yumi environment in Mujoco. Control type is set to be "mocap"

    :param xml_name: xml file of the robot model and the environment
    :param action_dim: action dimension of the rubot
    """      
    def __init__(self, xml_name='yumi_box.xml', action_dim=4):
        self.action_dim = action_dim
        super(YumiMocapXYZGEnv, self).__init__(xml_name=xml_name, control_type='mocap')

    def set_action_space(self):
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)

    def process_mocap_action(self, action):
        pos_ctrl, gripper_ctrl = action[:3], action[-1]
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([0, 0, 0, 0, 0, 0, 0, gripper_ctrl, gripper_ctrl])
        mocat_action = np.concatenate([pos_ctrl, rot_ctrl])
        return mocat_action, gripper_ctrl


class GoalProposeYumiMocapXYZGEnv(GoalProposeYumiEnv):
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
    def __init__(self, xml_name='yumi_box.xml', action_dim=4, reward_type='dense', goal_dim=3,
                    mocap_high=None, mocap_low=None,
                    mocap_init_pos=None, mocap_init_quat=None):
        self.action_dim = action_dim
        super(GoalProposeYumiMocapXYZGEnv, self).__init__(xml_name=xml_name, control_type='mocap', reward_type=reward_type, 
                                                            goal_dim=goal_dim, mocap_high=mocap_high, mocap_low=mocap_low,
                                                            mocap_init_pos=mocap_init_pos, mocap_init_quat=mocap_init_quat)
        self.initial_state = copy.deepcopy(self.sim.get_state())

    def set_action_space(self):
        gripper_force = 1
        low = -np.ones(self.action_dim)
        low[-1] =  -gripper_force
        high = -low
        
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def process_mocap_action(self, action):
        pos_ctrl, gripper_ctrl = action[:3], action[-1]
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([0, 0, 0, 0, 0, 0, 0, gripper_ctrl, gripper_ctrl])
        mocap_action = np.concatenate([pos_ctrl, rot_ctrl])
        return mocap_action, gripper_ctrl

    def _reset_sim(self):
        self.ep_trajectory = []
        # self.sim.set_state(self.initial_state)
        self.set_state(self.initial_qpos, self.initial_qvel)

        self.selected_goal = self.sample_goal()
        print('selected goal', self.selected_goal)
        self.reset_goal()
        # print('reset mocap')
        # self.data.set_mocap_pos('mocap', self.mocap_init_pos)
        # self.data.set_mocap_quat('mocap', self.mocap_init_quat)
        self.sim.forward()
        self.sim.step()
        # input()
        return True        

    def _reset_sim_ep(self):
        self.ep_trajectory = []
        self.selected_goal = self.sample_goal()
        print('selected goal', self.selected_goal)
        self.reset_goal()
        self.sim.forward()
        return True         