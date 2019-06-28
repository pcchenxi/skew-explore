from envs.yumi.yumi_door_env import YumiDoorEnv
from gym import utils, spaces
import numpy as np
import os, copy

class YumiDoorButtonEnv(YumiDoorEnv, utils.EzPickle):
    """
    Mujoco yumi environment for button pressing task.
    The robot needs to open the door first then press the button and close the door.
    The extrinsic reward will be given when the button is pressed and the door is closed.

    :param goal_dim: the dimension of the goal state
    :param reward_type: the type for reward computing. The reward type can be "dense", "sparse", "density" and "rnd" 
    :param mocap_high: the up-limit of the mocap position
    :param mocap_low: the low-limit of the mocap position
    :param mocap_init_pos: the initial position of mocap
    :param mocap_init_quat: the initial orientation of mocap
    """     
    def __init__(self, reward_type='sparse', goal_dim=6, 
                    mocap_high=np.hstack((0.0, 0.13, 0.35)), mocap_low=np.hstack((-0.01, -0.155, 0.19)),
                    mocap_init_pos=np.array([-0.04670722, -0.150812, 0.265042291]), 
                    mocap_init_quat=np.array([-0.70699531, 0.09357381, -0.4721928, 0.51810765])):


        YumiDoorEnv.__init__(self, xml_name='yumi_door_button.xml', reward_type=reward_type, goal_dim=goal_dim,
                            mocap_high=mocap_high, mocap_low=mocap_low, 
                            mocap_init_pos=mocap_init_pos, mocap_init_quat=mocap_init_quat)
        utils.EzPickle.__init__(self)
        self.name = 'yumi_box_pick'
        x_scale, y_scale, z_scale, gripper_scale = 0.015, 0.015, 0.01, 1000
        self.action_scale = np.array([x_scale, y_scale, z_scale, gripper_scale])

        self.initial_qpos = np.array([ 2.08826485e+00,  1.17524374e-01, -2.81825192e+00,  7.20298118e-01,
                                       4.08614131e+00,  1.79107153e-01, -2.93887965e+00,  2.03631476e-06,
                                      -1.09979637e-02, -4.01938280e-06,  0.0])

        self.initial_qvel = np.zeros(11)                                           
        self.reset()
        self.always_render = False       

    def compute_extrinsic_reward(self, achieved_goal):
        task_angle1 = 0
        task_angle2 = 0.02
        task_weight = 10.0
        x1 = achieved_goal[-2] - task_angle1
        x2 = achieved_goal[-1] - task_angle2

        delta_alpha = 0.5
        angle_error1 = 0.1
        angle_error2 = 0.012
        if np.abs(x1) < angle_error1 and np.abs(x2) < angle_error2:
            task_reward1 = np.exp(-(x1/delta_alpha)**2)/(np.abs(delta_alpha)*np.sqrt(np.pi))
            task_reward2 = np.exp(-(x2/delta_alpha)**2)/(np.abs(delta_alpha)*np.sqrt(np.pi))
            task_reward = (task_reward1 + task_reward2)/2
        else:
            task_reward = 0

        return task_reward * task_weight         

    def set_extra_task_params(self):
        """
        Thsi function is used for reading any task-related information from xml fills.(i.e. requires element id in xml file.)
        """
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.end_effector = 'gripper_r_base'
        # self.gripper_joints = [self.sim.model.get_joint_qpos_addr('gripper_r_joint'), self.sim.model.get_joint_qpos_addr('gripper_r_joint_m')]
        self.gripper_joints = [self.sim.model.get_joint_qpos_addr('gripper_r_joint')]
        self.door_joint = [self.sim.model.get_joint_qpos_addr('door_l_joint'), 
                            self.sim.model.get_joint_qpos_addr('door_r_joint')]  
                                  
        xyz_range = self.mocap_high - self.mocap_low

        #finger_range = self.sim.model.jnt_qposadr('gripper_r_joint')
        gripper_r_joint_id =  self.sim.model.joint_name2id('gripper_r_joint')
        left_limit = self.sim.model.jnt_range[gripper_r_joint_id]

        gripper_r_joint_id_m =  self.sim.model.joint_name2id('gripper_r_joint_m')
        right_limit = self.sim.model.jnt_range[gripper_r_joint_id_m]

        # gripper_range = np.array([left_limit[1] - left_limit[0],
        #                 right_limit[1] - right_limit[0]])
        gripper_range = [left_limit[1] - left_limit[0]]

        door_l_joint_id =  self.sim.model.joint_name2id('door_l_joint')
        door_l_limit = self.sim.model.jnt_range[door_l_joint_id]
        door_range = [door_l_limit[1] - door_l_limit[0]]

        door_r_joint_id =  self.sim.model.joint_name2id('door_r_joint')
        door_r_limit = self.sim.model.jnt_range[door_r_joint_id]

        # if self.args.use_auto_scale:
        self.obs_scale = np.ones(self.goal_dim)
        # else:
        # self.obs_scale = np.concatenate(([1,1,1,], np.max(xyz_range)/gripper_range, 
                                        # np.max(xyz_range)/door_range, np.max(xyz_range)/door_range))
        # self.obs_scale = np.array([1,1,1,1,1])

        self.xyz_start = self.mocap_low * self.obs_scale[0:3]
        self.xyz_end = self.mocap_high * self.obs_scale[0:3]
        self.gripper_start = left_limit[0] * self.obs_scale[3]
        self.gripper_end = left_limit[1] * self.obs_scale[3]
        self.door_l_start = door_l_limit[0] * self.obs_scale[4]
        self.door_l_end = door_l_limit[1] * self.obs_scale[4]
        self.door_r_start = door_r_limit[0] * self.obs_scale[5]
        self.door_r_end = door_r_limit[1] * self.obs_scale[5]        