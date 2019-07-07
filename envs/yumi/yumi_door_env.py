import os, copy
import numpy as np
from gym import utils, spaces, error
from gym.envs.mujoco import mujoco_env
import time, math
from sklearn.neighbors.kde import KernelDensity

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))
from envs.yumi.yumi_env_mocap import YumiMocapXYZGEnv, GoalProposeYumiMocapXYZGEnv

class YumiDoorEnv(GoalProposeYumiMocapXYZGEnv, utils.EzPickle):
    """
    Mujoco yumi environment for door opening task.
    The robot needs to fully open the door.
    The extrinsic reward will be given when the door is fully openned.

    :param xml_name: xml file of the robot model and the environment
    :param goal_dim: the dimension of the goal state
    :param reward_type: the type for reward computing. The reward type can be "dense", "sparse", "density" and "rnd" 
    :param mocap_high: the up-limit of the mocap position
    :param mocap_low: the low-limit of the mocap position
    :param mocap_init_pos: the initial position of mocap
    :param mocap_init_quat: the initial orientation of mocap
    """      
    def __init__(self, reward_type='sparse', goal_dim=5, xml_name='yumi_door.xml', 
                    mocap_high=np.hstack((0.0, 0.03, 0.3)), mocap_low=np.hstack((-0.1, -0.08, 0.2)),
                    mocap_init_pos=None, mocap_init_quat=None):

        GoalProposeYumiMocapXYZGEnv.__init__(self, xml_name=xml_name, reward_type=reward_type, 
                                            goal_dim=goal_dim, mocap_high=mocap_high, mocap_low=mocap_low,
                                            mocap_init_pos=mocap_init_pos, mocap_init_quat=mocap_init_quat)
        utils.EzPickle.__init__(self)
        x_scale, y_scale, z_scale, gripper_scale = 0.005, 0.005, 0.005, 3000
        self.action_scale = np.array([x_scale, y_scale, z_scale, gripper_scale])
        self.always_render = False

        # the threshold used to identifly if the goal state has been reached.  
        # only used for training goal conditioned policy
        self.distance_threshold = 0.05
        self.ep_length = 1000

        obs = self.get_current_obs()
        self.obs_mean = np.zeros_like(obs) 
        self.obs_std = np.ones_like(obs)
        self.density_estimator = None
        self.use_global_density = False
        self.use_extrinsic_reward = False

    def set_density_estimator(self, density_estimator):
        self.density_estimator = density_estimator

    def update_reward_scale(self, mean, std):
        self.obs_mean = mean 
        self.obs_std = std 
        print('update reward norm', self.obs_mean, self.obs_std)

    def reset_goal(self):
        # Visualize target.
        print('reset goal')
        render_goal = self.selected_goal/self.obs_scale
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('goal')
        self.sim.model.site_pos[site_id] = render_goal[0:3] - sites_offset[0]

    def set_args_params(self, args):
        self.args = args 
        self.use_index = self.args.use_index
        self.reward_type = self.args.reward_type
        self.ep_length = self.args.ep_length
        self.always_render = self.args.render
        self.use_global_density = self.args.use_global_density
        self.use_extrinsic_reward = self.args.use_extrinsic_reward

        self.kde_goal = KernelDensity(kernel='gaussian', bandwidth=self.args.goal_bandwidth)
        self.kde_tra = KernelDensity(kernel='gaussian', bandwidth=self.args.trajectory_bandwidth)
        
        self.set_observation_space()   
        print('use index', self.use_index)
        print('reward_type', self.reward_type)
        print('ep_length', self.ep_length)

        if self.always_render:
            self.viewer = self._get_viewer('human')
            self.viewer._run_speed = 100
            # self.viewer._paused = True
    def set_extra_task_params(self):
        """
        Thsi function is used for reading any task-related information from xml fills.(i.e. requires element id in xml file.)
        """
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.end_effector = 'gripper_r_base'
        # self.gripper_joints = [self.sim.model.get_joint_qpos_addr('gripper_r_joint'), self.sim.model.get_joint_qpos_addr('gripper_r_joint_m')]
        self.gripper_joints = [self.sim.model.get_joint_qpos_addr('gripper_r_joint')]
        self.door_joint = [self.sim.model.get_joint_qpos_addr('door_l_joint')]
        
        xyz_range = self.mocap_high - self.mocap_low

        #finger_range = self.sim.model.jnt_qposadr('gripper_r_joint')
        gripper_r_joint_id =  self.sim.model.joint_name2id('gripper_r_joint')
        left_limit = self.sim.model.jnt_range[gripper_r_joint_id]

        gripper_r_joint_id_m =  self.sim.model.joint_name2id('gripper_r_joint_m')
        right_limit = self.sim.model.jnt_range[gripper_r_joint_id_m]

        # gripper_range = np.array([left_limit[1] - left_limit[0],
        #                 right_limit[1] - right_limit[0]])
        gripper_range = [left_limit[1] - left_limit[0]]

        doorjoint_id =  self.sim.model.joint_name2id('door_l_joint')
        door_limit = self.sim.model.jnt_range[doorjoint_id]
        door_range = [door_limit[1] - door_limit[0]]

        # if self.args.use_auto_scale:
        self.obs_scale = np.ones(self.goal_dim)
        # else:
        # self.obs_scale = np.concatenate(([1,1,1,], np.max(xyz_range)/gripper_range, np.max(xyz_range)/door_range))

        self.xyz_start = self.mocap_low * self.obs_scale[0:3]
        self.xyz_end = self.mocap_high * self.obs_scale[0:3]
        self.gripper_start = left_limit[0] * self.obs_scale[3]
        self.gripper_end = left_limit[1] * self.obs_scale[3]
        self.door_start = door_limit[0] * self.obs_scale[4]
        self.door_end = door_limit[1] * self.obs_scale[4]

    def get_current_obs(self):
        # print('in get current_obs')
        # 5 dim observation space: x, y, z, gripper_l, door opening
        # get x, y, z
        ee_pos = self.data.get_body_xpos(self.end_effector)
        # get gripper opening
        grip_angles = self.sim.data.qpos[self.gripper_joints]
        door_angle = self.sim.data.qpos[self.door_joint]
        obs = np.concatenate((ee_pos, grip_angles, door_angle))
        return obs *self.obs_scale

    def get_achieved_goal(self):
        return self.get_current_obs()

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
            achieved_goal_norm = (achieved_goal-self.obs_mean)/self.obs_std
            desired_goal_norm =  (desired_goal-self.obs_mean)/self.obs_std            
            # achieved_goal_norm_clip = np.clip(achieved_goal_norm, -5, 5)
            # desired_goal_norm_clip = np.clip(desired_goal_norm, -5, 5)

            self.kde_goal.fit([desired_goal_norm])

            if self.use_index:
                tra = np.array(trajectory)[:,:-1]
            else:
                tra = np.array(trajectory)

            tra_norm = (tra-self.obs_mean)/self.obs_std
            self.kde_tra.fit(tra_norm)

            log_g_density = self.kde_goal.score_samples([desired_goal_norm])
            log_density_from_g = self.kde_goal.score_samples([achieved_goal_norm])
            log_density_from_t = self.kde_tra.score_samples([achieved_goal_norm])

            if self.density_estimator is not None and self.use_global_density:
                log_global_density = self.density_estimator.score_samples([achieved_goal_norm])
            else:
                log_global_density = 0

            # log_density_from_g = np.clip(log_density_from_g, -np.abs(log_g_density)*5, 100)
            log_density_from_g = np.clip(log_density_from_g, -20, 100)
            log_density_from_t = np.clip(log_density_from_t, -50, 100)
            log_global_density = np.clip(log_global_density, -50, 100)

            if self.use_global_density:
                reward = ((log_density_from_g - (log_density_from_t*0.9 + log_global_density*0.1))[0]) * 0.005
            else:
                reward = ((log_density_from_g - log_density_from_t)[0]) * 0.005

        else:
            pass

        if self.use_extrinsic_reward:
            task_reward = self.compute_extrinsic_reward(achieved_goal)
        else:
            task_reward = 0

        # if info is not None and self.reward_type == 'density':
        #     # print(desired_goal, desired_goal_norm)
        #     # print(achieved_goal, achieved_goal_norm)            
        #     print(task_reward, log_density_from_g, log_density_from_t, log_global_density, (log_density_from_g - log_density_from_t), log_g_density, achieved_goal[-2:])
        return reward + task_reward

    def compute_extrinsic_reward(self, achieved_goal):
        task_angle = -0.304 * 4
        task_weight = 10.0
        x = achieved_goal[-1] - task_angle
        delta_alpha = 0.5
        if np.abs(x) < 0.035:
            task_reward = np.exp(-(x/delta_alpha)**2)/(np.abs(delta_alpha)*np.sqrt(np.pi))
        else:
            task_reward = 0
            
        return task_reward * task_weight    

    def get_extrinsic_reward(self, achieved_goal_list):
        task_rewards = []
        print('achieved_goal_list size', achieved_goal_list.shape)
        for achieved_goal in achieved_goal_list:
            task_reward = self.compute_extrinsic_reward(achieved_goal)
            task_rewards.append(task_reward)

        return np.array(task_rewards)