import os, copy
import numpy as np
from gym import spaces, error
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

from envs.yumi.yumi_controller import Controller, Mode
from envs.yumi import utils 

DEFAULT_SIZE = 500

class YumiEnv(mujoco_env.MujocoEnv):
    """
    Simulated Yumi environment in Mujoco.

    :param xml_name: xml file of the robot model and the environment
    :param control_type: the control type of yumi robot, can be 'mocap' or 'controller'
    :param mocap_high: the up-limit of the mocap position
    :param mocap_low: the low-limit of the mocap position
    :param mocap_init_pos: the initial position of mocap
    :param mocap_init_quat: the initial orientation of mocap
    """
    def __init__(self, xml_name='yumi_reach.xml', control_type='mocap', 
                    mocap_high=np.hstack((0.1, 0.1, 0.3)), mocap_low=np.hstack((-0.15, -0.2, 0.2)),
                    mocap_init_pos=None, mocap_init_quat=None):
        self.name = 'yumi'
        self.end_effector = 'gripper_r_base'
        self.action_scale = None 
        self.mocap_low = mocap_low 
        self.mocap_high = mocap_high 
        self.always_render = False

        if mocap_init_pos is None:
            mocap_init_pos = np.array([0, 0, 0.4])
        if mocap_init_quat is None:
            mocap_init_quat = np.array([-0.70699531,0.09357381,-0.4721928,0.51810765])

        self.mocap_init_quat = mocap_init_quat
        self.mocap_init_pos = mocap_init_pos
        
        self.set_mujoco_basic(xml_name)
        self.set_extra_task_params()
        
        self.set_observation_space()
        self.set_action_space()
        self.seed()

        self.control_type = control_type
        if control_type == 'mocap':  
            if self.action_scale is None:
                self.action_scale = np.array([2./100] * 3) 
        elif control_type == 'controller':
            self.ctrl = Controller(self.sim)

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def set_mujoco_basic(self, xml_name, rgb_rendering_tracking=True):
        root_dir = os.path.dirname(__file__)
        model_path = os.path.join(root_dir, 'yumi_model', xml_name)     

        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        n_substeps = 10
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.data = self.sim.data
        self.viewer = None
        self.rgb_rendering_tracking = rgb_rendering_tracking
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_qpos = initial_state.qpos 
        self.initial_qvel = initial_state.qvel

    def set_extra_task_params(self):
        """
        function to set task related paramaters
        """
        pass

    def set_observation_space(self):
        observation = self._get_obs()
        self.obs_dim = observation.size

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_dim = self.model.nu

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_action(self, action):
        # assert (action.shape == (self.action_dim,)),"action dimension does not match!"
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if self.control_type == 'mocap':
            action = action*self.action_scale
            mocap_action, ctrl_action = self.process_mocap_action(action)
            # print(mocap_action, ctrl_action)
            utils.mocap_set_action(self.sim, mocap_action, self.mocap_low, self.mocap_high, self.end_effector)
        if self.control_type == 'controller':
            ctrl_action = self.process_controller_action(action)

        utils.ctrl_set_action(self.sim, ctrl_action)

    def process_mocap_action(self, action):
        raise NotImplementedError()

    def process_controller_action(self, action):
        raise NotImplementedError()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            np.clip(self.sim.data.qvel.flat[:7], -10, 10)
        ])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        obs = self._get_obs()

        done = self.is_done()
        reward = self.compute_reward()
        if self.always_render:
            self.render()
        return obs, reward, done, {}

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.data.set_mocap_pos('mocap', self.mocap_init_pos)
        self.data.set_mocap_quat('mocap', self.mocap_init_quat)   

        self.sim.step()
        obs = self._get_obs()
        return obs

    def reset_ep(self):
        print('in reset ep')
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim_ep()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewers[mode] = self.viewer
        return self.viewer

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_qpos, self.initial_qvel)
        self.sim.forward()
        return True

    def _reset_sim_ep(self):
        self.sim.set_state(self.initial_qpos, self.initial_qvel)
        self.sim.forward()
        return True

    def compute_reward(self):
        raise NotImplementedError()

    def is_done(self):
        return False









