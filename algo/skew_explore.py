import os
import glob
import time
import logging
import numpy as np
from functools import reduce
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from stable_baselines.common.running_mean_std import RunningMeanStd

logging.basicConfig(level=logging.INFO)

class SkewExploreBase():
    """
    Class for state density estimation, goal proposing and result plotting.
    The goal proposing distribution is computed using Skew-fit algorithm (https://arxiv.org/abs/1903.03698) 
        with a sklearn kernel density estimation. 

    :param env: the environment, used to access environment properties such as state range, 
        pass proposed goals and pass the on-line mean and standard diviation for state normalization 
    :param args: additional arguments for configuring result plotting
    """

    def __init__(self, env, args):
        self.density_estimator_raw = None #KernelDensity(kernel='gaussian', bandwidth=0.1)
        self.density_estimator = None
        self.args = args
        self.env = env        
        self.obs_rms = None #RunningMeanStd(shape=env.observation_space)

        self.skew_sample_num = 10000 #25000
        self.skew_alpha = args.skew_alpha #-2.5 #-2.3 #-2.1
        self.goal_sampling_num = 100

        self.init_buffer = False
        self.obs_hist = None 
        self.obs_next_hist = None
        self.dones = None

        self.obs_in_use = None
        self.obs_new = None

        self.plot_coverage = False
        self.plot_density = False
        self.plot_overall_coverage = False
        self.plot_entropy = False

        self.count = 0
        self.coverages = []
        self.entropy = []
        self.task_reward = []

        self.obs_mean = None 
        self.obs_std = None

        self.plot_coverage = self.args.plot_coverage
        self.plot_density = self.args.plot_density
        self.plot_overall_coverage = self.args.plot_overall_coverage
        self.plot_entropy = self.args.plot_entropy

        # for coverage plotting
        if self.args.env == 'maze':
            self.bandwidth = 0.1
            self.init_maze_plotting_params()
            sigma = 0.1
        elif self.args.env == 'yumi':
            # self.bandwidth = 0.003
            self.bandwidth = 0.1
            self.init_door_plotting_params()
            sigma = 0.1
        elif self.args.env == 'yumi_box_pick' or self.args.env == 'yumi_door_button':
            self.bandwidth = 0.11
            self.init_boxpick_plotting_params()
            sigma = 0.005

        self.beta = 1/(sigma**2 * 2)

    def init_maze_plotting_params(self):
        """
        Initialize parameters to evaluate and plot results of point maze environment
        """        
        xbins=50j
        ybins=50j
        x_start=-6
        x_end=6
        y_start=-12
        y_end=4

        self.xx, self.yy = np.mgrid[x_start:x_end:xbins,
                          y_start:y_end:ybins]        
        self.eval_sample = np.vstack([self.yy.ravel(), self.xx.ravel()]).T
        self.eval_sample_min_dist = np.ones(len(self.eval_sample))

        self.skewed_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        self.entropy_shift = np.array([y_start, x_start])
        self.entropy_scale = np.array([(y_end-y_start, x_end-x_start)])
  
    def init_door_plotting_params(self):
        """
        Initialize parameters to evaluate and plot results of yumi door opening environment
        """           
        xbins, ybins, zbins, gbins, dbins = 10j, 10j, 10j, 2j, 10j
        self.x_start, self.y_start, self.z_start = self.env.xyz_start
        self.x_end, self.y_end, self.z_end = self.env.xyz_end
        self.g_start, self.g_end = self.env.gripper_start, self.env.gripper_end
        self.d_start, self.d_end = self.env.door_start, self.env.door_end

        # for xy and door angle plotting
        self.mesh_xx, self.mesh_yy = np.mgrid[self.x_start:self.x_end:xbins, self.y_start:self.y_end:ybins]
        self.dd = np.mgrid[self.d_start:self.d_end:dbins]
        self.xy_eval_sample = np.vstack([self.mesh_xx.ravel(), self.mesh_yy.ravel()]).T
        self.door_eval_sample = np.vstack([self.dd.ravel()]).T
        self.door_eval_sample_min_dist = np.ones(len(self.door_eval_sample))  

        # for coverage plotting
        self.xx, self.yy, self.zz, self.gg, self.dd = np.mgrid[  self.x_start:self.x_end:xbins,
                                                                            self.y_start:self.y_end:ybins,
                                                                            self.z_start:self.z_end:zbins,
                                                                            self.g_start:self.g_end:gbins,
                                                                            self.d_start:self.d_end:dbins]
        self.eval_sample = np.vstack([self.xx.ravel(), self.yy.ravel(), self.zz.ravel(), self.gg.ravel(), self.dd.ravel()]).T
        self.eval_sample_min_dist = np.ones(len(self.eval_sample))  

        self.xy_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.doorangle_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        self.skewed_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        self.entropy_shift = np.array([self.x_start, self.y_start, self.z_start, self.g_start, self.d_start])
        self.entropy_scale = np.array([self.x_start-self.x_end, self.y_start-self.y_end, self.z_start-self.z_end, self.g_start-self.g_end, self.d_start-self.d_end])

    def init_boxpick_plotting_params(self):
        """
        Initialize parameters to evaluate and plot results of yumi door button environment
        """          
        xbins, ybins, zbins, gbins, dlbins, drbins = 1j, 1j, 1j, 1j, 10j, 5j
        self.x_start, self.y_start, self.z_start = self.env.xyz_start
        self.x_end, self.y_end, self.z_end = self.env.xyz_end
        self.g_start, self.g_end = self.env.gripper_start, self.env.gripper_end
        self.dl_start, self.dl_end = self.env.door_l_start, self.env.door_l_end
        self.dr_start, self.dr_end = self.env.door_r_start, self.env.door_r_end

        # for xy and door angle plotting
        self.mesh_xx, self.mesh_yy = np.mgrid[self.x_start:self.x_end:xbins, self.y_start:self.y_end:ybins]
        self.xy_eval_sample = np.vstack([self.mesh_xx.ravel(), self.mesh_yy.ravel()]).T

        self.mesh_ld, self.mesh_rd = np.mgrid[self.dl_start:self.dl_end:dlbins, self.dr_start:self.dr_end:drbins]
        self.door_eval_sample = np.vstack([self.mesh_ld.ravel(), self.mesh_rd.ravel()]).T
        self.door_eval_sample_min_dist = np.ones(len(self.door_eval_sample))  

        # for coverage plotting
        self.xx, self.yy, self.zz, self.gg, self.dl, self.dr = np.mgrid[  self.x_start:self.x_end:xbins,
                                                                            self.y_start:self.y_end:ybins,
                                                                            self.z_start:self.z_end:zbins,
                                                                            self.g_start:self.g_end:gbins,
                                                                            self.dl_start:self.dl_end:dlbins,
                                                                            self.dr_start:self.dr_end:drbins]
        self.eval_sample = np.vstack([self.xx.ravel(), self.yy.ravel(), self.zz.ravel(), self.gg.ravel(), self.dl.ravel(), self.dr.ravel()]).T
        self.eval_sample_min_dist = np.ones(len(self.eval_sample))  

        self.xy_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.doorangle_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        self.skewed_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

        self.entropy_shift = np.array([self.x_start, self.y_start, self.z_start, self.g_start, self.dl_start, self.dr_start])
        self.entropy_scale = np.array([self.x_start-self.x_end, self.y_start-self.y_end, self.z_start-self.z_end, 
                                        self.g_start-self.g_end, self.dl_start-self.dl_end, self.dr_start-self.dr_end])

    def plot_maze_metrics(self):
        """
        Plot intermediate result for point maze environment
        """          
        fig,(ax1, ax2, ax3, ax4, ax5)  = plt.subplots(5,1, figsize=(5,20))

        if self.plot_entropy or self.plot_density:
            eval_sample_log_density = self.get_log_density(self.eval_sample)
            eval_sample_density = np.exp(eval_sample_log_density)            

        if self.plot_density:
            zz_density= np.reshape(eval_sample_density, self.xx.shape)            
            im = ax1.pcolormesh(self.yy, self.xx, zz_density)

        if self.plot_entropy:
            entropy = self.compute_entropy(eval_sample_density, eval_sample_log_density)
            self.entropy.append(entropy)
            ax3.plot(self.entropy)
            # np.save(self.args.save_path + '/entropy.npy', self.entropy)

        if self.plot_coverage:
            z_coverage = self.get_coverage()
            zz_coverage = np.reshape(z_coverage, self.xx.shape)            
            im = ax2.pcolormesh(self.yy, self.xx, zz_coverage, vmin=0, vmax=1)         

        # if the use_extrinsic_reward flag is true, it will only plot the curve of task_reward
        if self.args.use_extrinsic_reward:
            ax4.plot(self.task_reward)
        elif self.plot_overall_coverage:
            self.coverages.append(z_coverage.mean())
            ax4.plot(self.coverages)
            # np.save(self.args.save_path + '/coverage.npy', self.coverages)          

        sample_goal = self.sample_goal(200)
        ax5.scatter(sample_goal[:, 0], sample_goal[:, 1], s=10, color='red')
        ax5.set_xlim([-12, 4])
        ax5.set_ylim([-6, 6])

        if self.plot_density or self.plot_coverage or self.plot_overall_coverage or self.plot_entropy:
            plt.savefig(self.args.save_path + '/coverage_' + str(self.count)+'.svg')
        plt.close()

    def plot_door_metrics(self):
        """
        Plot intermediate result for door opening environment
        """          
        fig, ax = plt.subplots(3,2, figsize=(10,15))

        if self.plot_density:
            xy_eval_sample_norm = (self.xy_eval_sample - self.obs_mean[:2])/self.obs_std[:2]
            xy_sample_density = np.exp(self.xy_estimator.score_samples(xy_eval_sample_norm))
            xy_density = np.reshape(xy_sample_density, self.mesh_xx.shape)

            door_eval_sample_norm = (self.door_eval_sample - self.obs_mean[-1])/self.obs_std[-1]
            door_sample_density = np.exp(self.doorangle_estimator.score_samples(door_eval_sample_norm))

            im = ax[0][0].pcolormesh(self.mesh_xx, self.mesh_yy, xy_density)
            im = ax[0][1].scatter(self.door_eval_sample, door_sample_density)
            ax[0][0].set_xlim([self.x_start-0.05, self.x_end+0.05])
            ax[0][0].set_ylim([self.y_start-0.05, self.y_end+0.05])
            ax[0][1].set_xlim([self.d_start-0.05, self.d_end+0.05])
            ax[0][1].set_ylim([-0.05, 1])          

        if self.plot_coverage:
            door_sample_coverage = self.get_door_coverage(self.door_eval_sample, -1)
            ax[2][1].scatter(self.door_eval_sample, door_sample_coverage)

        # if the use_extrinsic_reward flag is true, it will only plot the curve of task_reward
        if self.args.use_extrinsic_reward:
            ax[2][0].plot(self.task_reward)
        elif self.plot_overall_coverage:
            eval_sample_coverage = self.get_coverage()
            self.coverages.append(eval_sample_coverage.mean())
            ax[2][0].plot(self.coverages)            

        if self.plot_entropy:
            eval_sample_scaled = (self.eval_sample - self.entropy_shift)/self.entropy_scale
            eval_sample_log_density = self.density_estimator_raw.score_samples(eval_sample_scaled) #self.get_log_density(eval_sample_norm)
            eval_sample_density = np.exp(eval_sample_log_density)
            entropy = self.compute_entropy(eval_sample_density, eval_sample_log_density)
            self.entropy.append(entropy)
            ax[0][0].plot(self.entropy)
            np.save(self.args.save_path + '/entropy', np.array(self.entropy))

        sample_goal = self.sample_goal(200)
        ax[1][0].scatter(sample_goal[:, 0], sample_goal[:, 1], s=10, color='red')
        ax[1][0].set_xlim([self.x_start-0.05, self.x_end+0.05])
        ax[1][0].set_ylim([self.y_start-0.05, self.y_end+0.05])

        ax[1][1].scatter(sample_goal[:, -1], np.ones(len(sample_goal)), s=1, color='red')
        ax[1][1].set_xlim([self.d_start-0.05, self.d_end+0.05])

        if self.plot_density or self.plot_coverage or self.plot_overall_coverage or self.plot_entropy:
            plt.savefig(self.args.save_path + '/coverage_' + str(self.count)+'.svg')
        plt.close()

    def plot_boxpick_metrics(self):
        """
        Plot intermediate result for door button environment
        """          
        fig, ax = plt.subplots(3,2, figsize=(15,15))

        if self.plot_density:
            xy_eval_sample_norm = (self.xy_eval_sample - self.obs_mean[:2])/self.obs_std[:2]
            xy_sample_density = np.exp(self.xy_estimator.score_samples(xy_eval_sample_norm))
            xy_density = np.reshape(xy_sample_density, self.mesh_xx.shape)

            door_eval_sample_norm = (self.door_eval_sample - self.obs_mean[-2:])/self.obs_std[-2:]
            door_sample_density = np.exp(self.doorangle_estimator.score_samples(door_eval_sample_norm))
            door_density = np.reshape(door_sample_density, self.mesh_ld.shape)

            im = ax[0][0].pcolormesh(self.mesh_xx, self.mesh_yy, xy_density)
            im = ax[0][1].pcolormesh(self.mesh_ld, self.mesh_rd, door_density)

            ax[0][0].set_xlim([self.x_start-0.05, self.x_end+0.05])
            ax[0][0].set_ylim([self.y_start-0.05, self.y_end+0.05])
            ax[0][1].set_xlim([self.dl_start, self.dl_end])
            ax[0][1].set_ylim([self.dr_start, self.dr_end])

        # if the use_extrinsic_reward flag is true, it will only plot the curve of task_reward
        if self.args.use_extrinsic_reward:
            ax[2][0].plot(self.task_reward)
        elif self.plot_overall_coverage:
            eval_sample_coverage = self.get_coverage()
            self.coverages.append(eval_sample_coverage.mean())
            ax[2][0].plot(self.coverages)

        if self.plot_coverage:
            door_sample_coverage = self.get_door_coverage(self.door_eval_sample, -2)
            door_coverage = np.reshape(door_sample_coverage, self.mesh_ld.shape)
            ax[2][1].pcolormesh(self.mesh_ld, self.mesh_rd, door_coverage, vmin=0, vmax=1)

        if self.plot_entropy:
            eval_sample_norm = (self.eval_sample - self.obs_mean)/self.obs_std
            eval_sample_log_density = self.get_log_density(eval_sample_norm)
            eval_sample_density = np.exp(eval_sample_log_density)
            entropy = self.compute_entropy(eval_sample_density, eval_sample_log_density)
            self.entropy.append(entropy)      
            np.save(self.args.save_path + '/entropy', np.array(self.entropy))

        sample_goal = self.sample_goal(200)
        ax[1][0].scatter(sample_goal[:, 0], sample_goal[:, 1], s=10, color='red')
        ax[1][0].set_xlim([self.x_start-0.05, self.x_end+0.05])
        ax[1][0].set_ylim([self.y_start-0.05, self.y_end+0.05])

        ax[1][1].scatter(sample_goal[:, -2], sample_goal[:, -1], s=1, color='red')
        ax[1][1].set_xlim([self.dl_start, self.dl_end])
        ax[1][1].set_ylim([self.dr_start, self.dr_end])

        if self.plot_density or self.plot_coverage or self.plot_overall_coverage or self.plot_entropy:
            plt.savefig(self.args.save_path + '/coverage_' + str(self.count)+'.svg')
        plt.close()

    def activate_buffer(self):
        """
        Update the history buffer, 
        update the state density estimation model
        update the goal proposing distribution model 
        """  
        start_time = time.time()
        if self.obs_hist is None:
            self.obs_hist = self.obs_new
            # self.obs_next_hist = obs_next
            # self.done_hist = dones
        else:
            self.obs_hist = np.concatenate((self.obs_hist, self.obs_new), axis=0)
        
        self.fit_model()
        fitmodel_time = time.time()
        logging.info("fit model time cost: %f" %(fitmodel_time-start_time))

        self.train_skew_generator()
        fitskew_time = time.time()
        logging.info("fit skew-model time cost: %f" %(fitskew_time-start_time))

        # update goal samples in the environment and update the obs mean and std
        if self.args.use_auto_scale:
            self.env.update_reward_scale(self.obs_mean, self.obs_std)
        sampled_goal = self.sample_goal(self.goal_sampling_num)
        self.env.set_goals(sampled_goal)
        self.env.set_density_estimator(self.density_estimator)
  
        # compute task_reward
        if self.args.use_extrinsic_reward:
            # dones = self.dones.astype(int)
            task_reward = self.env.get_extrinsic_reward(self.obs_new) # * dones
            self.task_reward.append(task_reward.mean())

        # plotting
        if self.plot_density or self.plot_coverage or self.plot_overall_coverage or self.plot_entropy:
            if self.args.env == 'maze':
                self.plot_maze_metrics()
            elif self.args.env == 'yumi':
                self.plot_door_metrics()
            elif self.args.env == 'yumi_box_pick' or self.args.env == 'yumi_door_button':
                self.plot_boxpick_metrics()
        self.obs_new = None
        self.dones = None
        finish_time = time.time()
        logging.info('time cost: %f'%(finish_time - start_time))

        np.save(self.args.save_path + '/entropy', np.array(self.entropy))
        np.save(self.args.save_path + '/coverage', np.array(self.coverages))
        np.save(self.args.save_path + '/task_reward', np.array(self.task_reward))
        logging.info('end of activate buffer')

    def train_skew_generator(self):
        """
        Update the goal proposing distribution using the Skew-fit algorithm
        (https://arxiv.org/abs/1903.03698) 
        """          
        # NOTE: The skewed samples are sampled from density estimator
        self.skew_samples, skew_samples_density = self.get_samples_and_density(self.skew_sample_num)

        # self.skew_samples = self.density_estimator.sample(self.skew_sample_num)
        # skew_samples_density = np.exp(self.density_estimator.score_samples(self.skew_samples))
        skew_unnormalized_weights = skew_samples_density * skew_samples_density ** self.skew_alpha

        skew_zeta_alpha = np.sum(skew_unnormalized_weights)
        self.skew_weights = skew_unnormalized_weights/skew_zeta_alpha

        self.skewed_estimator.fit(self.skew_samples, sample_weight=self.skew_weights)        

    def sample_goal(self, goal_num):
        """
        Sample goal states from the goal proposing distribution
        """
        sampled_data = self.skewed_estimator.sample(goal_num)
        sampled_data = sampled_data * self.obs_std + self.obs_mean
        return sampled_data #sampled_data[goal_index]

    def get_samples_and_density(self, sample_num):
        raise NotImplementedError()

    def fit_model(self):
        raise NotImplementedError()

    def get_pvisited(self, obs_test):
        raise NotImplementedError()

    def get_log_density(self, obs_test):
        raise NotImplementedError()

    def get_coverage(self):
        """
        Compute the current coverage of the states used for evaluation
        """        
        p_coverage = np.zeros(len(self.eval_sample))

        for i in range(len(self.eval_sample)):
            obs = self.eval_sample[i]
            obs_diff = self.obs_new - obs

            diff_norm = LA.norm(obs_diff, axis=1)
            min_dist = diff_norm.min()

            current_min_dist = self.eval_sample_min_dist[i]
            new_min_dist = np.minimum(current_min_dist, min_dist)
            self.eval_sample_min_dist[i] = new_min_dist

            pv = np.exp(-new_min_dist*new_min_dist*self.beta)
            p_coverage[i] = 1-pv

        return p_coverage

    def get_door_coverage(self, door_eval_sample, index):
        """
        Compute the current coverage of the door states used for evaluation
        """         
        p_coverage = np.zeros(len(door_eval_sample))

        for i in range(len(door_eval_sample)):
            obs = door_eval_sample[i]
            obs_diff = self.obs_new[:,index:] - obs

            diff_norm = LA.norm(obs_diff, axis=1)
            min_dist = diff_norm.min()

            current_min_dist = self.door_eval_sample_min_dist[i]
            new_min_dist = np.minimum(current_min_dist, min_dist)
            self.door_eval_sample_min_dist[i] = new_min_dist

            pv = np.exp(-new_min_dist*new_min_dist*self.beta)
            p_coverage[i] = pv

        return p_coverage

    def compute_entropy(self, density, log_density):
        """
        Compute the entropy
        """           
        d_mul_logd = density*log_density
        entropy = -np.sum(d_mul_logd)
        return entropy

    def get_preach(self, obs_from):
        raise NotImplementedError()

    def get_preal(self, obs_test):
        raise NotImplementedError()

    def compute_reward(self, obs_test, use_sampling=False):
        raise NotImplementedError()

    def update_history(self, obs, dones):
        """
        Save the new states in the self.obs_new buffer.
        the self.obs_new buffer will be merged to the self.obs buffer
        in activate_buffer() function.
        """ 
        if self.args.use_index:
            obs = obs[:,:-1]
        if self.obs_mean is None:
            self.obs_mean = np.zeros_like(obs)[0]
            self.obs_std = np.ones_like(obs)[0]
            self.obs_rms = RunningMeanStd(shape=obs.shape)

        if self.obs_new is None:
            self.obs_new = obs
            # self.obs_next_hist = obs_next
            self.dones = dones
        else:
            self.obs_new = np.concatenate((self.obs_new, obs), axis=0)
            # self.obs_next_hist = np.concatenate((self.obs_next_hist, obs_next), axis=0)
            self.dones = np.concatenate((self.dones, dones), axis=0)
        self.obs_rms.update(obs)


class SkewExploreKDE(SkewExploreBase):
    """
    Class for state density estimation using sklearn kernel density estimator, goal proposing and result plotting.
    The goal proposing distribution is computed using Skew-fit algorithm (https://arxiv.org/abs/1903.03698) 
        with a sklearn kernel density estimation. 

    :param env: the environment, used to access environment properties such as state range, 
        pass proposed goals and pass the on-line mean and standard diviation for state normalization 
    :param args: additional arguments for configuring result plotting
    """    
    def __init__(self, env, args):
        super().__init__(env, args)
        self.density_estimator = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.density_estimator_raw = KernelDensity(kernel='gaussian', bandwidth=0.1)
        self.sample_prob = []

        ## for coverage computation
        self.obs_in_use = None
        self.num_points_estimator = 50000 #70000

    def fit_model(self):
        """
        fit the kernel density model
        """
        self.count += 1
        logging.info('Activate buffer')
        self.init_buffer = True

        selected_index = np.random.randint(len(self.obs_hist), size=self.num_points_estimator)
        self.obs_in_use = self.obs_hist[selected_index]

        # only yumi environments need to normalize the observation states on-line
        if self.args.use_auto_scale:
            if self.args.env == 'yumi' or self.args.env == 'yumi_box_pick' or self.args.env == 'yumi_door_button':
                if self.count %2 == 0:
                    self.obs_mean = self.obs_rms.mean[0] #np.mean(self.obs_in_use, axis=0)
                    self.obs_std = np.sqrt(self.obs_rms.var[0]) + 1e-8 #np.std(self.obs_in_use, axis=0) + 0.000000001

        self.obs_nomalized = (self.obs_in_use - self.obs_mean)/self.obs_std
        self.density_estimator.fit(self.obs_nomalized)

        # scale the observation for entropy computation
        self.obs_scaled = (self.obs_in_use - self.entropy_shift)/self.entropy_scale
        self.density_estimator_raw.fit(self.obs_scaled)

        if self.plot_density:
            if self.args.env == 'yumi':
                self.xy_estimator.fit(self.obs_nomalized[:, 0:2])
                self.doorangle_estimator.fit(self.obs_nomalized[:, -1:])
            elif self.args.env == 'yumi_box_pick' or self.args.env == 'yumi_door_button':
                self.xy_estimator.fit(self.obs_nomalized[:, 0:2])
                self.doorangle_estimator.fit(self.obs_nomalized[:, -2:])            

    def get_samples_and_density(self, sample_num):
        """
        Sample states from the density model and compute the sample density
        """        
        samples = self.density_estimator.sample(self.skew_sample_num)
        samples_density = np.exp(self.density_estimator.score_samples(samples))
        return samples, samples_density

    def get_log_density(self, obs_test):
        """
        Compute log density
        """            
        log_density = self.density_estimator.score_samples(obs_test)
        return log_density

    def get_density(self, obs_test):
        """
        Compute density
        """         
        density = np.exp(self.density_estimator.score_samples(obs_test))
        return density        
