import os, time
import argparse
import logging, json
import numpy as np
import gym

from algo.sac_goals import SAC
from algo.ppo_goals import PPO2
from algo.her_goal import HER
from algo.sac_policy import MlpPolicy
from algo.ppo_policy import LSTMPolicyRND, LSTMPolicy

from envs.yumi.goal_vec_env import GaolDummyVecEnv
from envs.yumi.yumi_door_env import YumiDoorEnv
from envs.yumi.yumi_door_button_env import YumiDoorButtonEnv
from envs.point_maze.point_maze_env import GoalProposePointMazeEnv

def make_env(env_keyword, input_args):
    """
    create environment

    :param env_keyword: the keyword for environment id
    :param input_args: additional arguments for environment settings, 
        such as episode length, reward type and kernel bandwidth
    """
    def _init():
        if env_keyword == 'yumi':
            env = YumiDoorEnv()
            env.set_args_params(input_args)
        elif env_keyword == 'yumi_door_button':
            env = YumiDoorButtonEnv()
            env.set_args_params(input_args)        
        elif env_keyword == 'maze':
            env = GoalProposePointMazeEnv()
            env.set_args_params(input_args)
        else:
            raise Exception('please select environment from the list: ["yumi", "maze",  "yumi_door_button"]')
        return env
    return _init

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", help="select the RL algorithm.", type=str, default='ppo')
    parser.add_argument("--env", help="select the environment.", type=str, default='maze')
    parser.add_argument("--n_env", help="select the number of environment to run in parallel.", type=int, default=10)
    parser.add_argument("--plot_coverage", help="plot the coverage graph when update reward function.", action='store_true')
    parser.add_argument("--plot_density", help="plot the density graph when update reward function", action='store_true')
    parser.add_argument("--plot_overall_coverage", help="plot the overall coverage trend when update reward function.", action='store_true')
    parser.add_argument("--plot_entropy", help="plot the history states when update reward function.", action='store_true')
    parser.add_argument("--policy_type", help="select which policy type should be used.", type=str, default='mlp')
    parser.add_argument("--reward_type", help="select which reward function should be used.", type=str, default='density')
    parser.add_argument("--save_path", help="enter the path for saving training result.", type=str, default='')
    parser.add_argument("--n_sampled_goal", help="number of goal sampling in HER", type=int, default=10)
    parser.add_argument("--history_buffer_size", help="the size of history buffer", type=int, default=1000000)
    parser.add_argument("--goal_selection_strategy", help="goal selection strategy in HER", type=str, default='future')
    parser.add_argument("--goal_bandwidth", help="the bandwidth for goal distribution", type=float, default=2)
    parser.add_argument("--trajectory_bandwidth", help="the bandwidth for trajectory distribution", type=float, default=0.5)
    parser.add_argument("--use_index", help="the bandwidth for trajectory distribution", action='store_true')
    parser.add_argument("--ep_length", help="the number of steps for one trajectory", type=int, default=200)
    parser.add_argument("--use_auto_scale", help="the bandwidth for trajectory distribution", action='store_true')
    parser.add_argument("--skew_alpha", help="alpha for skew-fit goal distribution", type=float, default=-2.1)
    parser.add_argument("--render", help="render environment", action='store_true')
    parser.add_argument("--use_global_density", help="use global density in reward computation", action='store_true')
    parser.add_argument("--use_extrinsic_reward", help="use extrinsic density in reward computation", action='store_true')

    return parser.parse_args()

def main():
    args = arg_parser()
    exp_name = args.save_path
    # args.save_path = '/home/xi/model/final_exp/' + args.save_path
    args.save_path = '/home/xi/model/paper_exp/' + args.save_path
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # save args
    with open(args.save_path + '/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.alg == 'ppo':
        env = GaolDummyVecEnv([make_env(args.env, args) for i in range(args.n_env)])
    else:
        env = make_env(args.env, args)()

    if args.alg == 'ppo':
        if args.reward_type == 'rnd':
            use_policy = LSTMPolicyRND
        else:
            use_policy = LSTMPolicy

        n_steps = 500
        nminibatches = args.n_env
        model = PPO2(use_policy, env, n_steps=n_steps, ent_coef=0.0, 
                        nminibatches=nminibatches, noptepochs=15, args=args, verbose=1, gamma=0.99)
        # model.tensorboard_log = '/home/xi/model/log/' + exp_name + '/'
        model.learn(total_timesteps=int(5e7))

    elif args.alg == 'her_sac':
        kwargs = {'learning_rate': 1e-3, 'args':args}
        buffer_size = args.history_buffer_size
        model = HER('MlpPolicy', env, SAC, n_sampled_goal=args.n_sampled_goal, 
                    goal_selection_strategy=args.goal_selection_strategy,
                    verbose=1, buffer_size=buffer_size, gamma=0.98, batch_size=1024,
                    policy_kwargs=dict(layers=[32, 64, 128, 64, 32]), **kwargs)

        # model.model.tensorboard_log = '/home/xi/model/log/' + exp_name + '/'
        model.learn(total_timesteps=int(5e7))

    if args.alg  == 'test':
        env = make_env(args.env, args)()
        model = SAC.load('/home/xi/model/model.pkl', env=env, args=args)
        model.test(120)

if __name__ == '__main__':
    main()
