# Skew-Explore

This is the repository hosting the code used for the paper Skew-Explore. The code contains the implementation of the Skew-Explore algorithm, 3 goal-proposing environments in Mujoco, RL algorithms including PPO, SAC, HER, and RND. The PointMaze environment is modified from the [rllab](https://github.com/rll/rllab), the RL algorithms are modified from the [stable-baselines](https://github.com/hill-a/stable-baselines) and RND is inspired by [random-network-distillation](https://github.com/openai/random-network-distillation).

## Installation

### Dependencies

#### Python version

For this experiment, we use Python `3.6`. Here's a [guide to installing Python `3.6` in Ubuntu 16.04](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/)

#### Python Dependencies

For managing Python dependencies it is possible to use [conda](https://conda.io/en/latest/). Once you have cloned the repository, proceed to install the dependencies defined in the `Pipfile` or 'environment.yml'

#### conda

```
conda update conda
conda env create -f environment.yml
``` 

### Install rllab
The "point_maze" environment requires [rllab](https://github.com/rll/rllab) and MuJoCo version 131

+ First clone rllab from [HERE](https://github.com/rll/rllab.git).
+ Then download mujoco version 131 [HERE](https://www.roboti.us/download/mjpro131_linux.zip).
+ cd to the rllab repo and run "pip install ." to install it.
+ Run "./scripts/setup_mujoco.sh" to setup MuJoCo environment.
+ The "vendor" folder in rllab should contain two sub-folders "mujoco" and "mujoco_models". 
+ You may need to copy the "vendor" folder to "./anaconda3/envs/skew_explore/lib/python3.6/site-packages" if it cannot be found.

other dependencies:
```
pip install cached_property mako theano
``` 

### Install Mojuco and mujoco_py
The "yumi" and "yumi_door_button" environments are implemented using MuJoCo version 150.
Please follow this [link](https://github.com/openai/mujoco-py) to install MuJoCo 150.


### Install stable-baselines from the submodule
The code should be run with the stable-baselines from the submodule.
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install stable-baselines
``` 

### Install gym
```
pip install gym
``` 

### Install tensorflow
```
pip install tensorflow
``` 

### Getting started:
To reproduce the result of the exploration experiments, please run the following commands:
+ for point_maze environment
```
python run_skew_explore.py --plot_coverage --alg her_sac --env maze --save_path maze_hersac_distrib --reward_type density --goal_bandwidth 2 --trajectory_bandwidth 0.5 --use_index --render
``` 
+ for yumi_door environment
```
python run_skew_explore.py --plot_coverage --alg her_sac --env yumi --save_path door_hersac_distrib --reward_type density --goal_bandwidth 1.5 --trajectory_bandwidth 0.05 --use_index --use_auto_scale  --render
``` 


To reproduce the result of the sparse reward experiments, please run the following commands:
+ for point_maze environment
```
python run_skew_explore.py --plot_coverage --alg her_sac --env maze --save_path maze_hersac_distrib_sparse --reward_type density --goal_bandwidth 2 --trajectory_bandwidth 0.5 --use_index --use_extrinsic_reward  --render
``` 
+ for yumi_door environment
```
python run_skew_explore.py --plot_coverage --alg her_sac --env yumi --save_path door_hersac_distrib_sparse --reward_type density --goal_bandwidth 1.5 --trajectory_bandwidth 0.05 --use_index --use_auto_scale --use_extrinsic_reward  --render
``` 
+ for yumi_door_button environment
```
python run_skew_explore.py --plot_coverage --alg her_sac --env yumi_door_button --save_path doorbutton_hersac_distrib_sparse --reward_type density --goal_bandwidth 1.5 --trajectory_bandwidth 0.2 --use_index --use_auto_scale --use_extrinsic_reward --use_extrinsic_reward --history_buffer_size 5000000  --render
``` 

