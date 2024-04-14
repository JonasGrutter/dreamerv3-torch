import argparse
import functools
import os
import pathlib
import sys



import numpy as np
import ruamel.yaml as yaml

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy
from gym.spaces import Box

import torch
from torch import nn
from torch import distributions as torchd


#----- ORBIT-Start

import argparse
import os
import traceback
from omni.isaac.orbit.app import AppLauncher
from datetime import datetime
import carb
from utils.wandbutils import WandbSummaryWriter


# local imports
import cli_args  # i
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
args_cli.num_envs = 2
args_cli.task= 'Isaac-m545-v0'

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym
from omni.isaac.orbit.envs import RLTaskEnvCfg
import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


from dreamer_class import Dreamer

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def load_agent(agent, path):
    checkpoint = torch.load(path)
    agent.load_state_dict(checkpoint["agent_state_dict"])

def play_policy(agent, env, logdir):
    # Load the Policy
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        # Ensure the agent does not pretrain again if it has already completed pretraining.
        agent._should_pretrain._once = False
    obs = env.reset()
    done = np.full(env.num_envs, True)
    obs['is_first'] = np.full(env.num_envs, True)
    obs['is_terminal'] = np.full(env.num_envs, False)
    while True:
        with torch.no_grad():
            action, _ = agent(obs, done, training=False)
        obs_dreamer, reward, done, info = env.step(action)
        # Put back the obs in orbit format since we just want to feed the agent and not log
        indices = []
        if done.any():
            indices = np.nonzero(done)[0]
            obs_dreamer = env.reset_idx(indices)
        obs = {k: np.stack([o[k] for o in obs_dreamer]) for k in obs_dreamer[0] if "log_" not in k}
        obs['is_first'] = np.full(env.num_envs, False)
        obs['is_terminal'] = np.full(env.num_envs, False)
        for i in indices:
            obs['is_first'][i] = True

        print('Hey')
        

def make_env_orbit():

    env_cfg: RLTaskEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)


    # Override env cfg
    env_cfg.reset.only_above_soil = False
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    #wrap it
    env = wrappers.OrbitNumpyExcavation(env)


    return env, env_cfg

def main(config):
    envs, envs_cfg = make_env_orbit()
    logdir = pathlib.Path(config.logdir).expanduser()
    # Initialize logging directory and announce its location.
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    #config.traindir.mkdir(parents=True, exist_ok=True)
    # Count the steps already present in the training directory to resume training appropriately.
    step = 1 # random
    logger = tools.Logger(logdir, config.action_repeat * step)
    dataset = None
    acts = Box(-1, 1, (envs.num_envs, envs.m545_measurements.num_dofs), 'float32')
    config.num_actions = acts.shape[1]
    agent = Dreamer(
        envs,
        config,
        logger,
        None,
    ).to(config.device)

    play_policy(agent, envs, logdir)


if __name__ == "__main__":
    try:
        # Bypass all the args parsing
        import yaml
        from argparse import Namespace
        # Function to convert a nested dictionary into a Namespace
        def dict_to_namespace(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = dict_to_namespace(value)
            return Namespace(**d)

        # Load the YAML file
        with open('configs.yaml', 'r') as file:
            parameters = yaml.safe_load(file)
        # Convert the dictionary to a Namespace object
        args_main = dict_to_namespace(parameters)
        main(args_main)
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()



