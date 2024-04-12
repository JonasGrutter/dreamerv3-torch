import argparse
import functools
import os
import pathlib
import sys



import numpy as np
import ruamel.yaml as yaml
from collections import deque
import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy
from gym.spaces import Box
import statistics
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
#from omni.isaac.orbit_tasks.m545.excavation_utils.excavation_utils import multipage
import matplotlib.pyplot as plt


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
args_cli.headless = True
args_cli.num_envs = 4096
args_cli.task= 'Isaac-m545-v0'
EXCAVATION = False

LOGGER_TYPE = "Tensorboard" # "Wandb"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym
from omni.isaac.orbit.envs import RLTaskEnvCfg
import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from matplotlib.backends.backend_pdf import PdfPages


from dreamer_class import Dreamer

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format="pdf", dpi=dpi, bbox_inches="tight")
    pp.close()

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def load_agent(agent, path):
    checkpoint = torch.load(path)
    agent.load_state_dict(checkpoint["agent_state_dict"])

def benchmark_policy(agent, env, logdir, num_steps):
    # Load the Policy
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        # Ensure the agent does not pretrain again if it has already completed pretraining.
        agent._should_pretrain._once = False

    # Set up env buffers
    num_data = env.num_envs * num_steps
    bucket_aoa = torch.zeros(env.num_envs, num_steps, device=env.device)
    bucket_vel = torch.zeros(env.num_envs, num_steps, device=env.device)
    base_vel = torch.zeros(env.num_envs, num_steps, device=env.device)
    max_depth = torch.zeros(env.num_envs, num_steps, device=env.device)
    pullup_dist = torch.zeros(env.num_envs, num_steps, device=env.device)
    in_soil = torch.zeros(env.num_envs, num_steps, device=env.device)
    bucket_x = torch.zeros(env.num_envs, num_steps, device=env.device)
    bucket_z = torch.zeros(env.num_envs, num_steps, device=env.device)
    # need to keep track of it manually, because step() resets if done but we only log after step()
    ep_lens = torch.zeros(env.num_envs, device=env.device)
    ep_len_counts = {}
    ep_len_counts["timeout"] = deque()
    for name in env.unwrapped.termination_excavation.neg_term_names:
        ep_len_counts["neg_" + name] = deque()
    ep_len_counts["close"] = deque()
    ep_len_counts["full"] = deque()


    
    
    #--- Run Experiment
    episode_count = 0
    obs = env.reset()
    done = np.full(env.num_envs, True)
    obs['is_first'] = np.full(env.num_envs, True)
    obs['is_terminal'] = np.full(env.num_envs, False)
    for i in range(num_steps):
        if i% 20 == 0:
            print("steps{}/{}".format(i,num_steps))
        with torch.no_grad():
            action, _ = agent(obs, done, training=False)
        # Minimal logging
        bucket_aoa[:, i] = env.unwrapped.m545_measurements.bucket_aoa
        bucket_vel[:, i] = torch.linalg.norm(env.unwrapped.m545_measurements.bucket_vel_w, dim=-1)
        base_vel[:, i] = torch.linalg.norm(env.unwrapped.m545_measurements.root_lin_vel_w, dim=-1)
        bucket_z[:, i] = env.unwrapped.m545_measurements.bucket_pos_w[:, 2]
        max_depth[:, i] = env.unwrapped.soil.get_max_depth_height_at_pos(env.unwrapped.m545_measurements.bucket_pos_w[:, 0:1]).squeeze()
        pullup_dist[:, i] = env.unwrapped.pullup_dist
        bucket_x[:, i] = env.unwrapped.m545_measurements.bucket_pos_w[:, 0]
        in_soil[:, i] = (env.unwrapped.soil.get_bucket_depth() > 0.0).squeeze()
        
        # Step
        obs_dreamer, reward, done, info = env.step(action)
        ep_lens += 1
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
        # additional logging
        episode_count += len(indices)
        ep_len_counts["timeout"].extend(ep_lens[env.unwrapped.termination_excavation.time_out_buf].tolist())
        for name in env.unwrapped.termination_excavation.neg_term_names:
            ep_len_counts["neg_" + name].extend(ep_lens[env.unwrapped.termination_excavation.episode_neg_term_buf[name]].tolist())
        ep_len_counts["close"].extend(ep_lens[env.unwrapped.termination_excavation.close_pos_term_buf].tolist())
        ep_len_counts["full"].extend(ep_lens[env.unwrapped.termination_excavation.full_pos_term_buf].tolist())
    
    #-- Plotting
    # terminations, percentage [0,1]
    values = []
    labels = []
    for key, value in env.unwrapped.termination_excavation.episode_neg_term_counts.items():
        values.append(torch.sum(value).item() / episode_count)
        labels.append(key)

    sum_pos_term = 0
    for key, value in env.unwrapped.termination_excavation.episode_pos_term_counts.items():
        values.append(torch.sum(value).item() / episode_count)
        labels.append(key)
        sum_pos_term += int(torch.sum(value).item())

    full_term = torch.sum(env.unwrapped.termination_excavation.episode_pos_term_counts["desired_full"]).item() / episode_count
    close_term = torch.sum(env.unwrapped.termination_excavation.episode_pos_term_counts["desired_close"]).item() / episode_count


    values.append(torch.sum(env.unwrapped.termination_excavation.time_out_count).item() / episode_count)
    labels.append("timeout")

    _, ax = plt.subplots()
    ax.tick_params(axis="x", which="major", labelsize=6)
    ax.bar(np.arange(len(values)), values, tick_label=labels)
    ax.set_title(
        "close ({:.2f}) & full ({:.2f}) term/tot term: {} / {} [{:.2f}%]".format(
            close_term, full_term, sum_pos_term, episode_count, 100.0 * sum_pos_term / episode_count
        )
    )
    ax.grid()

    # stats violating negative termination conditions
    def log_and_print_stats(name, errs, num_data, error_dict):
        error_dict[name] = errs
        print(
            "{:<25} num/num_data: {:<10.2e} mean: {:<7.2f} std: {:<7.2f} min: {:<7.2f} max: {:<7.2f}".format(
                name,
                len(errs) / num_data,
                statistics.mean(errs) if len(errs) > 1 else np.nan,
                statistics.stdev(errs) if len(errs) > 1 else np.nan,
                min(errs) if len(errs) > 0 else np.nan,
                max(errs) if len(errs) > 0 else np.nan,
            )
        )


    print("num data samples: ", num_data)
    error_dict = {}

    # bucket aoa
    bad_aoa = bucket_aoa < 0.0
    fast_enough = bucket_vel > env.cfg.terminations_excavation.bucket_vel_aoa_threshold
    ids = torch.where(torch.logical_and(in_soil, torch.logical_and(bad_aoa, fast_enough)))
    errs = bucket_aoa[ids] - 0.0
    log_and_print_stats("bucket_aoa", errs.tolist(), num_data, error_dict)
    # bucket vel
    ids = torch.where(env.unwrapped.m545_measurements.bucket_vel_norm > env.cfg.terminations_excavation.max_bucket_vel)
    errs = env.unwrapped.m545_measurements.bucket_vel_norm[ids] - env.cfg.terminations_excavation.max_bucket_vel
    log_and_print_stats("bucket_vel", errs.tolist(), num_data, error_dict)

    # base vel
    ids = torch.where(base_vel > env.cfg.terminations_excavation.max_base_vel)
    errs = base_vel[ids] - env.cfg.terminations_excavation.max_base_vel
    log_and_print_stats("base_vel", errs.tolist(), num_data, error_dict)

    # max depth
    ids = torch.where(bucket_z < (max_depth - env.cfg.terminations_excavation.max_depth_overshoot))
    errs = bucket_z[ids] - (max_depth[ids] - env.cfg.terminations_excavation.max_depth_overshoot)
    log_and_print_stats("max_depth", errs.tolist(), num_data, error_dict)

    # pullup
    ids = torch.where(bucket_x < pullup_dist)
    errs = bucket_x[ids] - pullup_dist[ids]
    log_and_print_stats("pullup", errs.tolist(), num_data, error_dict)

    # episode lengths
    log_and_print_stats("len timeout", ep_len_counts["timeout"], num_data, error_dict)
    log_and_print_stats("len close", ep_len_counts["close"], num_data, error_dict)
    log_and_print_stats("len full", ep_len_counts["full"], num_data, error_dict)

    for name in env.unwrapped.termination_excavation.neg_term_names:
        log_and_print_stats("len neg_" + name, ep_len_counts["neg_" + name], num_data, error_dict)


    for key, value in error_dict.items():
        fig, ax = plt.subplots()
        ax.boxplot(value)
        ax.set_xticklabels([key], fontsize=6)
        quantiles = np.quantile(value, [0.25, 0.5, 0.75]) if len(value) > 1 else [0, 0, 0]
        ax.set_yticks(quantiles)
        ax.set_title(
            "q1-q3: {}, steps/total_steps: {} / {} [{:.2f}%]".format(
                " | ".join([str(np.round(x, 2)) for x in quantiles]), len(value), num_data, 100.0 * len(value) / num_data
            )
        )
        ax.grid()

    # save all plots in pdf and open
    filename = os.path.join("/home/jonas/Desktop", "stats.pdf")
    multipage(filename)
    os.system("xdg-open " + filename)
    plt.close("all")
        
    env.close()
        

def make_env_orbit():

    env_cfg: RLTaskEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)


    # Override env cfg
    env_cfg.reset.only_above_soil = True
    env_cfg.disable_negative_termination = True
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
    num_steps = 500
    benchmark_policy(agent, envs, logdir, num_steps=num_steps)
    envs.close()


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
        if EXCAVATION:
            with open('configs.yaml', 'r') as file:
                parameters = yaml.safe_load(file)
        else:
            with open('configs_dmc_proprio.yaml', 'r') as file:
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



