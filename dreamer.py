import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

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


from dreamer_class import Dreamer, EXCAVATION

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
args_cli.num_envs = 100
args_cli.task= 'Isaac-m545-v0'

if EXCAVATION:
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from omni.isaac.orbit.envs import RLTaskEnvCfg
    import omni.isaac.contrib_tasks  # noqa: F401
    import omni.isaac.orbit_tasks  # noqa: F401
    from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper



# For Pre training
from rsl_rl.runners import OnPolicyRunner
'''if ORBIT:
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from datetime import datetime
    import gymnasium as gym
    from omni.isaac.orbit.envs import RLTaskEnvCfg
    import omni.isaac.contrib_tasks  # noqa: F401
    import omni.isaac.orbit_tasks  # noqa: F401
    from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    args_cli.task = 'Isaac-m545-v0'

    # For Pre training
    from rsl_rl.runners import OnPolicyRunner'''



#----- ORBIT-End


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset



def make_envs_dmc(config):
    def make_1_env_dmc(config, id):
        suite, task = config.task.split("_", 1)
        import envs.dmc as dmc
        env = dmc.DeepMindControl(
            task, config.action_repeat, tuple(config.size), seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)

        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        #env = wrappers.UUID(env) 

        return env
    
    make = lambda id: make_1_env_dmc(config, id)
    envs = [make(i) for i in range(args_cli.num_envs)]
    
    envs = [Damy(env) for env in envs]

    envs = wrappers.LikeOrbitNumpyDMC(envs)


    return envs


#----- ORBIT-Start

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

    #----- ORBIT-End

# Main function to configure and execute the training of the Dreamer model.
def main(config):
    # Set a fixed random seed for reproducibility across runs.
    tools.set_seed_everywhere(config.seed)
    # Enable deterministic operations for reproducibility if specified in config.
    if config.deterministic_run:
        tools.enable_deterministic_run()

    # Prepare logging directories for training and evaluation data.
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    # Adjust various config settings based on the action repeat parameter for consistent timing.
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # Initialize logging directory and announce its location.
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    # Count the steps already present in the training directory to resume training appropriately.
    step = count_steps(config.traindir)



    # Load episodes for training and evaluation, potentially from specified offline directories.
    print("Create envs.")
    # Determine the directory for loading training episodes, favoring an offline directory if specified.
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    # Load training episodes from the specified directory, up to the configured dataset size limit.
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)

    # -- Orbit
    if EXCAVATION:
        train_envs, train_envs_cfg = make_env_orbit()
        acts = Box(-1, 1, (train_envs.num_envs, train_envs.m545_measurements.num_dofs), 'float32')
    else:
        train_envs = make_envs_dmc(config)
        acts = train_envs.envs[0].action_space
    
    
    #logger = tools.Logger(logdir, config.action_repeat * step)
    logger = tools.WandbLogger(logdir, config, config.action_repeat * step)
    
    # Reset
    train_envs.reset()
    # -- Orbit
   # Box(-1.0, 1.0, (train_envs.m545_measurements.num_dofs,), 'float32')
    # Determine the action space from the first training environment and log it.
    print("Action Space", acts)
    # Set the number of actions in the configuration based on the determined action space.
    if EXCAVATION:
        config.num_actions = acts.shape[1]
    else:
        config.num_actions = acts.shape[0]
    # Initialize the state for potentially pre-filling the replay buffer.
    state = None
    # If not using an offline training directory, calculate how much pre-filling is needed based on existing data.
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        # Determine the type of random actor based on the action space being discrete or continuous.
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            if EXCAVATION:
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.Tensor(acts.low),#.repeat(config.envs, 1),
                        torch.Tensor(acts.high),#.repeat(config.envs, 1),
                    ),
                    1,
                )
            else:
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.Tensor(acts.low).repeat(train_envs.num_envs, 1),
                        torch.Tensor(acts.high).repeat(train_envs.num_envs, 1),
                    ),
                    1,
                )
        # Define a random agent that samples actions uniformly and calculates their log probability.
        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        # Define a zero agent that provides 0 actions
        def zero_agent(o, d, s):
            action = torch.zeros([train_envs.num_envs,train_envs.m545_measurements.num_dofs])
            logprob = torch.zeros([train_envs.num_envs])
            return {"action": action, "logprob": logprob}, None
        
        
        # Define a pretrained agent using PPO to learn the world model better
        def pretrained_agent(o, d, s):
            agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
            # specify directory for logging experiments
            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            # load previously trained model
            ppo_runner = OnPolicyRunner(train_envs, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            ppo_runner.load(resume_path)
            policy = ppo_runner.get_inference_policy(device=train_envs.unwrapped.device)
            action = policy(o)
            logprob = random_actor.log_prob(action)
            
            return {"action": action, "logprob": logprob}, None
        



        # Simulate the random agent in the training environments to pre-fill the dataset, logging progress.
        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        # Update the logger with the number of steps simulated during pre-filling.
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")
    
    # Start of the agent simulation process.
    print("Simulate agent.")
    # Create training and evaluation datasets from loaded episodes.
    train_dataset = make_dataset(train_eps, config)

    # Initialize the Dreamer agent with the specified configurations and datasets.
    agent = Dreamer(
        train_envs,
        config,
        logger,
        train_dataset
    ).to(config.device)
    # Disable gradients for the entire agent model to freeze its parameters during certain operations.
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        # Ensure the agent does not pretrain again if it has already completed pretraining.
        agent._should_pretrain._once = False

    #---------------- TRAINING LOOP ---------------#

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        # If the configuration specifies evaluation episodes, proceed with evaluation.
        print("Agent step in outside loop:", agent._step, "/", config.steps+config.eval_every)

        print("Start training.")
        # Simulate the agent in training mode, updating its parameters based on interactions with the environment.
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        # Save model
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, os.path.join(logdir, f"model_{agent._step}.pt"))
    
    train_envs.close()


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
        if EXCAVATION:
            simulation_app.close()