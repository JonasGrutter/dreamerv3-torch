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
args_cli.num_envs = 1000
args_cli.task='Isaac-m545-v0'
EXCAVATION = False

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

to_np = lambda x: x.detach().cpu().numpy()

# Dreamer class implements the Dreamer agent model, integrating a world model and behavior models for decision making.
class Dreamer(nn.Module):
    def __init__(self, envs, config, logger, dataset):
        super(Dreamer, self).__init__()
        # Environment storing       
        self.envs = envs
        if EXCAVATION:
            self.act_space = Box(-1, 1, (self.envs.num_envs, 4), 'float32')
            self.obs_space = self.envs.observation_space
        else:
            self.act_space =  self.envs.envs[0].action_space
            self.obs_space = self.envs.envs[0].observation_space
        # Store the configuration settings, logger, and dataset for use within the class.
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        # Determine how often training should occur based on configuration settings.
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        # Condition for determining when to reset the environment or simulation.
        self._should_reset = tools.Every(config.reset_every)
        # Controls exploration until a certain number of actions have been taken.
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # Normalize the step count based on the action repeat setting in the config.
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        # Initialize the world model with the observation space, action space, current step, and configuration.
        self._wm = models.WorldModel(self.obs_space, self.act_space, self._step, config)
        # Initialize the behavior model for task-specific actions using the world model.
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        # Define the exploration behavior based on the configuration. This can be greedy, random, or plan2explore.
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        # Dynamically select the exploration behavior based on the configuration setting.
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, self.act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        #print('call function called')
        step = self._step
        if training:
            print('training')
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):#500
                self._train(next(self._dataset)) # iterator on next element of data set
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor.dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        # World Model Training
        metrics = {}
        #print("Train world model")
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        # Actor Critic Learning
        #print("Train Actor Critic")
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

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
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)

        env = wrappers.TimeLimit(env, config.time_limit)
        #env = wrappers.SelectAction(env, key="action")
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
    env_cfg.reset.only_above_soil = True
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
    env = wrappers.OrbitNumpy(env)


    return env

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
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)
    # Load episodes for training and evaluation, potentially from specified offline directories.
    print("Create envs.")
    # Determine the directory for loading training episodes, favoring an offline directory if specified.
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    # Load training episodes from the specified directory, up to the configured dataset size limit.
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir

    # -- Orbit
    if EXCAVATION:
        train_envs = make_env_orbit()
        acts = Box(-1, 1, (train_envs.num_envs, 4), 'float32')
    else:
        train_envs = make_envs_dmc(config)
        acts = train_envs.envs[0].action_space

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
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low),#.repeat(config.envs, 1),
                    torch.Tensor(acts.high),#.repeat(config.envs, 1),
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

    train_eps

    # Initialize the Dreamer agent with the specified configurations and datasets.
    agent = Dreamer(
        train_envs,
        config,
        logger,
        train_dataset,
    ).to(train_envs.device)
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
        print("Agent step in outside loop:", agent._step)

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
        # Enhance Data Set
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")


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
        #args_main = Namespace(act='SiLU', action_repeat=2, actor={'layers': 2, 'dist': 'normal', 'entropy': 0.0003, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0, 'temp': 0.1, 'lr': 3e-05, 'eps': 1e-05, 'grad_clip': 100.0, 'outscale': 1.0}, batch_length=64, batch_size=16, compile=True, cont_head={'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0}, critic={'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-05, 'eps': 1e-05, 'grad_clip': 100.0, 'outscale': 0.0}, dataset_size=1000000, debug=False, decoder={'mlp_keys': '.*', 'cnn_keys': '$^', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0}, deterministic_run=False, device='cuda:0', disag_action_cond=False, disag_layers=4, disag_log=True, disag_models=10, disag_offset=1, disag_target='stoch', disag_units=400, discount=0.997, discount_lambda=0.95, dyn_deter=512, dyn_discrete=32, dyn_hidden=512, dyn_mean_act='none', dyn_min_std=0.1, dyn_rec_depth=1, dyn_scale=0.5, dyn_std_act='sigmoid2', dyn_stoch=32, encoder={'mlp_keys': '.*', 'cnn_keys': '$^', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True}, eval_episode_num=10, eval_every=10000.0, eval_state_mean=False, evaldir=None, expl_behavior='greedy', expl_extr_scale=0.0, expl_intr_scale=1.0, expl_until=0, grad_clip=1000, grad_heads=('decoder', 'reward', 'cont'), grayscale=False, imag_gradient='dynamics', imag_gradient_mix=0.0, imag_horizon=15, initial='learned', kl_free=1.0, log_every=10000.0, logdir='./logdir/dmc_walker_walk', model_lr=0.0001, norm=True, offline_evaldir='', offline_traindir='', opt='adam', opt_eps=1e-08, parallel=False, precision=32, prefill=2500, pretrain=100, rep_scale=0.1, reset_every=0, reward_EMA=True, reward_head={'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0}, seed=0, size=(64, 64), steps=500000.0, task='dmc_walker_walk', time_limit=1000, train_ratio=512, traindir=None, unimix_ratio=0.01, units=512, video_pred_log=False, weight_decay=0.0)

        main(args_main)
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        if EXCAVATION:
            simulation_app.close()