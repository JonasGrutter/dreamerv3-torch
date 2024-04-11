import os
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

EXCAVATION = True

to_np = lambda x: x.detach().cpu().numpy()

# Dreamer class implements the Dreamer agent model, integrating a world model and behavior models for decision making.
class Dreamer(nn.Module):
    def __init__(self, envs, config, logger, dataset):
        super(Dreamer, self).__init__()
        # Environment storing       
        self.envs = envs
        if EXCAVATION:
            self.act_space = Box(-1, 1, (self.envs.num_envs, self.envs.m545_measurements.num_dofs), 'float32')
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
            #print('training')
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
