import datetime
import gym
import numpy as np
import uuid

import torch

to_np = lambda x: x.detach().cpu().numpy()


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)

class LikeOrbitNumpyDMC():

    def __init__(self, env):
        """
            Receives a collection of environment
        """
        # Identify each env uniquely
        self.unique_indices = []
        for i in range(len(env)):
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            id = f"{timestamp}-{str(uuid.uuid4().hex)}"
            self.unique_indices.append(id)

        self.envs = env
        self.num_envs = len(env)

    def step(self, action):
        results = [e.step(a) for e, a in zip(self.envs, action)]
        obs, reward, done,infos = zip(*[p[:4] for p in results])
        return obs, reward, done, infos 
    
    def reset(self):
        for i in range(len(self.num_envs)):
            self.envs[i].reset()
        
    def reset_idx(self, indices):
        obs = [self.envs[i].reset() for i in indices] # Reset and get initial observations.
        obs = [r() for r in obs] # Assume these are async calls, so we get the results.
        # We need the full set of obs here
        for ids in indices:
            obs[ids]['is_first'] = True
            obs[ids]['is_terminal'] = False
        return obs


class OrbitNumpy(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Identify each env uniquely
        self.unique_indices = []
        for i in range(env.unwrapped.num_envs):
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            id = f"{timestamp}-{str(uuid.uuid4().hex)}"
            self.unique_indices.append(id)

        


    def step(self, action):
        # Transform Action to torch tensor
        action_torch = action.clone().detach().to(self.env.unwrapped.device)
        # Orbit Env Stepping
        obs_dict, rew, terminated, truncated, extras = self.env.step(action_torch)
        # rew: torch tensor, terminated: torch tensor, extras['log']: float but extras 
        
        # compute dones 
        dones = (terminated | truncated)
        # move extra observations to the extras dict

        extras["observations"] = obs_dict # TODO: Why ?
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = to_np(truncated)
        # Convert to numpy
        # Concvert obs to the dreamer format
        obs_np = [
            {obs_key: obs_dict[obs_key][env_idx].cpu().numpy() for obs_key in obs_dict} # only key is policy anyway
            for env_idx in range(self.env.unwrapped.num_envs)
        ]

        rew_np =  to_np(rew)
        done_np = to_np(dones)
        extras_np = {}
        extras_np['episode_neg_term_counts'] = {}
        extras_np['episode_pos_term_counts'] = {}

        for key in extras['episode_neg_term_counts']:
            extras_np['episode_neg_term_counts'][key] = to_np(extras['episode_neg_term_counts'][key])

        for key in extras['episode_pos_term_counts']:
            extras_np['episode_neg_term_counts'][key] = to_np(extras['episode_pos_term_counts'][key])


        return obs_np, rew_np, done_np, extras_np
    
    def reset(self):
        
        # Get obs
        obs_dict, _ = self.env.reset()

        # Generate new indices
        for i in range(self.env.num_envs):
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            id = f"{timestamp}-{str(uuid.uuid4().hex)}"
            self.unique_indices.append(id)

        return obs_dict
    
    def reset_idx(self, indices):
        '''
            Reset env for given indices and return obs in dreamer format
        '''
        reset_env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.env.unwrapped.reset_idx(reset_env_ids)

        # Update derived measurements for resetted envs
        self.env.update_derived_measurements(reset_env_ids)
        # Update the obs at first, gives full set of obs
        self.env.obs_buf = self.env.unwrapped.observation_manager.compute()
        self.env.last_actions[reset_env_ids] = 0

        obs_dreamer = [
            {obs_key:  self.env.obs_buf[obs_key][env_idx].cpu().numpy() for obs_key in  self.env.obs_buf} # only key is policy anyway
            for env_idx in range(self.env.num_envs)
        ]

        # complete the obs with is_first and is_terminal
        for ids in range(self.env.num_envs): # TODO: Remove when DMC comissioned
            # Cache Resetted state and ger
            if ids in indices:
                # Adapt obs of new resetted env
                obs_dreamer[ids]['is_first'] = True
                obs_dreamer[ids]['is_terminal'] = False
                # Give the resetted env a new unique index
            else:
                # Adapt obs for not resetted env since is_first and is_terminal have been remobved in envs.unwrapped.observation_manager.compute()
                obs_dreamer[ids]['is_first'] = False
                obs_dreamer[ids]['is_terminal'] = False

        return obs_dreamer

class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()
