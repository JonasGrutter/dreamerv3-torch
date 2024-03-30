import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    # Initialization of the world model with observation space, action space, current step, and configuration settings.
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        # Store current step and whether to use automatic mixed precision (AMP) based on config.
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        # Preprocess observation space to get the shapes for encoder input.
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        # Initialize the encoder to process observations into a lower-dimensional embedding.
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        # Output dimension of the encoder, used to size the input for other components.
        self.embed_size = self.encoder.outdim
        # Initialize the RSSM dynamics model with configuration parameters.
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        # Module dictionary to hold various prediction heads (e.g., decoder, reward predictor).
        self.heads = nn.ModuleDict()
        # Compute feature size for the heads based on discrete and deterministic components.
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # Initialize decoder and reward prediction head with the computed feature size.
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        # Continuation prediction head to estimate whether the current state is terminal.    
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        # Ensure all specified gradient heads are defined.
        for name in config.grad_heads:
            assert name in self.heads, name
        # Initialize the optimizer for the world model components.
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # Loss scaling factors for reward and continuation heads.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    # Train the world model on a batch of data.
    # It uses gradient descent to minimize a composite loss function that includes:
    # - The KL divergence between predicted and actual next states,
    # - Prediction errors for auxiliary tasks (e.g., reward prediction),
    # - Any additional specified losses (e.g., for continuous control).
    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        # Preprocess the input data (e.g., normalize images, prepare discounts). TRANSFORM AS TORCH TENSOR
        data = self.preprocess(data)

        # Enable gradient computation and automatic mixed precision if configured.
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Encode the observations to get embeddings.
                embed = self.encoder(data)
                # Using the embedded observations and actions, update the internal state of the dynamics model.
                # This produces "posterior" predictions of the next state given the current state and action,
                # as well as a "prior" prediction based on the model's internal dynamics.
                # post: actual next states, prior: predicted by the model
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                # Calculate the KL loss between posterior and prior, along with dynamic and representation losses.
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                # Initialize a dictionary to hold predictions from various model heads (e.g., reward prediction).
                preds = {}
        
                # For each head in the model (e.g., decoder, reward), generate predictions.
                # If a head is marked for gradient computation, use the learned features; otherwise, detach them to stop gradients.

                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                # Compute loss for each prediction head, typically based on how well the model's predictions match the observed data.
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                # Apply specified scaling factors to each loss component before summing them up with the KL divergence loss.
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            # Use the composite model loss to update model parameters via backpropagation.
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
    # Also include entropy metrics for the prior and posterior distributions as an indication of model certainty.
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        
        # Initialize actor network with the calculated feature size and configuration parameters
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        # Initialize value network with the calculated feature size and configuration parameters
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        # Initialize optimizers for actor and value networks with specified parameters and weight decay
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        # Log the number of variables in the actor optimizer
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    # Train the imagined behavior model
    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # Generate imagined trajectories using the world model and actor policy
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                # Calculate rewards for the imagined trajectories using the objective function
                reward = objective(imag_feat, imag_state, imag_action)
                # Calculate entropy for the actor policy and world model dynamics
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # Compute the target values for policy optimization
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                # Compute the loss for the actor policy
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                # Actor update
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                # estimate the value for the imagined states 
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                #  discrepancy between the predicted values by the value network and the computed targets,
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    # Generates imagined trajectories over a specified horizon using the world model and policy.
    def _imagine(self, start, policy, horizon):
        # Access the dynamics model from the world model for simulating state transitions.
        dynamics = self._world_model.dynamics
        # Flatten function to reshape state tensors for processing.
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # Apply flatten to each tensor in the start dictionary to prepare for simulation.
        start = {k: flatten(v) for k, v in start.items()}

        # Define a step function to simulate one time step using the dynamics model and policy.
        def step(prev, _):
            state, _, _ = prev
            # Get the current feature representation of the state.
            feat = dynamics.get_feat(state)
            # Detach feature tensor to prevent gradients from flowing into the dynamics model.
            inp = feat.detach()
            # Sample an action from the policy based on the current state feature.
            action = policy(inp).sample()
            # Predict the next state using the dynamics model and the sampled action.
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        # Use tools.static_scan to apply the step function across the specified horizon, 
        # effectively unrolling the dynamics model to generate imagined trajectories.
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        # Reconstruct the sequence of states from the successive predictions.
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    # Computes target values for optimization using the rewards from imagined trajectories.
    # Value Function Learning: The target value represents the expected return (cumulative discounted rewards) from a state or state-action pair. 
    def _compute_target(self, imag_feat, imag_state, reward):
        # If the model includes a continuation head ("cont"), use it to compute the discount factor.
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            # The continuation prediction modifies the discount factor based on the likelihood of continuing.
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            # Otherwise, use a constant discount factor as specified in the config.
            discount = self._config.discount * torch.ones_like(reward)
        # Compute the value estimates for each imagined state feature.
        value = self.value(imag_feat).mode()
        # Calculate the target values (returns) using the lambda-return formula,
        # which blends immediate rewards with future value estimates according to the discount factor.
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        # Compute cumulative product of discounts to weight the importance of each timestep's return.
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,     # The features of imagined (simulated) states.
        imag_action,   # Actions taken in the imagined states.
        target,        # The computed target values for optimization.
        weights,       # Weights applied to each timestep in the loss calculation.
        base,          # The baseline value used for calculating advantages.
    ):
        metrics = {}   # Dictionary to store various metrics for monitoring.
        inp = imag_feat.detach()  # Detach imagined features to prevent gradients from flowing back.
        policy = self.actor(inp)  # Get the policy distribution based on the current state.

        # Stack the target values along a new dimension.
        target = torch.stack(target, dim=1)

        # If using Exponential Moving Average (EMA) for reward normalization:
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)  # Calculate the offset and scale for normalization.
            normed_target = (target - offset) / scale  # Normalize the targets.
            normed_base = (base - offset) / scale  # Normalize the baseline.
            adv = normed_target - normed_base  # Calculate the advantage.
            # Update metrics with the normalized target values and EMA constants.
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        # Calculate the actor target based on the chosen gradient calculation method.
        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
        elif self._config.imag_gradient == "both":
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
            mix = self._config.imag_gradient_mix  # Mixing coefficient for combining gradients.
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)

        # Calculate the final actor loss by applying weights and summing over timesteps.
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics


    def _update_slow_target(self):
        # Check if the slow target update mechanism is enabled.
        if self._config.critic["slow_target"]:
            # Perform an update only at specified intervals.
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]  # The mixing ratio for the slow target update.
                # Update the slow target parameters by mixing them with the main network parameters.
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1  # Increment the update counter.