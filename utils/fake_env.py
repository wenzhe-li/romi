import numpy as np
import tensorflow as tf

class FakeEnv:
    def __init__(self, model, config, args,
                 is_use_reward=True,
                 is_use_oracle_reward=False,
                 is_fake_deterministic=False):
        self.model = model
        self.config = config
        self.args = args
        self.is_use_reward = is_use_reward
        self.is_use_oracle_reward = is_use_oracle_reward
        self.is_fake_deterministic = is_fake_deterministic

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''

    def step(self, obs, act):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if self.is_fake_deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        if not self.is_fake_deterministic:
            #### choose one model from ensemble
            num_models, batch_size, _ = ensemble_model_means.shape
            model_inds = self.model.random_inds(batch_size)
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
            ####
        else:
            samples = np.mean(ensemble_samples, axis=0)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        if not self.args.is_forward_rollout:
            terminals = self.config.termination_fn(next_obs, act, obs)
            if self.is_use_oracle_reward:
                rewards = self.config.reward_fn(next_obs, act, obs)
        else:
            terminals = self.config.termination_fn(obs, act, next_obs)
            if self.is_use_oracle_reward:
                rewards = self.config.reward_fn(obs, act, next_obs)

        penalized_rewards = rewards

        if return_single:
            next_obs = next_obs[0]
            penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]

        return next_obs, penalized_rewards, terminals, None
