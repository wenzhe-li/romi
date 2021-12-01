import numpy as np
from mopo.models.constructor import construct_model, format_samples_for_training, format_reverse_samples_for_training
import mopo.static # TODO: add static functions
from numpy.linalg import norm
from functools import partial
from utils.fake_env import FakeEnv
from utils.oracle_env import OracleEnv
import utils.dataset_utils as dataset_utils
import d4rl
import copy
import sys


def model_rollout(policy, rbc_policy, fake_env, dataset_replay_buffer, replay_buffer, args, is_uniform=False):
    state, action, next_state, reward, weight, not_done = dataset_replay_buffer.sample(args.rollout_batch_size)
    if not args.is_forward_rollout:
        next_observations = state.cpu().data.numpy()
        weights = weight.cpu().data.numpy()
        rollout_batch_size = args.rollout_batch_size
        for i in range(args.rollout_length):
            if is_uniform:
                action_size = (rollout_batch_size, policy.action_dim)
                actions = np.random.uniform(low=-1, high=1, size=action_size)
            else:
                actions = rbc_policy.select_action(next_observations)
            observations, rewards, dones, _ = fake_env.step(next_observations, actions)
            for j in range(rollout_batch_size):
                replay_buffer.add(observations[j], actions[j], next_observations[j], rewards[j], weights[j], dones[j])

            non_dones = ~dones.squeeze(-1)
            if non_dones.sum() == 0:
                print("Model rollout break early")
                break

            next_observations = observations[non_dones]
            weights = weights[non_dones]
            rollout_batch_size = next_observations.shape[0]
    else:
        observations = state.cpu().data.numpy()
        weights = weight.cpu().data.numpy()
        rollout_batch_size = args.rollout_batch_size
        for i in range(args.rollout_length):
            if is_uniform:
                action_size = (rollout_batch_size, policy.action_dim)
                actions = np.random.uniform(low=-1, high=1, size=action_size)
            else:
                actions = rbc_policy.select_action(observations)
            next_observations, rewards, dones, _ = fake_env.step(observations, actions)
            for j in range(rollout_batch_size):
                replay_buffer.add(observations[j], actions[j], next_observations[j], rewards[j], weights[j], dones[j])

            non_dones = ~dones.squeeze(-1)
            if non_dones.sum() == 0:
                print("Model rollout break early")
                break

            observations = next_observations[non_dones]
            weights = weights[non_dones]
            rollout_batch_size = observations.shape[0]


def initialize_fake_env(env, args):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    model_type = "mlp"
    hidden_dim = 200
    num_networks = 7
    num_elites = 5
    separate_mean_var = True
    model_name = None
    if args.is_forward_rollout:
        load_dir = args.forward_model_load_path
    else:
        load_dir = args.reverse_model_load_path
    deterministic = False
    model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                            num_networks=num_networks, num_elites=num_elites,
                            model_type=model_type, separate_mean_var=separate_mean_var,
                            name=model_name, load_dir=load_dir, deterministic=deterministic)
    domain = args.domain
    static_fns = mopo.static[domain.lower()]
    if domain is 'antmaze':
        static_fns.termination_fn = partial(static_fns.termination_fn, env=env)
    fake_env = FakeEnv(model, static_fns, args)

    if 'antmaze' in args.env_name:
        samples = dataset_utils.processed_qlearning_dataset(env, args.env_name, timeout_frame=False, done_frame=True)
    else:
        samples = dataset_utils.processed_qlearning_dataset(env, args.env_name, timeout_frame=False, done_frame=False)
    samples['rewards'] = np.expand_dims(samples['rewards'], axis=1)
    samples['terminals'] = np.expand_dims(samples['terminals'], axis=1)
    if not args.is_forward_rollout:
        inputs, outputs = format_reverse_samples_for_training(samples)
    else:
        inputs, outputs = format_samples_for_training(samples)

    model.load(inputs, outputs, holdout_ratio=0.2)

    return fake_env


def initialize_true_env(env, args):
    true_env = OracleEnv(env, args.test_model_length, args.test_padding)
    return true_env


def test_model_accuracy(policy, fake_env, true_env, dataset_replay_buffer, rollout_batch_size, rollout_length, is_uniform=False):
    state, action, next_state, reward, weight, not_done = dataset_replay_buffer.sample(rollout_batch_size)
    for test_length in range(1, rollout_length + 1):
        next_obs = copy.deepcopy(state.cpu().data.numpy())
        act_list = []
        next_obs_list = [next_obs]
        rew_list = []
        rollout_batch_size = next_obs.shape[0]
        try:
            for i in range(test_length):
                if is_uniform:
                    action_size = (rollout_batch_size, policy.action_dim)
                    act = np.random.uniform(low=-1, high=1, size=action_size)
                else:
                    act = policy.select_action(next_obs)

                next_obs, rew, dones, _ = fake_env.step(next_obs, act)
                non_dones = ~dones.squeeze(-1)
                for j in range(len(next_obs_list)):
                    next_obs_list[j] = next_obs_list[j][non_dones]
                for j in range(len(rew_list)):
                    rew_list[j] = rew_list[j][non_dones]
                for j in range(len(act_list)):
                    act_list[j] = act_list[j][non_dones]

                next_obs = next_obs[non_dones]
                rew = rew[non_dones]
                act = act[non_dones]
                next_obs_list.append(next_obs)
                rew_list.append(rew)
                act_list.append(act)
                
                rollout_batch_size = next_obs.shape[0]

            obs = copy.deepcopy(next_obs)
            for i in range(test_length - 1, -1, -1):
                
                act = act_list[i]
                obs_fake = next_obs_list[i]
                rew_fake = rew_list[i]
                obs, rew, dones, _ = true_env.step(obs, act)

            obs_diff = np.mean(np.mean(np.square(obs - obs_fake), axis=-1), axis=-1)
            rew_diff = np.mean(np.mean(np.square(rew - rew_fake), axis=-1), axis=-1)
            print('[ Test Model ] Rollout {} | Obs: {} | Rew: {}'.format(test_length, obs_diff, rew_diff))
            sys.stdout.flush()

        except:
            break
