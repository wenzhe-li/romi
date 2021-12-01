import time
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import argparse, os
import numpy as np

import h5py
import gym
import d4rl

import continuous_bcq.ReverseBC as ReverseBC
import continuous_bcq.ForwardBC as ForwardBC
import continuous_bcq.replay_buffer
import continuous_bcq.BCQ as BCQ
import utils.dataset_processor as dataset_processor
import utils.model_utils as model_utils

import torch


def load_buffer(args):

    env = gym.make(args.env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    setting = "{}_{}_{}".format(args.env_name, args.seed, args.entropy_weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_rollout:
        if not args.is_forward_rollout:
            rbc_policy = ReverseBC.ReverseBC(state_dim, action_dim, max_action, device, args.entropy_weight)
            if not args.is_uniform_rollout:
                rbc_policy.load("reverse_bc_models/reverse_bc_{}_500000.0".format(setting))
                print("reverse_bc: loading reverse_bc_models/reverse_bc_{}_500000.0".format(setting))
        else:
            rbc_policy = ForwardBC.ForwardBC(state_dim, action_dim, max_action, device, args.entropy_weight)
            if not args.is_uniform_rollout:
                rbc_policy.load("forward_bc_models/forward_bc_{}_500000.0".format(setting))
                print("forward_bc: loading forward_bc_models/forward_bc_{}_500000.0".format(setting))

    if args.is_prioritized_buffer:
        dataset_replay_buffer = continuous_bcq.replay_buffer.PrioritizedReplayBuffer(state_dim, action_dim, ptu.device)
    else:
        dataset_replay_buffer = continuous_bcq.replay_buffer.ReplayBuffer(state_dim, action_dim, ptu.device)
    
    processor = dataset_processor.DatasetProcessor(args)
    dataset = processor.get_dataset()
    N = dataset['rewards'].shape[0]
    print('Loading buffer!')
    for i in range(N):
        obs = dataset['observations'][i]
        new_obs = dataset['next_observations'][i]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        weight = dataset['weights'][i]
        done_bool = bool(dataset['terminals'][i])
        dataset_replay_buffer.add(obs, action, new_obs, reward, weight, done_bool)
    print('Loaded buffer')

    idle_policy = BCQ.BCQ(state_dim, action_dim, max_action, ptu.device)

    if args.model_rollout:
        fake_env = model_utils.initialize_fake_env(env, args)
        model_replay_buffer = continuous_bcq.replay_buffer.ReplayBuffer(state_dim, action_dim, ptu.device)
        while model_replay_buffer.size < model_replay_buffer.max_size:
            model_utils.model_rollout(idle_policy, rbc_policy, fake_env, dataset_replay_buffer, model_replay_buffer, args,
                                      is_uniform=args.is_uniform_rollout)

    buffer = [(dataset_replay_buffer, 1.)]
    if args.model_rollout:
        buffer = [(dataset_replay_buffer, 1. - args.model_ratio), (model_replay_buffer, args.model_ratio)]
    
    return buffer

def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)  
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size

def experiment(variant, buffer=None):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env
    
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M], 
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    elif 'random-expert' in variant['env_name']:
        load_hdf5(d4rl.basic_dataset(eval_env), replay_buffer) 
    else:
        load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)
       
    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        buffer=buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=1500,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,  
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,   # Defaults to true
            lagrange_thresh=10.0,
            
            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--max_q_backup", type=str, default="False")          # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str, default="True")   # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=40000, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--min_q_weight', default=5.0, type=float)            # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy_lr', default=1e-4, type=float)              # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)               # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho)) 
    parser.add_argument('--lagrange_thresh', default=-1.0, type=float)         # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--seed', default=10, type=int)

    # parser.add_argument("--model_rollout", action="store_true")
    # parser.add_argument("--is_uniform_rollout", action="store_true")
    # parser.add_argument("--is_prioritized_buffer", action="store_true")
    parser.add_argument("--is_forward_rollout", action="store_true")

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    if args.lagrange_thresh < 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False
    
    variant['buffer_filename'] = None

    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed

    args.env_name = args.env
    args.task_name = args.env_name
    args.entropy_weight = 0.5
    args.rollout_batch_size = 1000
    args.weight_k = 0
    args.model_rollout = True

    if args.task_name[:7] == 'maze2d-' or args.task_name[:8] == 'antmaze-' or \
            args.task_name[:7] == 'hopper-' or args.task_name[:12] == 'halfcheetah-' or \
            args.task_name[:4] == 'ant-' or args.task_name[:9] == 'walker2d-':
        args.forward_model_load_path = 'mopo_models/' + args.task_name + '_forward_{}/'.format(args.seed)
        args.reverse_model_load_path = 'mopo_models/' + args.task_name + '_reverse_{}/'.format(args.seed)
    else:
        raise NotImplementedError

    if args.task_name[:6] == 'maze2d':
        args.domain = 'maze2d'
        args.test_model_length = 2
        args.rollout_length = 5
        if 'large' in args.task_name:
            args.model_ratio = 0.3
        elif 'medium' in args.task_name:
            args.model_ratio = 0.5
        elif 'umaze' in args.task_name:
            args.model_ratio = 0.7
        else:
            raise NotImplementedError
        args.is_uniform_rollout = True
        args.is_prioritized_buffer = False
    elif args.task_name[:7] == 'antmaze':
        args.domain = 'antmaze'
        args.test_model_length = 15
        args.rollout_length = 5
        args.model_ratio = 0.1
        args.is_uniform_rollout = False
        args.is_prioritized_buffer = True
    elif args.task_name[:11] == 'halfcheetah':
        args.domain = 'halfcheetah'
        args.test_model_length = 13
        args.rollout_length = 1
        args.model_ratio = 0.1
        args.is_uniform_rollout = False
        args.is_prioritized_buffer = True
    elif args.task_name[:6] == 'hopper':
        args.domain = 'hopper'
        args.test_model_length = 5
        args.rollout_length = 5
        args.model_ratio = 0.1
        args.is_uniform_rollout = True
        args.is_prioritized_buffer = True
    elif args.task_name[:8] == 'walker2d':
        args.domain = 'walker2d'
        args.test_model_length = 13
        args.rollout_length = 1
        args.model_ratio = 0.1
        args.is_uniform_rollout = False
        args.is_prioritized_buffer = True
    else:
        raise NotImplementedError
    
    args.save_id = str(time.time())
    print('save_id: ', args.save_id)
    args.save_path = args.task_name + '/' + args.save_id + '/'

    print('save_path: ', args.save_path)

    buffer = load_buffer(args)


    rnd = np.random.randint(0, 1000000)
    setup_logger(os.path.join('CQL_offline_mujoco_runs', str(rnd)), variant=variant, base_log_dir='./data')
    ptu.set_gpu_mode(True)
    experiment(variant, buffer=buffer)
