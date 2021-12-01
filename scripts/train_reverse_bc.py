import argparse
import sys
import gym
import numpy as np
import os
import torch

import d4rl
import uuid
import json

import time

import continuous_bcq.ReverseBC as ReverseBC
import continuous_bcq.ForwardBC as ForwardBC
import continuous_bcq.replay_buffer as replay_buffer
import utils.dataset_processor as dataset_processor
import utils.model_utils as model_utils
import utils.filesystem as filesystem
import utils.bc as bc

from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def train_reverse_bc(env, state_dim, action_dim, max_action, device, output_dir, args):
    filesystem.mkdir('./reverse_bc_models')
    filesystem.mkdir('./forward_bc_models')
    pre_time = time.time()

    # For saving files
    # setting = f"{args.env_name}_{args.seed}_{args.entropy_weight}"
    # buffer_name = f"{args.buffer_name}_{setting}"
    setting = "{}_{}_{}".format(args.env_name, args.seed, args.entropy_weight)
    buffer_name = "{}_{}".format(args.buffer_name, setting)


    # Initialize policy
    if not args.is_forward_rollout:
        policy = ReverseBC.ReverseBC(state_dim, action_dim, max_action, device, args.entropy_weight)
    else:
        policy = ForwardBC.ForwardBC(state_dim, action_dim, max_action, device, args.entropy_weight)
    # policy = bc.GaussianReverseBC(state_dim, action_dim, max_action, device, args.entropy_weight)

    # Load buffer
    if args.is_prioritized_reverse_bc:
        dataset_replay_buffer = replay_buffer.PrioritizedReplayBuffer(state_dim, action_dim, device)
    else:
        dataset_replay_buffer = replay_buffer.ReplayBuffer(state_dim, action_dim, device)
    # dataset = env.get_dataset()
    processor = dataset_processor.DatasetProcessor(args)
    dataset = processor.get_dataset(done_frame=False)
    # dataset = d4rl.qlearning_dataset(env)
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
    # replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0
    last_save = 0

    fake_env = model_utils.initialize_fake_env(env, args)
    true_env = model_utils.initialize_true_env(env, args)

    sys.stdout.flush()

    if not args.is_forward_rollout:
        policy_dir = "./reverse_bc_models/reverse_bc"
    else:
        policy_dir = "./forward_bc_models/forward_bc"

    while training_iters < args.max_timesteps:
        print('Train step:', training_iters)
        pol_vals = policy.train(dataset_replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        model_utils.test_model_accuracy(policy, fake_env, true_env, dataset_replay_buffer, args.rollout_batch_size,
                                        args.rollout_length, is_uniform=args.is_uniform_rollout)

        training_iters += args.eval_freq
        # print(f"Training iterations: {training_iters}, used time: {time.time() - pre_time}")

        sys.stdout.flush()

        if args.train_reverse_bc and training_iters - last_save > args.save_interval:
            # policy.save(f"{policy_dir}_{setting}_{training_iters}")
            policy.save("{}_{}_{}".format(policy_dir, setting, training_iters))
            # print(f"Save reverse_bc: {training_iters}")
            print("Save reverse_bc")
            last_save = training_iters

    if args.train_reverse_bc:
        # policy.save(f"{policy_dir}_{setting}_{training_iters}")
        policy.save("{}_{}_{}".format(policy_dir, setting, training_iters))
        # print(f"Save reverse_bc: {training_iters}")
        print("Save reverse_bc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-random-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")  # Prepends name to filename
    parser.add_argument("--eval_freq", default=5e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3,
                        type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3,
                        type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3,
                        type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)  # Mini batch size for networks
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--train_reverse_bc", action="store_true")  # If true, train RBC
    parser.add_argument("--save_interval", default=50000, type=int)  # Save interval
    parser.add_argument("--load_stamp", default="_205000.0", type=str)  # load_stamp

    # parser.add_argument("--is_uniform_rollout", action="store_true")
    # parser.add_argument("--is_prioritized_reverse_bc", action="store_true")
    parser.add_argument("--is_forward_rollout", action="store_true")

    args = parser.parse_args()
    # d4rl.set_dataset_path('/datasets')

    args.task_name = args.env_name
    args.is_uniform_rollout = False
    args.weight_k = 0
    args.is_prioritized_reverse_bc = False
    args.entropy_weight = 0.5
    args.rollout_batch_size = 1000
    args.rollout_length = 5


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
    elif args.task_name[:7] == 'antmaze':
        args.domain = 'antmaze'
        args.test_model_length = 15
    elif args.task_name[:11] == 'halfcheetah':
        args.domain = 'halfcheetah'
        args.test_model_length = 13
    elif args.task_name[:6] == 'hopper':
        args.domain = 'hopper'
        args.test_model_length = 5
    elif args.task_name[:8] == 'walker2d':
        args.domain = 'walker2d'
        args.test_model_length = 13
    elif args.task_name[:4] == 'ant-':
        args.domain = 'ant'
        args.test_model_length = 13
    else:
        raise NotImplementedError

    if args.task_name[:6] == 'maze2d' or args.task_name[:7] == 'antmaze':
        args.test_padding = 0
    elif args.task_name[:6] == 'hopper' or args.task_name[:11] == 'halfcheetah' or args.task_name[:3] == 'ant' or args.task_name[:8] == 'walker2d':
        args.test_padding = 1
    else:
        raise NotImplementedError

    args.save_path = args.task_name + '/' + str(time.time()) + '/'
    print('args.save_path: ', args.save_path)

    # print("---------------------------------------")
    # if args.train_behavioral:
    #     print(f"Setting: Training behavioral, Env: {args.env_name}, Seed: {args.seed}")
    # elif args.generate_buffer:
    #     print(f"Setting: Generating buffer, Env: {args.env_name}, Seed: {args.seed}")
    # else:
    #     print(f"Setting: Training reverse_bc, Env: {args.env_name}, Seed: {args.seed}")
    # print("---------------------------------------")

    results_dir = os.path.join(args.output_dir, 'reverse_bc', str(uuid.uuid4()))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'params.json'), 'w') as params_file:
        json.dump({
            'env_name': args.env_name,
            'seed': args.seed,
        }, params_file)

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_reverse_bc(env, state_dim, action_dim, max_action, device, args.output_dir, args)
