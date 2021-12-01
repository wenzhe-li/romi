import argparse
import gym
import numpy as np
import os
import torch
import sys

import d4rl
import uuid
import json

import time

import continuous_bcq.ReverseBC as ReverseBC
import continuous_bcq.ForwardBC as ForwardBC
import continuous_bcq.BCQ as BCQ
import continuous_bcq.replay_buffer as replay_buffer
import utils.model_utils as model_utils
import utils.dataset_processor as dataset_processor
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

# render BCQ offline
def render_BCQ(env, state_dim, action_dim, max_action, device, output_dir, args):
    filesystem.mkdir('./videos')

    # For saving files
    setting = f"{args.env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda,
                                    args.phi)
    policy.load(f"./{args.bcq_policy_dir}/BCQ_{setting}{args.load_stamp}")

    video_env = gym.make(args.env_name)
    video_env.seed(args.seed + 100)
    video_episodes = 10
    # video_env.render()
    for time in range(video_episodes):
        state, done = video_env.reset(), False
        avg_reward = 0.
        i = 0
        frames = []

        while not done:
            #print(state)

            frames.append(video_env.render(mode="rgb_array"))
            #video_env.mj_render()
            action = policy.select_action(np.array(state))
            #print(action)
            state, reward, done, _ = video_env.step(action)
            #print(reward)
            avg_reward += reward
            #time.sleep(1.0)
            i += 1
            # if i >= 500:
            #    break
        print(avg_reward)
        length = len(frames)
        print(length)
        if i >= 500:
            abstract = []
            step = int(length / 500) + 1
            for j in range(length):
                if j % step == 0:
                    abstract.append(frames[j])
        else:
            abstract = frames
        # if avg_reward > 0:
        # if i > 50:
        #     print(frames)
        print('saving video')
        save_frames_as_gif(abstract, path='./videos/', filename='{}_{}_{}.gif'.format(args.env_name, time, int(avg_reward)))
        print('video saved')


# Trains BCQ offline
def train_BCQ(env, state_dim, action_dim, max_action, device, output_dir, args):
    filesystem.mkdir(f"./{args.bcq_policy_dir}")
    pre_time = time.time()

    # For saving files
    setting = f"{args.env_name}_{args.seed}_{args.entropy_weight}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda,
                                    args.phi)

    # Load buffer
    if args.is_prioritized_buffer:
        dataset_replay_buffer = replay_buffer.PrioritizedReplayBuffer(state_dim, action_dim, device)
    else:
        dataset_replay_buffer = replay_buffer.ReplayBuffer(state_dim, action_dim, device)
    # dataset = env.get_dataset()
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
    # replay_buffer.load(f"./buffers/{buffer_name}")

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
    
    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0
    last_save = 0

    if args.model_rollout:
        fake_env = model_utils.initialize_fake_env(env, args)
        model_replay_buffer = replay_buffer.ReplayBuffer(state_dim, action_dim, device, max_size=int(4e6))
        while model_replay_buffer.size < model_replay_buffer.max_size:
            model_utils.model_rollout(policy, rbc_policy, fake_env, dataset_replay_buffer, model_replay_buffer, args,
                                      is_uniform=args.is_uniform_rollout)

    buffer = [(dataset_replay_buffer, 1.)]
    if args.model_rollout:
        buffer = [(dataset_replay_buffer, 1. - args.model_ratio), (model_replay_buffer, args.model_ratio)]
    
    while training_iters < args.max_timesteps:

        print('Train step:', training_iters)
        pol_vals = policy.train(buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env_name, args.seed, training_iters, args))
        np.save(os.path.join(output_dir, f"BCQ_{setting}"), evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}, used time: {time.time() - pre_time}")

        sys.stdout.flush()

        if args.train_bcq and training_iters - last_save > args.save_interval:
            policy.save(f"./{args.bcq_policy_dir}/BCQ_{setting}_{training_iters}")
            print(f"Save BCQ: {training_iters}")
            last_save = training_iters

    if args.train_bcq:
        policy.save(f"./{args.bcq_policy_dir}/BCQ_{setting}_{training_iters}")
        print(f"Save BCQ: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, training_iters, args, eval_episodes=10):
    filesystem.mkdir('./videos')

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    avg_length = 0.
    for i in range(eval_episodes):
        frames = []

        state, done = eval_env.reset(), False
        while not done:
            # if i == 0:
            #     frames.append(eval_env.render(mode="rgb_array"))

            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            avg_length += 1

        # if i == 0 and training_iters % 20000 == 0:
        #     video_len = len(frames)
        #     if video_len >= 100:
        #         abstract = []
        #         step = int(video_len / 100) + 1
        #         for j in range(video_len):
        #             if j % step == 0:
        #                 abstract.append(frames[j])
        #     else:
        #         abstract = frames
        #     print('saving video')
        #     save_frames_as_gif(abstract, path='./videos/',
        #                        filename='{}_{}_{}.gif'.format(env_name, training_iters, args.save_id))
        #     print('video saved')


    avg_reward /= eval_episodes
    avg_length /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, length: {avg_length:.3f}")
    print("---------------------------------------")
    return avg_reward


def render_traj(args):
    filesystem.mkdir('./videos')

    # For saving files
    setting = f"{args.env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # dataset = env.get_dataset()
    processor = dataset_processor.DatasetProcessor(args)
    dataset = processor.get_dataset(is_render_traj=True)
    # dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]

    eval_env = gym.make(args.env_name)
    frames = []

    initial_state = eval_env.reset()
    qpos = dataset['observations'][0][:args.test_model_length]
    qvel = dataset['observations'][0][args.test_model_length:]
    if args.test_padding > 0:
        qpos = np.concatenate([[0, ], qpos], axis=0)
    eval_env.env.set_state(qpos, qvel)
    error = []
    for i in range(N - 1):
        frames.append(eval_env.render(mode="rgb_array"))
        action = dataset['actions'][i]
        state, reward, done, _ = eval_env.step(action)
        error.append(np.linalg.norm(state - dataset['observations'][i + 1]))
        qpos = dataset['observations'][i + 1][:args.test_model_length]
        qvel = dataset['observations'][i + 1][args.test_model_length:]
        if args.test_padding > 0:
            qpos = np.concatenate([[0, ], qpos], axis=0)
        eval_env.env.set_state(qpos, qvel)
    print('simulation error:', np.mean(error))

    video_len = len(frames)
    if video_len >= 100:
        abstract = []
        step = int(video_len / 100) + 1
        for j in range(video_len):
            if j % step == 0:
                abstract.append(frames[j])
    else:
        abstract = frames
    print('saving video')
    save_frames_as_gif(abstract, path='./videos/',
                       filename='{}_{}_{}.gif'.format(args.env_name, "render_traj", time.time()))
    print('video saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-random-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")  # Prepends name to filename
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3,
                        type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3,
                        type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3,
                        type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)  # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--train_bcq", action="store_true")  # If true, train BCQ
    parser.add_argument("--save_interval", default=200000, type=int)  # Save interval
    parser.add_argument("--render_bcq", action="store_true")  # If true, record BCQ
    parser.add_argument("--load_stamp", default="_205000.0", type=str)  # load_stamp
    parser.add_argument("--bcq_policy_dir", default="bcq_models", type=str)

    parser.add_argument("--render_traj", action="store_true")
    # parser.add_argument("--model_rollout", action="store_true")
    # parser.add_argument("--is_uniform_rollout", action="store_true")
    # parser.add_argument("--is_prioritized_buffer", action="store_true")
    parser.add_argument("--is_forward_rollout", action="store_true")

    args = parser.parse_args()
    # d4rl.set_dataset_path('/datasets')

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

    if args.task_name[:6] == 'maze2d' or args.task_name[:7] == 'antmaze':
        args.test_padding = 0
    elif args.task_name[:6] == 'hopper' or args.task_name[:11] == 'halfcheetah' or args.task_name[:3] == 'ant' or args.task_name[:8] == 'walker2d':
        args.test_padding = 1
    else:
        raise NotImplementedError

    args.save_id = str(time.time())
    print('save_id: ', args.save_id)
    args.save_path = args.task_name + '/' + args.save_id + '/'

    print('save_path: ', args.save_path)

    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env_name}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env_name}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    results_dir = os.path.join(args.output_dir, 'BCQ', str(uuid.uuid4()))
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

    if args.render_bcq:
        render_BCQ(env, state_dim, action_dim, max_action, device, args.output_dir, args)
    elif args.render_traj:
        render_traj(args)
    else:
        train_BCQ(env, state_dim, action_dim, max_action, device, args.output_dir, args)
