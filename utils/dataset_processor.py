import d4rl
import gym
import argparse
import numpy as np
from numpy.linalg import norm
import utils.dataset_utils as dataset_utils
import collections
import copy as cp


class DatasetProcessor:

    def __init__(self, args):
        self.args = args

    def get_dataset(self, is_render_traj=False, done_frame=True):
        env_name = self.args.env_name
        env = gym.make(env_name)

        if 'antmaze' in env_name:
            traj_list = dataset_utils.processed_sequence_dataset(env, env_name, timeout_frame=False, done_frame=True)
        else:
            traj_list = dataset_utils.processed_sequence_dataset(env, env_name, timeout_frame=False, done_frame=done_frame)

        self.traj_end = []
        self.data = collections.defaultdict(list)

        for traj in traj_list:
            num_tuple = traj['rewards'].shape[0]
            traj['weights'] = cp.deepcopy(traj['rewards'])
            sum_return = np.sum(traj['rewards'])
            traj['weights'][:] = sum_return

            if is_render_traj:
                if sum_return > 0. and num_tuple > 100:
                    for item in traj.keys():
                        self.data[item].append(traj[item][::-1])
                    break
            else:
                for item in traj.keys():
                    self.data[item].append(traj[item])

        for item in self.data.keys():
            self.data[item] = np.concatenate(self.data[item], axis=0)

        print('min_return: ', np.min(self.data['weights']))
        print('max_return: ', np.max(self.data['weights']))
        self.data['weights'] -= np.min(self.data['weights'])
        max_return = max(1., np.max(self.data['weights']))
        self.data['weights'] /= max_return
        self.data['weights'] = (self.args.weight_k + self.data['weights']) / (self.args.weight_k + 1.0)
        print('weights and numbers: ', np.sum(self.data['weights']), self.data['weights'].shape[0])

        return self.data
