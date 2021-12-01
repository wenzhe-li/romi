import numpy as np
import torch
from utils.proportional import Experience


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(4e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.weight = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device

		self.state_mean = None
		self.state_std = None
	
	def standardizer(self, state):
		if self.state_mean is None:
			self.state_mean = np.mean(self.state, axis=0, keepdims=True)
			self.state_std = np.std(self.state, axis=0, keepdims=True) + 1e-3
			self.state_mean = torch.FloatTensor(self.state_mean).to(self.device)
			self.state_std = torch.FloatTensor(self.state_std).to(self.device)
		
		return (state - self.state_mean) / self.state_std

	def unstandardizer(self, state):
		return state * self.state_std + self.state_mean

	def add(self, state, action, next_state, reward, weight, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.weight[self.ptr] = weight
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.weight[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, state_dim, action_dim, device, max_size=int(4e6)):
		super(PrioritizedReplayBuffer, self).__init__(state_dim, action_dim, device, max_size=max_size)
		self.proportional = Experience(max_size, alpha=1.0)


	def add(self, state, action, next_state, reward, weight, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.weight[self.ptr] = weight
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		self.proportional.add(weight)


	def sample(self, batch_size):
		ind = self.proportional.select(batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.weight[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
