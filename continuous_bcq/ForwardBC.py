import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BCQ import VAE


class ForwardBC:
	def __init__(self, state_dim, action_dim, max_action, device, entropy_weight=0.5):
		latent_dim = action_dim * 2

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

		self.max_action = max_action
		self.action_dim = action_dim
		self.device = device

		self.entropy_weight = entropy_weight

	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			action = self.vae.decode(state)
		return action.cpu().data.numpy()

	def train(self, replay_buffer, iterations, batch_size=100):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, weight, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + self.entropy_weight * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

	def save(self, filename):
		torch.save(self.vae.state_dict(), filename + "_FBC_vae")
		torch.save(self.vae_optimizer.state_dict(), filename + "_FBC_vae_optimizer")

	def load(self, filename):
		if not torch.cuda.is_available():
			self.vae.load_state_dict(torch.load(filename + "_FBC_vae", map_location=torch.device('cpu')))
			self.vae_optimizer.load_state_dict(torch.load(filename + "_FBC_vae_optimizer", map_location=torch.device('cpu')))
		else:
			self.vae.load_state_dict(torch.load(filename + "_FBC_vae"))
			self.vae_optimizer.load_state_dict(torch.load(filename + "_FBC_vae_optimizer"))
