import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import sys

SCALE_DIAG_MIN_MAX = (-20, 2)
EPS = 1e-6

def squashing_func(sample, logp):
    squashed_action = nn.Tanh()(sample)
    squashed_action_logp = logp - torch.sum(torch.log(1 - squashed_action ** 2 + EPS), dim=1)
    return squashed_action, squashed_action_logp

def atanh(x):
    return torch.log1p(2 * x / (1 - x)) / 2

class GaussianReverseBC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, entropy_weight=0.5, hidden_dim=256, replay_buffer=None):
        super(GaussianReverseBC, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.entropy_weight = entropy_weight

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer_mu = nn.Linear(hidden_dim, action_dim)
        self.layer_log_std = nn.Linear(hidden_dim, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters())
        if replay_buffer is None:
            self.standardizer = None
        else:
            self.standardizer = replay_buffer.standardizer
        self.to(device)
    
    def main(self, state):
        h = self.layers(state)
        mean = self.layer_mu(h)
        log_std = self.layer_log_std(h)
        std = torch.exp(torch.clamp(log_std, min=SCALE_DIAG_MIN_MAX[0], max=SCALE_DIAG_MIN_MAX[1]))
        return mean, std
    
    def forward(self, state):
        mean, std = self.main(state)
        cov = torch.diag_embed(std)
        dist = D.MultivariateNormal(mean, cov)

        sampled_action = dist.sample()
        sampled_action_logp = dist.log_prob(sampled_action)
        squashed_action, squashed_action_logp = squashing_func(sampled_action, sampled_action_logp)
        deterministic_action, _ = squashing_func(mean, dist.log_prob(mean))

        return deterministic_action, squashed_action, squashed_action_logp, dist
    
    def select_action(self, next_state, deterministic=False):
        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).to(self.device)
            next_state = self.standardizer(next_state)
            deterministic_action, squashed_action, squashed_action_logp, dist = self.forward(next_state)
            if deterministic:
                return deterministic_action.cpu().data.numpy()
            else:
                return squashed_action.cpu().data.numpy()

    def nlogp(self, dist, action):
        before_squashed_action = atanh(torch.clamp(action, min=-1+EPS, max=1-EPS))
        log_likelihood = dist.log_prob(before_squashed_action)
        log_likelihood = log_likelihood - torch.sum(torch.log(1 - action ** 2 + EPS), dim=1)
        return -torch.mean(log_likelihood)
    
    def train(self, replay_buffer, iterations, batch_size):
        self.standardizer = replay_buffer.standardizer
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, weight, not_done = replay_buffer.sample(batch_size)

            next_state = self.standardizer(next_state)
            deterministic_action, squashed_action, squashed_action_logp, dist = self.forward(next_state)

            dist_loss = self.nlogp(dist, action)
            entropy_loss = -torch.mean(dist.entropy())

            # print(dist_loss, entropy_loss)
            loss = dist_loss + self.entropy_weight * entropy_loss
            if it % 5000 == 0:
                print('[BC] Step: {} | Loss: {} | Dist Loss: {} | Entropy Loss: {}'.format(it, loss.data, dist_loss.data, entropy_loss.data))
                sys.stdout.flush()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('[BC] Loss: {} | Dist Loss: {} | Entropy Loss: {}'.format(loss.data, dist_loss.data, entropy_loss.data))


    def save(self, filename):
        torch.save(self.state_dict(), filename + "_RBC_gaussian")
        torch.save(self.optimizer.state_dict(), filename + "_RBC_gaussian_optimizer")

    def load(self, filename):
        if not torch.cuda.is_available():
            self.load_state_dict(torch.load(filename + "_RBC_gaussian", map_location=torch.device('cpu')))
            self.optimizer.load_state_dict(torch.load(filename + "_RBC_gaussian_optimizer", map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(filename + "_RBC_gaussian"))
            self.optimizer.load_state_dict(torch.load(filename + "_RBC_gaussian_optimizer"))