import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim, n_actions, hidden_layers_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_layers_dim)
        self.fc2 = nn.Linear(hidden_layers_dim, hidden_layers_dim)
        self.fc3 = nn.Linear(hidden_layers_dim, n_actions)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, observation_dim, hidden_layers_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_layers_dim)
        self.fc2 = nn.Linear(hidden_layers_dim, hidden_layers_dim)
        self.fc3 = nn.Linear(hidden_layers_dim, 1)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VPGAgent:
    """Vanilla Policy Gradient (REINFORCE) Agent with optional Baseline."""
    def __init__(self, observation_dim, n_actions, hidden_layers_dim=128, 
                 lr=1e-2, lr_v=1e-2, gamma=0.99, use_baseline=True):
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        self.pi = PolicyNetwork(observation_dim, n_actions, hidden_layers_dim)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=lr)
        
        if self.use_baseline:
            self.baseline = ValueNetwork(observation_dim, hidden_layers_dim)
            self.optimizer_v = optim.Adam(self.baseline.parameters(), lr=lr_v)
            self.loss_v_fn = nn.MSELoss()
        
    def get_policy(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.pi.device)
        logits = self.pi(obs_tensor)
        return Categorical(logits=logits)

    def get_action(self, obs):
        dist = self.get_policy(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob

    def discounted_future_reward(self, rewards):
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
            
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.pi.device)
        # Normalize
        if len(discounted_rewards) > 1:
            mean = discounted_rewards.mean()
            std = discounted_rewards.std() + 1e-8
            discounted_rewards = (discounted_rewards - mean) / std
            
        return discounted_rewards
    
    def update_value_function(self, states, returns):
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.baseline.device)
        # Value predictions
        values = self.baseline(states_tensor).squeeze()
        
        loss_v = self.loss_v_fn(values, returns)
        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()
        
        return values.detach()

    def learn(self, fut_rew, logprobs, states=None):
        if self.use_baseline and states is not None:
            values = self.update_value_function(states, fut_rew)
            advantage = fut_rew - values
        else:
            advantage = fut_rew
            
        logprobs_tensor = torch.stack(logprobs).to(self.pi.device)
        loss_pi = - (logprobs_tensor * advantage).mean()
        
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()
