import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class BCNetwork(nn.Module):
    """Neural Network policy for Behavior Cloning."""
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(BCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BCAgent:
    """Behavior Cloning Agent that learns from expert demonstrations."""
    def __init__(self, input_dims, n_actions, dataset=None, batch_size=64, epochs=100, lr=0.003, hidden_layers=64):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = dataset

        self.policy = BCNetwork(input_dims, hidden_layers, hidden_layers, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def shuffle_dataset(self):
        if self.dataset is None:
            return
        
        indices = np.arange(self.dataset['observations'].shape[0])
        np.random.shuffle(indices)
        self.dataset['observations'] = self.dataset['observations'][indices]
        self.dataset['actions'] = self.dataset['actions'][indices]

    def learn(self, verbose=True):
        if self.dataset is None:
            print("No dataset provided to learn from.")
            return
            
        for e in range(self.epochs):
            self.shuffle_dataset()
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, self.dataset['observations'].shape[0], self.batch_size):
                obs_batch = self.dataset['observations'][i:i+self.batch_size]
                act_batch = self.dataset['actions'][i:i+self.batch_size]

                obs_tensor = T.tensor(obs_batch, dtype=T.float).to(self.policy.device)
                act_tensor = T.tensor(act_batch, dtype=T.long).to(self.policy.device)

                self.optimizer.zero_grad()
                logits = self.policy.forward(obs_tensor)
                loss = self.loss_fn(logits, act_tensor)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            if verbose and (e + 1) % 10 == 0:
                print(f"Epoch {e+1}/{self.epochs}, Loss: {epoch_loss/num_batches:.4f}")

    def save(self, path):
        T.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(T.load(path))

    def predict(self, obs):
        obs_tensor = T.tensor(obs, dtype=T.float).unsqueeze(0).to(self.policy.device)
        self.policy.eval()
        with T.no_grad():
            logits = self.policy.forward(obs_tensor)
        self.policy.train()
        action = T.argmax(logits).item()
        return action
