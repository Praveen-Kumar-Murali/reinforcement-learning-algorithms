import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ReplayBuffer:
    """Experience Replay Buffer for Deep Q-Learning"""
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def get_batch(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class DeepQNetwork(nn.Module):
    """Standard Deep Q-Network"""
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def update_weights(self, q_network, soft=True, tau=0.001):
        if soft:
            for target_param, param in zip(self.parameters(), q_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        else:
            self.load_state_dict(q_network.state_dict())

class DuelingDeepQNetwork(nn.Module):
    """Dueling Network Architecture for Deep Q-Learning"""
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.value_stream = nn.Linear(self.fc2_dims, 1)
        self.advantage_stream = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def update_weights(self, q_network, soft=True, tau=0.001):
        if soft:
            for target_param, param in zip(self.parameters(), q_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        else:
            self.load_state_dict(q_network.state_dict())

class DQNAgent:
    """
    Unified DQN Agent supporting Basic DQN, DDQN (Double DQN), 
    and Dueling DQN through toggles.
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, 
                 use_target_network=True, use_double_dqn=False, use_dueling=False):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.use_target_network = use_target_network
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        
        self.replay_buffer = ReplayBuffer(max_mem_size, input_dims)
        
        if self.use_dueling:
            self.Q_eval = DuelingDeepQNetwork(input_dims, 256, 256, n_actions)
            if self.use_target_network:
                self.Q_target = DuelingDeepQNetwork(input_dims, 256, 256, n_actions)
                self.Q_target.load_state_dict(self.Q_eval.state_dict())
                self.Q_target.eval()
        else:
            self.Q_eval = DeepQNetwork(input_dims, 256, 256, n_actions)
            if self.use_target_network:
                self.Q_target = DeepQNetwork(input_dims, 256, 256, n_actions)
                self.Q_target.load_state_dict(self.Q_eval.state_dict())
                self.Q_target.eval()
                
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
    def choose_action(self, observation, deterministic=False):
        if np.random.random() > self.epsilon or deterministic:
            state = T.tensor(observation, dtype=T.float).to(self.Q_eval.device).unsqueeze(0)
            self.Q_eval.eval()
            with T.no_grad():
                q_values = self.Q_eval.forward(state)
            self.Q_eval.train()
            action = T.argmax(q_values).item()
        else:
            action = np.random.choice(self.action_space)  
        return action
        
    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return
            
        self.optimizer.zero_grad()
        
        states, actions, rewards, new_states, dones = self.replay_buffer.get_batch(self.batch_size)
        
        state_batch = T.tensor(states).to(self.Q_eval.device)
        new_state_batch = T.tensor(new_states).to(self.Q_eval.device)
        action_batch = T.tensor(actions).to(self.Q_eval.device)
        reward_batch = T.tensor(rewards).to(self.Q_eval.device)
        terminal_batch = T.tensor(dones).to(self.Q_eval.device)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        
        if self.use_double_dqn and self.use_target_network:
            q_eval_next = self.Q_eval.forward(new_state_batch)
            next_best_action = T.argmax(q_eval_next, dim=1)
            q_next = self.Q_target.forward(new_state_batch)
            q_next = q_next[batch_index, next_best_action]
        elif self.use_target_network:
            q_next = self.Q_target.forward(new_state_batch)
            q_next = T.max(q_next, dim=1)[0]
        else:
            q_next = self.Q_eval.forward(new_state_batch)
            q_next = T.max(q_next, dim=1)[0]
            
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next
        
        loss = self.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.eps_end:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_end
            
        if self.use_target_network:
            self.Q_target.update_weights(self.Q_eval, soft=True, tau=0.001)
