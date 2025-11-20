import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections
import numpy as np
from model import Network
from replay_buffer import ReplayBuffer
from collections import deque




class Agent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.n_actions = env.action_space.n
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.batch_size = 32
        self.buffer_size = 100000 # Large buffer
        self.target_update_freq = 1000
        
        # Networks
        self.policy_net = Network((4, 84, 84), self.n_actions).to(device)
        self.target_net = Network((4, 84, 84), self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(self.buffer_size)
        self.steps = 0

    def act(self, state):
        # Epsilon Greedy
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Fixed syntax: torch.no_grad() needs parentheses
        with torch.no_grad():
            state = state.unsqueeze(0) 
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample Batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Max Q(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Target Net
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay