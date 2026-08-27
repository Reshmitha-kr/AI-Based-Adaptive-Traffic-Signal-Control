import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the Neural Network for the Q-function
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Define the Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Use deque for efficient memory management
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Create the policy and target networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.update_steps = 0

    def remember(self, state, action, reward, next_state):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore: choose a random action
        
        # Exploit: choose the best action from Q-values
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size):
        """Train the network using a batch of experiences from memory"""
        if len(self.memory) < batch_size:
            return # Not enough memory to train
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Get current Q value from the policy network
            current_q = self.policy_net(state)[0][action]
            
            # Get target Q value using the Bellman equation and target network
            with torch.no_grad():
                target_q = reward + self.gamma * torch.max(self.target_net(next_state))

            # Calculate loss and perform backpropagation
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay epsilon after each training session
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Periodically update the target network
        self.update_steps += 1
        if self.update_steps % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())