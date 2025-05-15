import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import pygame
import sys


class Agent(nn.Module):
    def __init__(self, input_size, output_size):
        super(Agent, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state)
                action = torch.argmax(q_values).item()
            return action
    
    def train_model(self, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.forward(states).gather(1, actions)
        next_q_values = self.forward(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()